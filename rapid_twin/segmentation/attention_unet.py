"""
Attention U-Net for Retinal Vessel Segmentation.

Architecture: encoder-decoder with attention gates + deep supervision.
Losses: BCE + Dice + clDice (topology-preserving) + Tversky (thin-vessel) + boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Soft attention gate on skip connections — focuses on vessel regions."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True),
                                  nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True),
                                  nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True),
                                  nn.BatchNorm2d(1),
                                  nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0, use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(F_g=in_ch // 2, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.use_attention:
            skip = self.attn(x, skip)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.head = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.head(x), size=target_size, mode='bilinear', align_corners=False)


# ─── Attention U-Net ──────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net with deep supervision.

    Input:  (B, in_channels, H, W)
    Output: dict with 'segmentation', 'logits', optional 'deep_supervision' list
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: List[int] = None, dropout: float = 0.1,
                 use_attention: bool = True, deep_supervision: bool = True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.enc = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(ConvBlock(ch, f, dropout=dropout if ch > in_channels else 0))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout=dropout)

        self.dec = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev:
            self.dec.append(UpBlock(in_ch, f, f, dropout=dropout, use_attention=use_attention))
            in_ch = f

        self.final = nn.Conv2d(features[0], out_channels, 1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                DeepSupervisionHead(rev[i], out_channels)
                for i in range(1, min(4, len(rev)))
            ])

    def forward(self, x: torch.Tensor) -> dict:
        target_size = (x.shape[2], x.shape[3])

        skips = []
        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        dec_outs = []
        for i, dec_block in enumerate(self.dec):
            x = dec_block(x, skips[-(i + 1)])
            dec_outs.append(x)

        logits = self.final(x)
        result = {"segmentation": torch.sigmoid(logits), "logits": logits}

        if self.deep_supervision and self.training:
            ds_outputs = []
            for i, head in enumerate(self.ds_heads):
                ds_logits = head(dec_outs[i + 1], target_size)
                ds_outputs.append(ds_logits)
            result["deep_supervision"] = ds_outputs

        return result


# ─── Loss functions ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1.0 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss — generalised Dice with asymmetric FP/FN weighting.

    alpha < 0.5 penalises false positives less; beta > 0.5 penalises
    false negatives more, recovering thin vessels missed by plain Dice.
    Recommended for retinal vessel segmentation: alpha=0.3, beta=0.7.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class CLDiceLoss(nn.Module):
    """
    clDice — topology-preserving loss based on soft vessel skeletons.

    Reference: Shit et al. (2021) "clDice — A Novel Topology-Preserving
    Loss Function for Tubular Structure Segmentation", CVPR 2021.

    The soft skeleton is approximated via iterative morphological erosion
    using max-pooling (differentiable, GPU-compatible, no scipy required).
    """

    def __init__(self, iters: int = 3, smooth: float = 1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def _soft_erode(self, x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def _soft_dilate(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(x))

    def _soft_skel(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        skel = F.relu(x - self._soft_open(x))
        for _ in range(self.iters):
            x = self._soft_erode(x)
            delta = F.relu(x - self._soft_open(x))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, pred_sigmoid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        skel_pred = self._soft_skel(pred_sigmoid)
        skel_tgt  = self._soft_skel(target)

        prec_num = (skel_pred * target).sum() + self.smooth
        prec_den = skel_pred.sum() + self.smooth
        sens_num = (skel_tgt * pred_sigmoid).sum() + self.smooth
        sens_den = skel_tgt.sum() + self.smooth

        tprec = prec_num / prec_den
        tsens = sens_num / sens_den
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-8)


class BoundaryLoss(nn.Module):
    """
    Boundary (distance-transform) loss.

    Uses a precomputed distance map of the GT boundary as a weighting field,
    penalising predictions far from real vessel boundaries.

    When no distance map is supplied (default training setup without
    pre-computation), falls back to boundary-BCE via morphological dilation:
    the boundary mask is obtained as GT_dilated XOR GT.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def _boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        return (dilated - target).clamp(0, 1)

    def forward(self, pred_sigmoid: torch.Tensor,
                target: torch.Tensor,
                distance_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if distance_map is not None:
            return (pred_sigmoid * distance_map).mean()

        boundary = self._boundary_mask(target)
        if boundary.sum() < self.smooth:
            return torch.zeros(1, device=pred_sigmoid.device).squeeze()
        pred_on_boundary = (pred_sigmoid * boundary).sum()
        gt_on_boundary   = boundary.sum()
        return 1.0 - (pred_on_boundary + self.smooth) / (gt_on_boundary + self.smooth)


# ─── Combined segmentation loss ───────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Full topology-aware segmentation loss.

    Combines:
      - Weighted BCE (handles class imbalance)
      - Dice (volumetric overlap)
      - clDice (vessel topology / centerline continuity)
      - Tversky (thin-vessel recall, FN-heavy)
      - Boundary (edge precision)
      - Deep supervision (auxiliary multi-scale heads)

    Default weights are calibrated for retinal vessel segmentation.
    All weights are normalised so they sum to 1.0 for the main loss.
    """

    def __init__(
        self,
        pos_weight:      float = 5.0,
        bce_weight:      float = 0.25,
        dice_weight:     float = 0.25,
        cldice_weight:   float = 0.25,
        tversky_weight:  float = 0.20,
        boundary_weight: float = 0.05,
        ds_weight:       float = 0.30,
        tversky_alpha:   float = 0.30,
        tversky_beta:    float = 0.70,
        cldice_iters:    int   = 3,
    ):
        super().__init__()
        self.bce_w      = bce_weight
        self.dice_w     = dice_weight
        self.cldice_w   = cldice_weight
        self.tversky_w  = tversky_weight
        self.boundary_w = boundary_weight
        self.ds_weight  = ds_weight

        self.bce      = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]))
        self.dice     = DiceLoss()
        self.tversky  = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.cldice   = CLDiceLoss(iters=cldice_iters)
        self.boundary = BoundaryLoss()

    def _component_losses(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict:
        pred_sig = torch.sigmoid(logits)
        bce_l      = self.bce(logits, target)
        dice_l     = self.dice(logits, target)
        tversky_l  = self.tversky(logits, target)
        cldice_l   = self.cldice(pred_sig, target)
        boundary_l = self.boundary(pred_sig, target)

        total = (self.bce_w      * bce_l
                 + self.dice_w      * dice_l
                 + self.tversky_w   * tversky_l
                 + self.cldice_w    * cldice_l
                 + self.boundary_w  * boundary_l)

        return {
            "bce":      bce_l,
            "dice":     dice_l,
            "tversky":  tversky_l,
            "cldice":   cldice_l,
            "boundary": boundary_l,
            "main":     total,
        }

    def forward(self, outputs: dict, target: torch.Tensor) -> dict:
        logits = outputs["logits"]
        components = self._component_losses(logits, target)
        total = components["main"]

        ds_loss = torch.zeros(1, device=logits.device).squeeze()
        if "deep_supervision" in outputs and outputs["deep_supervision"]:
            for ds_logits in outputs["deep_supervision"]:
                c = self._component_losses(ds_logits, target)
                ds_loss = ds_loss + c["main"]
            ds_loss = ds_loss / len(outputs["deep_supervision"])
            total   = total + self.ds_weight * ds_loss

        return {
            "total":    total,
            "bce":      components["bce"],
            "dice":     components["dice"],
            "tversky":  components["tversky"],
            "cldice":   components["cldice"],
            "boundary": components["boundary"],
            "ds":       ds_loss,
        }
