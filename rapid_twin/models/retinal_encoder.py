"""
Retinal Structure Encoder — TIMM ConvNeXt backbone + FPN + Swin-style head.

Architecture:
  timm.create_model("convnext_tiny", pretrained=True)
    → Feature Pyramid Network (lateral 1×1 + top-down 3×3)
    → SwinTransformerHead on deepest FPN level
    → Multi-scale global-average pool of lower FPN levels
    → Fused output projection to fixed embed_dim

Design contracts:
  Input:  (B, 3, H, W)  float32  [0, 1]  preprocessed fundus image
  Output: (B, embed_dim) float32

All tensor operations are NCHW throughout; permute only inside Swin window blocks.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn(
        "timm not installed. RetinalStructureEncoder will use a lightweight fallback. "
        "Install with: pip install timm",
        RuntimeWarning,
    )


# ─── FPN helper ───────────────────────────────────────────────────────────────

class FPNLateralBlock(nn.Module):
    """Lateral + top-down merge block for FPN."""

    def __init__(self, in_ch: int, fpn_ch: int):
        super().__init__()
        self.lateral = nn.Conv2d(in_ch, fpn_ch, kernel_size=1, bias=False)
        self.refine   = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, fpn_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor,
                top_down: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.lateral(x)
        if top_down is not None:
            top_down = F.interpolate(top_down, size=out.shape[-2:],
                                     mode="nearest")
            out = out + top_down
        return self.refine(out)


# ─── Swin-style head ──────────────────────────────────────────────────────────

class _WindowAttn(nn.Module):
    """Multi-head window self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv   = nn.Linear(dim, dim * 3)
        self.proj  = nn.Linear(dim, dim)
        self.drop  = nn.Dropout(dropout)

        # Relative position bias table: (2W-1)² × nH
        self.rel_bias = nn.Embedding((2 * window_size - 1) ** 2, num_heads)
        coords = torch.arange(window_size)
        grid   = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)
        rel    = grid[:, :, None] - grid[:, None, :]
        rel[0] += window_size - 1
        rel[1] += window_size - 1
        rel[0] *= 2 * window_size - 1
        self.register_buffer("rel_idx", rel.sum(0).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        H = self.num_heads
        D = C // H

        qkv = self.qkv(x).reshape(B_, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.rel_bias(self.rel_idx).view(N, N, H).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        attn = self.drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinGlobalHead(nn.Module):
    """
    Lightweight Swin Transformer head.
    Consumes NCHW feature map, returns (B, out_dim) embedding.
    """

    def __init__(self, in_dim: int, out_dim: int, window_size: int = 7,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(out_dim),
                "attn":  _WindowAttn(out_dim, window_size, num_heads, dropout),
                "norm2": nn.LayerNorm(out_dim),
                "ffn":   nn.Sequential(
                    nn.Linear(out_dim, out_dim * 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim),
                ),
            }))
        self.norm = nn.LayerNorm(out_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # ── window helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

    @staticmethod
    def _reverse(x: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
        C  = x.shape[-1]
        nH = H // ws
        nW = W // ws
        B  = x.shape[0] // (nH * nW)
        x  = x.view(B, nH, nW, ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim, H, W) → (B, out_dim)"""
        x = self.proj_in(x)
        B, C, H, W = x.shape

        ws = self.window_size
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)  # NHWC
        for layer in self.layers:
            shortcut = x
            wins     = self._partition(layer["norm1"](x), ws)
            wins     = layer["attn"](wins)
            x        = shortcut + self._reverse(wins, ws, Hp, Wp)
            x        = x + layer["ffn"](layer["norm2"](x))
        x = self.norm(x)
        x = x[:, :H, :W, :].permute(0, 3, 1, 2)   # NCHW, crop padding
        return self.pool(x).flatten(1)              # (B, out_dim)


# ─── Lightweight fallback when TIMM is not available ─────────────────────────

class _FallbackBackbone(nn.Module):
    """Simple 4-stage CNN fallback (no pretrained weights)."""

    STAGE_DIMS = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        chs = self.STAGE_DIMS
        self.stages = nn.ModuleList()
        ch = in_channels
        for out_ch in chs:
            self.stages.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            ))
            ch = out_ch
        self.pool = nn.MaxPool2d(2)
        self.out_dims = chs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.stages) - 1:
                x = self.pool(x)
        return feats


# ─── Main Encoder ─────────────────────────────────────────────────────────────

class RetinalStructureEncoder(nn.Module):
    """
    Retinal image encoder.

    Backbone: timm ConvNeXt-Tiny (pretrained on ImageNet-1K by default).
    Head:     FPN → SwinGlobalHead on C4 + avg-pool of C2, C3 → projection.
    Output:   (B, embed_dim) — strict contract enforced by LayerNorm.

    Args:
        in_channels:  must be 3 (RGB fundus image)
        embed_dim:    output embedding size (default 256)
        backbone:     timm model name (default "convnext_tiny")
        pretrained:   load ImageNet weights (default True)
        fpn_dim:      internal FPN channel count (default 256)
        window_size:  Swin window size for global head (default 7)
        dropout:      dropout probability (default 0.1)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim:   int = 256,
        backbone:    str = "convnext_tiny",
        pretrained:  bool = True,
        fpn_dim:     int = 256,
        window_size: int = 7,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Backbone ──────────────────────────────────────────────────────────
        if TIMM_AVAILABLE:
            # out_indices are 0-based and refer to the 4 ConvNeXt stages
            # (stem=0, stage1=1, stage2=2, stage3=3 for convnext_tiny → channels [96,192,384,768])
            # Using (1,2,3,4) would request a non-existent 5th index → IndexError.
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),   # 4 stages: strides 4, 8, 16, 32
                in_chans=in_channels,
            )
            backbone_dims = self.backbone.feature_info.channels()   # e.g. [96,192,384,768]
        else:
            self.backbone  = _FallbackBackbone(in_channels)
            backbone_dims  = self.backbone.out_dims

        # ── Feature Pyramid Network ───────────────────────────────────────────
        # Lateral connections for C2, C3, C4 (bottom-up order)
        self.fpn = nn.ModuleList([
            FPNLateralBlock(d, fpn_dim) for d in backbone_dims
        ])

        # ── Global head on deepest (C4) FPN level ─────────────────────────────
        self.global_head = SwinGlobalHead(
            in_dim=fpn_dim, out_dim=embed_dim,
            window_size=window_size, dropout=dropout,
        )

        # ── Multi-scale pool over shallower FPN levels ────────────────────────
        # Pools each to 4×4 → flatten → concat → project
        num_lateral = len(backbone_dims)
        ms_flat_dim = fpn_dim * (num_lateral - 1) * 4 * 4
        self.ms_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.ms_proj = nn.Sequential(
            nn.Linear(ms_flat_dim, embed_dim * 2, bias=False),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

        # ── Final fusion ──────────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

        self._init_new_layers()

    def _init_new_layers(self):
        """Initialise non-pretrained layers (FPN, heads, projections)."""
        for m in [self.fpn, self.global_head, self.ms_proj, self.out_proj]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.trunc_normal_(p, std=0.02)
                elif p.dim() == 1:
                    nn.init.zeros_(p)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze pretrained backbone for fine-tuning control."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  float32  [0, 1]

        Returns:
            (B, embed_dim) float32
        """
        # ── Extract multi-scale backbone features ─────────────────────────────
        backbone_feats: List[torch.Tensor] = self.backbone(x)  # C1..C4 NCHW

        # ── Build FPN top-down pathway ────────────────────────────────────────
        fpn_feats: List[Optional[torch.Tensor]] = [None] * len(backbone_feats)
        # Top level — no top-down input
        fpn_feats[-1] = self.fpn[-1](backbone_feats[-1], top_down=None)
        # Remaining levels (top-down merge)
        for i in range(len(backbone_feats) - 2, -1, -1):
            fpn_feats[i] = self.fpn[i](backbone_feats[i], top_down=fpn_feats[i + 1])

        # ── Global embedding from deepest FPN level ───────────────────────────
        global_emb = self.global_head(fpn_feats[-1])   # (B, embed_dim)

        # ── Multi-scale embedding from shallower FPN levels ───────────────────
        ms_parts = []
        for feat in fpn_feats[:-1]:
            pooled = self.ms_pool(feat)         # (B, fpn_dim, 4, 4)
            ms_parts.append(pooled.flatten(1))  # (B, fpn_dim * 16)
        ms_cat = torch.cat(ms_parts, dim=1)     # (B, fpn_dim*(n-1)*16)
        ms_emb = self.ms_proj(ms_cat)           # (B, embed_dim)

        # ── Fuse and return ───────────────────────────────────────────────────
        fused = torch.cat([global_emb, ms_emb], dim=1)  # (B, embed_dim*2)
        return self.out_proj(fused)                      # (B, embed_dim)
