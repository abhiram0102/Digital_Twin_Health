"""
Physiological Signal Encoder.
Encodes time-series vitals + wearable signals using either
Temporal Convolutional Network (TCN) or Transformer.
Inputs: HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, activity_steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# ─── Temporal Convolution Network ────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal convolution (no future leakage)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        return self.conv(x)[:, :, :x.size(-1)]  # trim future


class TCNResidualBlock(nn.Module):
    """Dilated causal residual block for TCN."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class TCNEncoder(nn.Module):
    """Multi-scale Temporal Convolutional Network for physiological signals."""

    def __init__(self, input_dim=8, hidden_dim=128, embed_dim=256,
                 num_levels=6, kernel_size=3, dropout=0.1):
        super().__init__()
        channels = [hidden_dim] * num_levels
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(TCNResidualBlock(in_ch, ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, embed_dim)"""
        x = x.permute(0, 2, 1)              # (B, D, T)
        x = self.net(x)                      # (B, hidden, T)
        x = self.pool(x).squeeze(-1)        # (B, hidden)
        return self.proj(x)


# ─── Transformer Encoder ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class PhysioTransformerEncoder(nn.Module):
    """
    Transformer-based physiological signal encoder.
    Handles arbitrary-length time series with learnable temporal aggregation.
    """

    def __init__(self, input_dim=8, hidden_dim=128, embed_dim=256,
                 seq_len=168, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        # Feature normalization per signal
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=seq_len + 1, dropout=dropout)

        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Temporal importance weights (learned)
        self.time_attn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, input_dim) — physiological signals over time
        mask: (B, T) bool — True where data is missing
        Returns: (B, embed_dim)
        """
        B = x.size(0)
        x = self.input_norm(x)
        x = F.gelu(self.input_proj(x))         # (B, T, hidden)
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)          # (B, T+1, hidden)

        # Adjust mask for CLS token
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # CLS token output + attention-weighted mean
        cls_out = x[:, 0]                       # (B, hidden)
        seq_out = x[:, 1:]                      # (B, T, hidden)
        attn_w = self.time_attn(seq_out)        # (B, T, 1)
        weighted = (attn_w * seq_out).sum(1)    # (B, hidden)

        combined = cls_out + weighted
        return self.proj(combined)


class PhysioSignalEncoder(nn.Module):
    """
    Unified physiological signal encoder.
    Supports TCN or Transformer backbone.
    Also handles missing signals via learned imputation.
    """

    def __init__(self, input_dim=8, hidden_dim=128, embed_dim=256,
                 seq_len=168, num_heads=4, num_layers=3,
                 dropout=0.1, use_tcn=False):
        super().__init__()
        self.use_tcn = use_tcn
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Missing value imputation: learnable fill values per channel
        self.missing_fill = nn.Parameter(torch.zeros(1, 1, input_dim))
        # Missing value indicator embedding
        self.missing_indicator = nn.Linear(input_dim, input_dim)

        if use_tcn:
            self.encoder = TCNEncoder(input_dim, hidden_dim, embed_dim,
                                      dropout=dropout)
        else:
            self.encoder = PhysioTransformerEncoder(
                input_dim, hidden_dim, embed_dim, seq_len, num_heads, num_layers, dropout
            )

    def forward(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        missing_mask: (B, T, input_dim) bool, True = missing
        Returns: (B, embed_dim)
        """
        if missing_mask is not None:
            # Replace NaN/missing with learned fill
            fill = self.missing_fill.expand_as(x)
            x = torch.where(missing_mask, fill, x)

        return self.encoder(x)
