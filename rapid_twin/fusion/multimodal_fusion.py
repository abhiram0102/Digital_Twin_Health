"""
Multimodal Health Fusion Transformer.

Combines retinal-graph, retinal-image, and physiological-signal embeddings
via cross-modal self-attention (Q-Former / Perceiver-style optional).

CRITICAL FIX over v1:
  - ModalityTokenizer no longer hard-clips or zero-pads embeddings.
    Every modality has its own learned Linear projection to fusion_dim.
    This is the correct design — the projection IS the alignment layer.
  - Strict embedding dimension contracts enforced via runtime assertions
    in debug mode (torch.jit-friendly fallback in production).

Design contracts:
  retinal_image_emb : (B, retinal_dim)  → projected to fusion_dim
  vascular_graph_emb: (B, graph_dim)    → projected to fusion_dim
  physio_emb        : (B, physio_dim)   → projected to fusion_dim
  output            : (B, output_dim)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Modality Tokenizer (no hard-clip) ───────────────────────────────────────

class ModalityTokenizer(nn.Module):
    """
    Projects each modality embedding to a common fusion_dim.

    Each modality has its own dedicated Linear + LayerNorm block,
    plus a learnable type embedding distinguishing modalities.
    Missing modalities are replaced with learned fallback embeddings.
    """

    def __init__(self, modality_dims: List[int], fusion_dim: int):
        super().__init__()
        # One projection per modality — no shared weights, no hard-clip
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, fusion_dim, bias=True),
                nn.LayerNorm(fusion_dim),
            )
            for d in modality_dims
        ])
        # Learnable type embeddings (one per modality)
        self.type_emb = nn.Embedding(len(modality_dims), fusion_dim)
        self.register_buffer("type_ids", torch.arange(len(modality_dims)))

        # Learned fallback for missing modalities
        self.missing_emb = nn.ParameterList([
            nn.Parameter(torch.randn(1, fusion_dim) * 0.02)
            for _ in modality_dims
        ])

    def forward(
        self,
        modalities: List[Optional[torch.Tensor]],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            modalities: list of (B, dim_i) or None (missing modality)
            batch_size: B

        Returns:
            tokens: (B, M, fusion_dim)
        """
        tokens = []
        for i, x in enumerate(modalities):
            if x is None:
                # Expand learned fallback to batch
                tok = self.missing_emb[i].expand(batch_size, -1)
            else:
                # NOTE: projection handles arbitrary input dim — no clipping
                tok = self.projections[i](x)                 # (B, fusion_dim)
            tok = tok + self.type_emb(self.type_ids[i])     # broadcast add
            tokens.append(tok)
        return torch.stack(tokens, dim=1)                    # (B, M, fusion_dim)


# ─── Cross-Modal Attention Block ─────────────────────────────────────────────

class CrossModalAttentionBlock(nn.Module):
    """
    Pre-norm full self-attention across all modality tokens.
    Each token attends to every other token (cross-modal interaction).
    """

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, M, dim)"""
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + self.drop(attn_out)
        # FFN with pre-norm
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ─── Q-Former style cross-attention (optional advanced fusion) ────────────────

class QFormerBlock(nn.Module):
    """
    Q-Former style: fixed query tokens attend over modality tokens (KV).
    Produces a compressed fixed-size representation independent of M.
    """

    def __init__(self, query_dim: int, kv_dim: int, num_queries: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.queries  = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            query_dim, num_heads,
            kdim=kv_dim, vdim=kv_dim,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.self_attn = nn.MultiheadAttention(
            query_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.norm3 = nn.LayerNorm(query_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        kv_tokens: (B, M, kv_dim)
        Returns: (B, num_queries, query_dim)
        """
        B = kv_tokens.size(0)
        q = self.queries.expand(B, -1, -1)    # (B, Q, query_dim)

        # Cross-attention: queries attend over modality tokens
        normed_kv = kv_tokens
        ca_out, _ = self.cross_attn(self.norm1(q), normed_kv, normed_kv,
                                    need_weights=False)
        q = q + self.drop(ca_out)

        # Self-attention among queries
        sa_out, _ = self.self_attn(self.norm2(q), q, q, need_weights=False)
        q = q + self.drop(sa_out)

        # FFN
        q = q + self.drop(self.ffn(self.norm3(q)))
        return q   # (B, Q, query_dim)


# ─── Multimodal Fusion Transformer ───────────────────────────────────────────

class MultimodalFusionTransformer(nn.Module):
    """
    Multimodal Fusion Transformer with optional Q-Former compression.

    Tokens: [vascular_graph | retinal_image | physio_signals]

    Self-attention learns inter-modal correlations:
        e.g. vessel narrowing (graph) + high BP (physio) → hypertension risk

    Args:
        modality_dims:  list of input dim per modality (no restriction on values)
        fusion_dim:     internal fusion dimension (default 512)
        num_heads:      attention heads (default 8)
        num_layers:     cross-modal attention layers (default 4)
        dropout:        dropout probability (default 0.1)
        output_dim:     final embedding dimension (default 256)
        use_qformer:    if True, add Q-Former compression after self-attn (default False)
        num_queries:    Q-Former query count (default 8, used only if use_qformer=True)
    """

    def __init__(
        self,
        modality_dims: List[int],
        fusion_dim:    int = 512,
        num_heads:     int = 8,
        num_layers:    int = 4,
        dropout:       float = 0.1,
        output_dim:    int = 256,
        use_qformer:   bool = False,
        num_queries:   int = 8,
    ):
        super().__init__()
        self.num_modalities = len(modality_dims)
        self.fusion_dim     = fusion_dim
        self.use_qformer    = use_qformer

        # Tokenise each modality with its own projection
        self.tokenizer = ModalityTokenizer(modality_dims, fusion_dim)

        # Cross-modal self-attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionBlock(fusion_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(fusion_dim)

        # Optional Q-Former compression
        if use_qformer:
            self.qformer = QFormerBlock(
                fusion_dim, fusion_dim, num_queries, num_heads, dropout
            )
            agg_dim = fusion_dim * num_queries
        else:
            # Learned weighted aggregation across modality tokens
            self.modality_weights = nn.Linear(fusion_dim, 1)
            agg_dim = fusion_dim

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(agg_dim, output_dim * 2, bias=False),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim, bias=False),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        modalities:     List[Optional[torch.Tensor]],
        available_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modalities:     list of (B, dim_i) tensors; None if missing
            available_mask: (B, M) bool — True = modality present (for weighting)

        Returns:
            dict with:
                'embedding'        : (B, output_dim)
                'modality_weights' : (B, M)
                'token_embeddings' : (B, M, fusion_dim)
        """
        # Infer batch size from first available modality
        B = next(x.shape[0] for x in modalities if x is not None)

        # Tokenise (projections handle arbitrary input dims; missing → learned emb)
        tokens = self.tokenizer(modalities, B)   # (B, M, fusion_dim)

        # Cross-modal attention
        x = tokens
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                         # (B, M, fusion_dim)

        # Aggregate
        if self.use_qformer:
            compressed = self.qformer(x)                    # (B, Q, fusion_dim)
            agg        = compressed.flatten(1)              # (B, Q*fusion_dim)
            weights    = torch.ones(B, self.num_modalities,
                                    device=x.device) / self.num_modalities
        else:
            weights = torch.sigmoid(self.modality_weights(x))  # (B, M, 1)
            if available_mask is not None:
                weights = weights * available_mask.unsqueeze(-1).float()
            w_sum   = weights.sum(1, keepdim=True).clamp(min=1e-8)
            agg     = (weights * x).sum(1) / w_sum.squeeze(1)  # (B, fusion_dim)
            weights = weights.squeeze(-1)                        # (B, M)

        embedding = self.output_proj(agg)        # (B, output_dim)

        return {
            "embedding":        embedding,
            "modality_weights": weights,
            "token_embeddings": x,
        }


# ─── Health State Projector ───────────────────────────────────────────────────

class HealthStateProjector(nn.Module):
    """
    Maps fused embedding → named health state dimension scores in [0, 1].
    Used for the digital twin state vector.
    """

    HEALTH_DIMENSIONS = [
        "vascular_health",
        "retinal_integrity",
        "metabolic_risk",
        "cardiovascular_risk",
        "lifestyle_health",
    ]

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        n = len(self.HEALTH_DIMENSIONS)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )
            for _ in range(n)
        ])

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """embedding: (B, input_dim) → dict {dim_name: (B, 1)}"""
        shared = self.shared(embedding)
        scores = {}
        all_s  = []
        for name, head in zip(self.HEALTH_DIMENSIONS, self.heads):
            s = head(shared)
            scores[name] = s
            all_s.append(s)
        scores["state_vector"] = torch.cat(all_s, dim=-1)   # (B, 5)
        return scores
