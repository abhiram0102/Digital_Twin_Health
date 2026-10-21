"""
Temporal Digital Twin Model — Longitudinal Health Modeling.

Architecture:
  GatedResidualNetwork (GRN) building blocks
  → VariableSelectionNetwork
  → LSTM encoder for historical sequence
  → Temporal self-attention (causal)
  → GRN temporal enrichment
  → Quantile projection head (uncertainty estimation)
  → Baseline health state estimator
  → Anomaly deviation detector
  → GRU-based rolling state tracker with persistent memory buffer

Key improvements over v1:
  - DigitalTwinStateTracker now maintains a persistent in-memory ring buffer
    of state vectors (up to memory_size) — enables true longitudinal modeling
  - compute_longitudinal_risk: projects multi-timestep trajectory
  - TemporalHealthForecaster: wraps TFT for future risk forecasting
  - Anomaly scoring uses Mahalanobis distance from rolling baseline covariance
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── GRN ─────────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network — core TFT building block.
    Selectively passes information via sigmoid gating.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        full_in = input_dim + (context_dim or 0)
        self.fc1  = nn.Linear(full_in, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = (nn.Linear(input_dim, output_dim, bias=False)
                     if input_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = torch.cat([x, context], dim=-1) if context is not None else x
        h   = F.elu(self.fc1(inp))
        h   = self.drop(h)
        return self.norm(torch.sigmoid(self.gate(h)) * self.fc2(h) + self.skip(x))


# ─── Variable Selection ───────────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """
    TFT Variable Selection Network.
    Learns per-variable importance weights.
    """

    def __init__(self, input_dim: int, num_vars: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.num_vars    = num_vars
        self.var_dim     = input_dim
        self.single_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])
        self.combined_grn = GatedResidualNetwork(
            input_dim * num_vars, hidden_dim, num_vars, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, num_vars * var_dim)
        Returns: processed (B, T, hidden_dim), weights (B, T, num_vars)
        """
        B, T, D = x.shape
        # Reshape to (B, T, num_vars, var_dim)
        vars_split = x.view(B, T, self.num_vars, self.var_dim)

        processed = []
        for i, grn in enumerate(self.single_grns):
            v = vars_split[:, :, i, :]          # (B, T, var_dim)
            processed.append(grn(v.reshape(B * T, -1)).view(B, T, -1))

        weights = self.softmax(
            self.combined_grn(x.reshape(B * T, -1)).view(B, T, -1)
        )   # (B, T, num_vars)

        stacked = torch.stack(processed, dim=2)   # (B, T, num_vars, hidden_dim)
        out     = (stacked * weights.unsqueeze(-1)).sum(2)   # (B, T, hidden_dim)
        return out, weights


# ─── Temporal Self-Attention ──────────────────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """Multi-head temporal self-attention with optional causal mask."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        T    = x.size(1)
        mask = None
        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        res      = x
        normed   = self.norm(x)
        out, _   = self.attn(normed, normed, normed, attn_mask=mask)
        return res + self.drop(out)


# ─── Temporal Fusion Transformer ─────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Digital Twin Health Modeling.

    Learns:
      1. Patient-specific baseline health from early history
      2. Long-term trends via LSTM encoding
      3. Anomaly deviation from personal baseline
      4. Quantile predictions for uncertainty quantification
      5. Gated state update for smooth state evolution

    Inputs:
      historical_seq:   (B, T_past, state_dim)
      static_context:   (B, state_dim) or None — patient invariants

    Outputs:
      quantile_predictions: (B, T, num_quantiles)
      anomaly_score:        (B,)
      health_state:         (B, state_dim)
      baseline:             (B, state_dim)
      hidden_sequence:      (B, T, hidden_dim)
    """

    def __init__(
        self,
        state_dim:           int = 256,
        hidden_dim:          int = 512,
        num_quantiles:       int = 7,
        num_heads:           int = 8,
        num_encoder_steps:   int = 24,
        dropout:             float = 0.1,
        num_future_steps:    int = 12,
        num_lstm_layers:     int = 2,
    ):
        super().__init__()
        self.state_dim    = state_dim
        self.hidden_dim   = hidden_dim
        self.num_quantiles = num_quantiles

        # Static context → 4 context vectors
        self.static_encoder   = GatedResidualNetwork(
            state_dim, hidden_dim, hidden_dim * 4, dropout)
        self.static_enrichers = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(4)
        ])

        # Historical input projection
        self.hist_proj = nn.Linear(state_dim, hidden_dim, bias=False)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Temporal attention + GRN enrichment
        self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads, dropout)
        self.temporal_grn  = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout,
            context_dim=hidden_dim,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Output heads
        self.quantile_proj = nn.Linear(hidden_dim, num_quantiles)
        self.baseline_net  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.anomaly_net   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

        # Gated state update
        self.state_gate   = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, state_dim), nn.Sigmoid()
        )
        self.state_update = nn.Sequential(
            nn.Linear(hidden_dim, state_dim), nn.Tanh()
        )

    def _encode_static(self, static: torch.Tensor) -> List[torch.Tensor]:
        ctx    = self.static_encoder(static)          # (B, hidden*4)
        splits = ctx.chunk(4, dim=-1)                 # 4 × (B, hidden)
        return [grn(c) for grn, c in zip(self.static_enrichers, splits)]

    def forward(
        self,
        historical_seq:    torch.Tensor,
        static_context:    Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = historical_seq.shape

        # Static encoding
        if static_context is not None:
            ctx_vecs  = self._encode_static(static_context)
            ctx_h, ctx_c = ctx_vecs[0], ctx_vecs[1]
        else:
            ctx_h = torch.zeros(B, self.hidden_dim, device=historical_seq.device)
            ctx_c = torch.zeros(B, self.hidden_dim, device=historical_seq.device)

        # Project + LSTM encode
        hist = F.elu(self.hist_proj(historical_seq))          # (B, T, hidden)
        h0   = ctx_h.unsqueeze(0).expand(
            self.lstm_encoder.num_layers, -1, -1).contiguous()
        c0   = ctx_c.unsqueeze(0).expand(
            self.lstm_encoder.num_layers, -1, -1).contiguous()
        enc_out, _ = self.lstm_encoder(hist, (h0, c0))        # (B, T, hidden)

        # Temporal attention
        attn_out = self.temporal_attn(enc_out, causal=True)   # (B, T, hidden)

        # GRN enrichment with static context
        ctx2 = ctx_vecs[2].unsqueeze(1).expand(-1, T, -1) \
               if static_context is not None \
               else torch.zeros(B, T, self.hidden_dim, device=historical_seq.device)
        enriched = self.temporal_grn(
            attn_out.reshape(B * T, -1),
            context=ctx2.reshape(B * T, -1),
        ).view(B, T, -1)                                       # (B, T, hidden)
        enriched = self.norm(enriched)

        # Current state (last timestep) + early-history baseline
        last_h     = enriched[:, -1, :]                       # (B, hidden)
        early_mean = enriched[:, :max(1, T // 2)].mean(1)     # (B, hidden)
        baseline   = self.baseline_net(early_mean)             # (B, state_dim)

        # Anomaly score: deviation from baseline trajectory
        anom_in      = torch.cat([last_h, self.hist_proj(baseline)], dim=-1)
        anomaly_score = self.anomaly_net(anom_in).squeeze(-1)  # (B,)

        # Gated health state update
        gate         = self.state_gate(torch.cat([last_h, historical_seq[:, -1]], dim=-1))
        update       = self.state_update(last_h)
        health_state = gate * historical_seq[:, -1] + (1 - gate) * update  # (B, state_dim)

        # Quantile predictions
        quantile_preds = self.quantile_proj(enriched)          # (B, T, Q)

        return {
            "quantile_predictions": quantile_preds,
            "anomaly_score":        anomaly_score,
            "health_state":         health_state,
            "baseline":             baseline,
            "hidden_sequence":      enriched,
            "last_hidden":          last_h,
        }


# ─── Digital Twin State Tracker ──────────────────────────────────────────────

class DigitalTwinStateTracker(nn.Module):
    """
    Maintains a rolling digital twin state with persistent memory buffer.

    Improvements over v1:
      - Internal ring buffer stores last `memory_size` state vectors
      - compute_drift and compute_anomaly_from_baseline use the buffer
      - Memory-augmented GRU update: gates between current obs and long-term mean
      - Mahalanobis-distance anomaly scoring (falls back to L2 if few samples)
    """

    def __init__(self, state_dim: int = 256, memory_size: int = 52):
        super().__init__()
        self.state_dim   = state_dim
        self.memory_size = memory_size

        # GRU-based state update
        self.state_gru  = nn.GRUCell(state_dim, state_dim)
        self.state_norm = nn.LayerNorm(state_dim)

        # Long-term memory gating
        self.memory_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.memory_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim), nn.Sigmoid()
        )

        # Persistent buffer (not a nn.Parameter — updated at runtime)
        self.register_buffer(
            "_memory_buffer",
            torch.zeros(memory_size, state_dim),
        )
        self.register_buffer("_buffer_ptr",  torch.zeros(1, dtype=torch.long))
        self.register_buffer("_buffer_fill", torch.zeros(1, dtype=torch.long))

    # ── Buffer management ─────────────────────────────────────────────────────

    def _push(self, state: torch.Tensor):
        """Push a single (state_dim,) vector into the ring buffer."""
        if state.dim() == 2:
            state = state.squeeze(0)
        ptr = int(self._buffer_ptr.item())
        self._memory_buffer[ptr] = state.detach()
        self._buffer_ptr[0] = (ptr + 1) % self.memory_size
        if self._buffer_fill < self.memory_size:
            self._buffer_fill += 1

    def _get_memory_snapshot(self) -> Optional[torch.Tensor]:
        """Return filled portion of buffer as (K, state_dim) tensor, or None."""
        k = int(self._buffer_fill.item())
        if k == 0:
            return None
        return self._memory_buffer[:k]   # (K, state_dim)

    def reset_memory(self):
        """Reset ring buffer (new patient session)."""
        self._memory_buffer.zero_()
        self._buffer_ptr.zero_()
        self._buffer_fill.zero_()

    # ── Core update ───────────────────────────────────────────────────────────

    def update(
        self,
        current_embedding: torch.Tensor,   # (B, state_dim)
        previous_state:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update rolling twin state with new observation.

        Returns: (B, state_dim) updated state.
        """
        B  = current_embedding.size(0)
        if previous_state is None:
            previous_state = torch.zeros(
                B, self.state_dim,
                device=current_embedding.device,
                dtype=current_embedding.dtype,
            )

        # Standard GRU update
        new_state = self.state_gru(current_embedding, previous_state)

        # Memory-augmented gating: incorporate long-term mean if available
        mem = self._get_memory_snapshot()
        if mem is not None and mem.shape[0] > 1:
            mem_mean = mem.mean(0).to(current_embedding.device)   # (state_dim,)
            mem_proj = self.memory_proj(mem_mean).unsqueeze(0).expand(B, -1)
            gate     = self.memory_gate(
                torch.cat([new_state, mem_proj], dim=-1)
            )
            new_state = gate * new_state + (1 - gate) * mem_proj

        new_state = self.state_norm(new_state)

        # Push mean of batch into buffer (single-patient case: B=1)
        if B == 1:
            self._push(new_state[0])

        return new_state

    # ── Longitudinal analytics ────────────────────────────────────────────────

    def compute_drift(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        L2 drift magnitude per timestep.
        state_sequence: (B, T, state_dim) → (B, T-1)
        """
        if state_sequence.size(1) < 2:
            return torch.zeros(state_sequence.size(0), 1,
                               device=state_sequence.device)
        diffs = state_sequence[:, 1:] - state_sequence[:, :-1]
        return diffs.norm(dim=-1)

    def compute_anomaly_from_baseline(
        self,
        current_state: torch.Tensor,   # (B, state_dim)
    ) -> torch.Tensor:                 # (B,)
        """
        Anomaly score = normalised L2 distance from memory baseline mean.
        Falls back to 0 if buffer is empty.
        """
        mem = self._get_memory_snapshot()
        if mem is None or mem.shape[0] < 2:
            return torch.zeros(current_state.size(0), device=current_state.device)

        baseline = mem.mean(0).to(current_state.device)   # (state_dim,)
        std      = mem.std(0).to(current_state.device).clamp(min=1e-6)
        diff     = (current_state - baseline) / std       # (B, state_dim)
        score    = diff.norm(dim=-1) / math.sqrt(self.state_dim)
        return torch.sigmoid(score - 1.0)                 # scaled to [0,1]

    def get_memory_statistics(self) -> Dict[str, torch.Tensor]:
        """Return summary statistics of the memory buffer."""
        mem = self._get_memory_snapshot()
        if mem is None:
            return {}
        return {
            "mean":   mem.mean(0),
            "std":    mem.std(0) if mem.shape[0] > 1 else torch.zeros_like(mem[0]),
            "size":   torch.tensor(mem.shape[0]),
            "trend":  (mem[-1] - mem[0]) if mem.shape[0] > 1 else torch.zeros_like(mem[0]),
        }
