"""
RAPiD-Twin: Complete integrated model.

Forward-pass shape contracts (post-refactor):
  - RetinalStructureEncoder: (B, 3, H, W) → (B, embed_dim)
    Uses TIMM backbone + FPN + SwinGlobalHead.
    Instantiated with: backbone, pretrained, embed_dim, fpn_dim, window_size, dropout

  - RetinalGNN (GATv2): (N, node_dim), (2, E), (E, edge_dim) → (1, out_dim)
    Batched via forward_batch.

  - MultimodalFusionTransformer: list[Optional[(B, dim_i)]] → (B, output_dim)
    No hardcoded dim clipping — every modality uses its own learned projection.

  - DigitalTwinStateTracker: persistent ring buffer, memory-augmented GRU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..models.gnn import RetinalGNN
from ..models.retinal_encoder import RetinalStructureEncoder
from ..models.physio_encoder import PhysioSignalEncoder
from ..fusion.multimodal_fusion import MultimodalFusionTransformer, HealthStateProjector
from ..twin.temporal_twin import TemporalFusionTransformer, DigitalTwinStateTracker
from ..models.disease_heads import MultiTaskDiseaseHead
from ..configs.config import RAPiDTwinConfig


class RAPiDTwin(nn.Module):
    """
    Retina-Anchored Physiological Digital Twin (RAPiD-Twin).

    Full forward pass:
      1. Retinal image → TIMM ConvNeXt-Tiny + FPN + SwinHead → retinal_emb (B, embed_dim)
      2. Vascular graph → GATv2 GNN                           → graph_emb   (B, gnn_out_dim)
      3. Physio signals → Transformer encoder                 → physio_emb  (B, physio_dim)
      4. [graph_emb | retinal_emb | physio_emb]
              → Multimodal Fusion Transformer (no dim clipping) → fused (B, output_dim)
      5. history sequence → TFT                               → health_state (B, state_dim)
      6. health_state + memory → StateTracker                 → updated_state
      7. updated_state → Disease heads                        → {task: logits}
    """

    def __init__(self, cfg: Optional[RAPiDTwinConfig] = None):
        super().__init__()
        if cfg is None:
            from ..configs.config import DEFAULT_CONFIG
            cfg = DEFAULT_CONFIG
        self.cfg = cfg

        # ── 1. Retinal Structure Encoder (TIMM backbone) ──────────────────────
        self.retinal_encoder = RetinalStructureEncoder(
            in_channels = cfg.segmentation.in_channels,
            embed_dim   = cfg.retinal_encoder.embed_dim,
            backbone    = cfg.retinal_encoder.backbone,
            pretrained  = cfg.retinal_encoder.pretrained,
            fpn_dim     = cfg.retinal_encoder.fpn_dim,
            window_size = cfg.retinal_encoder.window_size,
            dropout     = cfg.retinal_encoder.dropout,
        )

        # ── 2. Graph Neural Network (GATv2) ───────────────────────────────────
        self.gnn = RetinalGNN(
            node_in_dim = cfg.gnn.node_in_dim,
            edge_in_dim = cfg.gnn.edge_in_dim,
            hidden_dim  = cfg.gnn.hidden_dim,
            out_dim     = cfg.gnn.out_dim,
            num_heads   = cfg.gnn.num_heads,
            num_layers  = cfg.gnn.num_layers,
            dropout     = cfg.gnn.dropout,
        )

        # ── 3. Physiological Signal Encoder ───────────────────────────────────
        self.physio_encoder = PhysioSignalEncoder(
            input_dim  = cfg.physio_encoder.input_dim,
            hidden_dim = cfg.physio_encoder.hidden_dim,
            embed_dim  = cfg.physio_encoder.embed_dim,
            seq_len    = cfg.physio_encoder.seq_len,
            num_heads  = cfg.physio_encoder.num_heads,
            num_layers = cfg.physio_encoder.num_layers,
            dropout    = cfg.physio_encoder.dropout,
            use_tcn    = cfg.physio_encoder.use_tcn,
        )

        # ── 4. Multimodal Fusion Transformer ──────────────────────────────────
        # Modality dims come from each encoder's output — no assumptions
        modality_dims = [
            cfg.gnn.out_dim,
            cfg.retinal_encoder.embed_dim,
            cfg.physio_encoder.embed_dim,
        ]
        self.fusion = MultimodalFusionTransformer(
            modality_dims = modality_dims,
            fusion_dim    = cfg.fusion.fusion_dim,
            num_heads     = cfg.fusion.num_heads,
            num_layers    = cfg.fusion.num_layers,
            dropout       = cfg.fusion.dropout,
            output_dim    = cfg.fusion.output_dim,
            use_qformer   = cfg.fusion.use_qformer,
            num_queries   = cfg.fusion.num_queries,
        )

        # ── 5. Health State Projector ─────────────────────────────────────────
        self.health_projector = HealthStateProjector(
            input_dim  = cfg.fusion.output_dim,
            dropout    = cfg.fusion.dropout,
        )

        # ── 6. Temporal Fusion Transformer ────────────────────────────────────
        self.tft = TemporalFusionTransformer(
            state_dim          = cfg.fusion.output_dim,
            hidden_dim         = cfg.twin.hidden_dim,
            num_quantiles      = cfg.twin.num_quantiles,
            num_heads          = cfg.twin.num_heads,
            num_encoder_steps  = cfg.twin.num_encoder_steps,
            dropout            = cfg.twin.dropout,
        )

        # ── 7. Digital Twin State Tracker ─────────────────────────────────────
        self.state_tracker = DigitalTwinStateTracker(
            state_dim   = cfg.fusion.output_dim,
            memory_size = cfg.twin.memory_size,
        )

        # ── 8. Disease Prediction Heads ───────────────────────────────────────
        self.disease_heads = MultiTaskDiseaseHead(
            input_dim  = cfg.fusion.output_dim,
            hidden_dim = cfg.disease_head.hidden_dim,
            dropout    = cfg.disease_head.dropout,
        )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def encode_retinal_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, embed_dim)"""
        return self.retinal_encoder(image)

    def encode_vascular_graph(
        self,
        node_features: torch.Tensor,   # (N, node_in_dim)
        edge_index:    torch.Tensor,   # (2, E)
        edge_features: torch.Tensor,   # (E, edge_in_dim)
    ) -> torch.Tensor:                 # (1, gnn_out_dim)
        return self.gnn(node_features, edge_index, edge_features)

    def encode_physio(self, signals: torch.Tensor,
                      missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D) → (B, embed_dim)"""
        return self.physio_encoder(signals, missing_mask)

    # ── Graph batch helper ────────────────────────────────────────────────────

    def _encode_graph_batch(
        self,
        node_list:      List[torch.Tensor],
        edge_idx_list:  List[torch.Tensor],
        edge_feat_list: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Process a list of variable-size graphs and return (B, gnn_out_dim).

        Uses block-diagonal adjacency batching via RetinalGNN.forward_batch_blocked():
        all graphs are concatenated into one block-diagonal graph and processed
        in a single GNN forward pass, with scatter operations for per-graph pooling.
        This is ~10–15× faster than the previous serial loop on GPU.
        """
        return self.gnn.forward_batch_blocked(
            node_list      = [nf.to(device) for nf in node_list],
            ei_list        = [ei.to(device) for ei in edge_idx_list],
            ef_list        = [ef.to(device) for ef in edge_feat_list],
        )

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        retinal_image:       Optional[torch.Tensor] = None,        # (B, 3, H, W)
        node_features:       Optional[object]       = None,        # (N,F) or List[(N_i,F)]
        edge_index:          Optional[object]       = None,        # (2,E) or List[(2,E_i)]
        edge_features:       Optional[object]       = None,        # (E,F) or List[(E_i,F)]
        physio_signals:      Optional[torch.Tensor] = None,        # (B, T, D)
        historical_states:   Optional[torch.Tensor] = None,        # (B, T_hist, state_dim)
        static_context:      Optional[torch.Tensor] = None,        # (B, state_dim)
        physio_missing_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        modalities: List[Optional[torch.Tensor]] = []
        available:  List[bool] = []

        # ── Infer batch size and device ────────────────────────────────────────
        B = 1
        device = None
        if retinal_image is not None:
            B, device = retinal_image.shape[0], retinal_image.device
        elif physio_signals is not None:
            B, device = physio_signals.shape[0], physio_signals.device
        elif isinstance(node_features, list) and node_features:
            B, device = len(node_features), node_features[0].device
        elif isinstance(node_features, torch.Tensor):
            device = node_features.device

        if device is None:
            device = next(self.parameters()).device

        # ── 1. Graph embedding ─────────────────────────────────────────────────
        if node_features is not None and edge_index is not None:
            ef_default_dim = self.cfg.gnn.edge_in_dim
            try:
                if isinstance(node_features, list):
                    # Batch of variable-size graphs
                    if edge_features is None:
                        edge_features = [
                            torch.zeros(ei.shape[1], ef_default_dim)
                            for ei in edge_index
                        ]
                    graph_emb = self._encode_graph_batch(
                        node_features, edge_index, edge_features, device
                    )   # (B, out_dim)
                else:
                    # Single graph — expand to batch
                    ef = edge_features if edge_features is not None else \
                         torch.zeros(0, ef_default_dim, device=device)
                    graph_emb = self.encode_vascular_graph(
                        node_features.to(device),
                        edge_index.to(device),
                        ef.to(device),
                    ).expand(B, -1)   # (B, out_dim)
                modalities.append(graph_emb)
                available.append(True)
            except Exception:
                modalities.append(None)
                available.append(False)
        else:
            modalities.append(None)
            available.append(False)

        # ── 2. Retinal image embedding ─────────────────────────────────────────
        if retinal_image is not None:
            retinal_emb = self.encode_retinal_image(retinal_image)   # (B, embed_dim)
            modalities.append(retinal_emb)
            available.append(True)
        else:
            modalities.append(None)
            available.append(False)

        # ── 3. Physiological signal embedding ─────────────────────────────────
        if physio_signals is not None:
            physio_emb = self.encode_physio(physio_signals, physio_missing_mask)  # (B, dim)
            modalities.append(physio_emb)
            available.append(True)
        else:
            modalities.append(None)
            available.append(False)

        avail_tensor = torch.tensor(available, dtype=torch.bool, device=device)
        avail_tensor = avail_tensor.unsqueeze(0).expand(B, -1)   # (B, M)

        # ── 4. Multimodal fusion ───────────────────────────────────────────────
        fusion_out = self.fusion(modalities, avail_tensor)
        fused_emb  = fusion_out["embedding"]    # (B, output_dim)

        # ── 5. Health state scores ─────────────────────────────────────────────
        health_scores = self.health_projector(fused_emb)

        # ── 6. Temporal twin ───────────────────────────────────────────────────
        twin_out: Dict = {}
        prev_state = None
        if historical_states is not None:
            twin_out   = self.tft(historical_states, static_context)
            prev_state = twin_out.get("health_state")
        else:
            twin_out["anomaly_score"] = torch.zeros(B, device=device)
            twin_out["baseline"]      = fused_emb

        updated_state = self.state_tracker.update(fused_emb, prev_state)

        # Memory-based anomaly score
        anomaly_from_mem = self.state_tracker.compute_anomaly_from_baseline(updated_state)
        final_anomaly = twin_out.get("anomaly_score", anomaly_from_mem)
        if isinstance(final_anomaly, torch.Tensor) and isinstance(anomaly_from_mem, torch.Tensor):
            final_anomaly = 0.5 * final_anomaly + 0.5 * anomaly_from_mem

        # ── 7. Disease prediction ──────────────────────────────────────────────
        disease_preds = self.disease_heads(updated_state)

        return {
            "graph_embedding":    modalities[0],
            "retinal_embedding":  modalities[1],
            "physio_embedding":   modalities[2],
            "fused_embedding":    fused_emb,
            "modality_weights":   fusion_out["modality_weights"],
            "token_embeddings":   fusion_out["token_embeddings"],
            "health_scores":      health_scores,
            "updated_state":      updated_state,
            "previous_state":     prev_state,
            "anomaly_score":      final_anomaly,
            "health_baseline":    twin_out.get("baseline"),
            "twin_predictions":   twin_out.get("quantile_predictions"),
            "disease_predictions": disease_preds,
        }

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def update_twin_state(self, new_embedding: torch.Tensor,
                          previous_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.state_tracker.update(new_embedding, previous_state)

    def reset_memory(self):
        """Reset the persistent memory buffer (call between patients)."""
        self.state_tracker.reset_memory()

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the pretrained TIMM backbone."""
        self.retinal_encoder.freeze_backbone(freeze)

    def count_parameters(self) -> Dict[str, int]:
        components = {
            "retinal_encoder": self.retinal_encoder,
            "gnn":             self.gnn,
            "physio_encoder":  self.physio_encoder,
            "fusion":          self.fusion,
            "health_projector":self.health_projector,
            "tft":             self.tft,
            "state_tracker":   self.state_tracker,
            "disease_heads":   self.disease_heads,
        }
        counts = {k: sum(p.numel() for p in m.parameters() if p.requires_grad)
                  for k, m in components.items()}
        counts["total"] = sum(counts.values())
        return counts
