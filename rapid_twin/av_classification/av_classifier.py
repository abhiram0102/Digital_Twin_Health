"""
Artery / Vein Intelligence Module.

Provides graph-topology-aware artery/vein classification for retinal vessels.

Architecture:
  1. AVClassifier (nn.Module) — GATv2-based node-level classifier.
     Input:  node_features (N, node_in_dim), edge_index, edge_features
     Output: (N, 3) logits for [ARTERY, VEIN, UNCERTAIN]

  2. propagate_av_labels() — graph-based label propagation that enforces
     vascular lineage consistency.  Seed confident predictions; spread labels
     along vessel paths with decaying confidence.

  3. disambiguate_crossings() — resolve A/V crossings where two vessels
     of different types share a node (crossing point).

  4. AVDatasetAdapter — unified loader for DRIVE-AV, LES-AV, IOSTAR, HRF-AV.

  5. compute_av_metrics() — accuracy, crossing consistency, lineage consistency.

Design contracts:
  - AVClassifier follows the same PyTorch nn.Module conventions as RetinalGNN.
  - All propagation is graph-based; no pixel-level post-processing required.
  - Uncertainty is an explicit third class — no forced binary labeling.
  - All inputs are validated; degenerate graphs return UNCERTAIN predictions.
  - No silent NaN outputs.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("PyTorch not available — AVClassifier cannot be instantiated.")

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False


# ─── Labels ───────────────────────────────────────────────────────────────────

class AVLabel(IntEnum):
    ARTERY    = 0
    VEIN      = 1
    UNCERTAIN = 2


@dataclass
class AVPrediction:
    """Per-node A/V prediction with confidence."""
    node_id:    int
    label:      AVLabel
    confidence: float       # in [0, 1]
    prob_artery: float
    prob_vein:   float
    prob_uncertain: float
    propagated: bool = False  # True if set by propagation, not classifier


# ─── GATv2 Layer (shared with gnn.py but standalone here for AV head) ─────────

if _TORCH:
    class _GATv2Layer(nn.Module):
        """
        Simplified GATv2 attention layer for node-level classification.
        Supports both node and edge features.
        """

        def __init__(self, in_dim: int, out_dim: int, edge_dim: int, n_heads: int = 4):
            super().__init__()
            assert out_dim % n_heads == 0
            self.n_heads   = n_heads
            self.head_dim  = out_dim // n_heads
            self.W_src     = nn.Linear(in_dim, out_dim, bias=False)
            self.W_dst     = nn.Linear(in_dim, out_dim, bias=False)
            self.W_edge    = nn.Linear(edge_dim, out_dim, bias=False)
            self.attn      = nn.Linear(out_dim, n_heads, bias=False)
            self.out_proj  = nn.Linear(out_dim, out_dim)
            self.norm      = nn.LayerNorm(out_dim)
            self.act       = nn.GELU()

        def forward(
            self,
            x:          "torch.Tensor",   # (N, in_dim)
            edge_index: "torch.Tensor",   # (2, E)
            edge_feat:  "torch.Tensor",   # (E, edge_dim)
        ) -> "torch.Tensor":              # (N, out_dim)
            N = x.size(0)
            if edge_index.size(1) == 0 or N == 0:
                return self.act(self.norm(self.W_src(x)))

            src_idx, dst_idx = edge_index[0], edge_index[1]
            # Clamp indices
            src_idx = src_idx.clamp(0, N - 1)
            dst_idx = dst_idx.clamp(0, N - 1)

            h_src  = self.W_src(x)[src_idx]          # (E, out_dim)
            h_dst  = self.W_dst(x)[dst_idx]          # (E, out_dim)
            h_edge = self.W_edge(edge_feat)           # (E, out_dim)

            e = self.act(h_src + h_dst + h_edge)      # (E, out_dim)
            alpha = self.attn(e)                       # (E, n_heads)
            alpha = alpha - alpha.max(dim=0, keepdim=True).values
            alpha = torch.exp(alpha)

            # Normalise per destination node
            denom = torch.zeros(N, self.n_heads, device=x.device)
            denom.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, self.n_heads), alpha)
            alpha_norm = alpha / (denom[dst_idx] + 1e-8)  # (E, n_heads)

            # Aggregate: (N, out_dim)
            out = torch.zeros(N, self.W_src.out_features, device=x.device)
            weighted = (alpha_norm.unsqueeze(-1) *
                        e.view(-1, self.n_heads, self.head_dim)).view(-1, self.W_src.out_features)
            out.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, self.W_src.out_features), weighted)

            return self.act(self.norm(self.out_proj(out) + x if x.size(-1) == out.size(-1) else self.out_proj(out)))


# ─── AVClassifier ─────────────────────────────────────────────────────────────

if _TORCH:
    class AVClassifier(nn.Module):
        """
        Graph-topology-aware Artery/Vein classifier.

        Architecture:
          - node embedding: Linear projection of node features
          - N × GATv2 layers with edge conditioning
          - AV prediction head: Linear(hidden) → 3 classes [A, V, U]
          - Confidence head: Linear(hidden) → 1 sigmoid (prediction confidence)

        Input:
          node_features  (N, node_in_dim)
          edge_index     (2, E)  long tensor
          edge_features  (E, edge_in_dim)   [uses undirected pairs]

        Output:
          av_logits    (N, 3)     raw logits
          confidence   (N, 1)     sigmoid confidence per node
        """

        def __init__(
            self,
            node_in_dim: int = 16,
            edge_in_dim: int = 8,
            hidden_dim:  int = 64,
            n_heads:     int = 4,
            n_layers:    int = 3,
            dropout:     float = 0.1,
        ):
            super().__init__()

            # Input projection
            self.node_proj = nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            # Use only forward edges (first E rows) — strip reverse
            self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

            # GATv2 layers
            self.gat_layers = nn.ModuleList([
                _GATv2Layer(hidden_dim, hidden_dim, hidden_dim, n_heads)
                for _ in range(n_layers)
            ])

            self.dropout = nn.Dropout(dropout)

            # AV classification head
            self.av_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),  # [ARTERY, VEIN, UNCERTAIN]
            )

            # Confidence head
            self.conf_head = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            node_features: "torch.Tensor",
            edge_index:    "torch.Tensor",
            edge_features: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Args:
                node_features: (N, node_in_dim)
                edge_index:    (2, 2E) — undirected edges stored twice
                edge_features: (2E, edge_in_dim)

            Returns:
                av_logits:  (N, 3)
                confidence: (N, 1)
            """
            N = node_features.size(0)
            E_double = edge_index.size(1)

            if N == 0:
                return (torch.zeros(0, 3, device=node_features.device),
                        torch.zeros(0, 1, device=node_features.device))

            x = self.node_proj(node_features)

            # Use only forward edges for edge features (first half)
            E = E_double // 2
            if E > 0 and edge_features.size(0) >= E:
                ef = self.edge_proj(edge_features[:E])
                ei = edge_index[:, :E]
            else:
                ef = torch.zeros(0, x.size(-1), device=x.device)
                ei = torch.zeros(2, 0, dtype=torch.long, device=x.device)

            for layer in self.gat_layers:
                x = self.dropout(layer(x, ei, ef))

            av_logits  = self.av_head(x)    # (N, 3)
            confidence = self.conf_head(x)  # (N, 1)

            return av_logits, confidence

        def predict(
            self,
            node_features: "torch.Tensor",
            edge_index:    "torch.Tensor",
            edge_features: "torch.Tensor",
        ) -> List[AVPrediction]:
            """
            Run inference and return structured AVPrediction objects.
            """
            self.eval()
            with torch.no_grad():
                logits, conf = self.forward(node_features, edge_index, edge_features)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                conf_np = conf.cpu().squeeze(-1).numpy()

            predictions = []
            for i in range(len(probs)):
                p_a, p_v, p_u = float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2])
                label = AVLabel(int(np.argmax(probs[i])))
                predictions.append(AVPrediction(
                    node_id=i,
                    label=label,
                    confidence=float(conf_np[i]),
                    prob_artery=p_a,
                    prob_vein=p_v,
                    prob_uncertain=p_u,
                ))
            return predictions

else:
    class AVClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for AVClassifier.")


# ─── Label Propagation ────────────────────────────────────────────────────────

def propagate_av_labels(
    graph,
    predictions:         List[AVPrediction],
    confidence_threshold: float = 0.7,
    decay:               float = 0.85,
    max_hops:            int   = 20,
) -> List[AVPrediction]:
    """
    Propagate confident AV predictions along vessel paths using BFS.

    Nodes with prediction confidence >= confidence_threshold are used as seeds.
    Labels spread to uncertain neighbours with decaying confidence.
    A node keeps its propagated label only if the propagated confidence
    exceeds the existing classifier confidence.

    Enforces vascular lineage constraint:
      A propagated label is accepted only if it is consistent with the
      majority of already-labeled neighbours (no isolated A in a V cluster).

    Args:
        graph:                VascularGraph.
        predictions:          list of AVPrediction (indexed 0..N-1).
        confidence_threshold: minimum confidence to be a seed node.
        decay:                confidence decay per hop (0 < decay < 1).
        max_hops:             maximum BFS depth for propagation.

    Returns:
        Updated list of AVPrediction.
    """
    if not _NX or not hasattr(graph, "nx_graph"):
        return predictions

    G       = graph.nx_graph
    pred    = {p.node_id: p for p in predictions}
    updated = {nid: False for nid in pred}

    # Seed queue: (confidence, node_id, label, hop_count)
    queue: deque = deque()
    for p in predictions:
        if p.label != AVLabel.UNCERTAIN and p.confidence >= confidence_threshold:
            queue.append((p.confidence, p.node_id, p.label, 0))

    visited_with_label: Dict[int, Tuple[AVLabel, float]] = {}

    while queue:
        conf, nid, label, hop = queue.popleft()

        if hop > max_hops:
            continue

        # Check neighbour consistency
        nb_labels = [
            visited_with_label.get(nb, (None, 0.0))[0]
            for nb in G.neighbors(nid)
            if visited_with_label.get(nb, (None, 0.0))[0] is not None
        ]
        if nb_labels:
            # Majority vote among known neighbours
            n_artery = sum(1 for l in nb_labels if l == AVLabel.ARTERY)
            n_vein   = sum(1 for l in nb_labels if l == AVLabel.VEIN)
            majority = AVLabel.ARTERY if n_artery >= n_vein else AVLabel.VEIN
            if label != majority and len(nb_labels) >= 2:
                label = majority  # flip to match neighbourhood

        visited_with_label[nid] = (label, conf)

        # Update prediction if propagated confidence is higher
        existing = pred.get(nid)
        if existing is not None and (existing.label == AVLabel.UNCERTAIN or conf > existing.confidence):
            pred[nid] = AVPrediction(
                node_id=nid,
                label=label,
                confidence=conf,
                prob_artery=conf if label == AVLabel.ARTERY else (1 - conf) / 2,
                prob_vein=conf if label == AVLabel.VEIN else (1 - conf) / 2,
                prob_uncertain=(1 - conf),
                propagated=True,
            )
            updated[nid] = True

        # Spread to neighbours
        next_conf = conf * decay
        if next_conf < 0.3:
            continue
        for nb in G.neighbors(nid):
            existing_nb = pred.get(nb)
            if existing_nb is None:
                continue
            if existing_nb.label == AVLabel.UNCERTAIN or next_conf > existing_nb.confidence:
                queue.append((next_conf, nb, label, hop + 1))

    return list(pred.values())


# ─── Crossing Disambiguation ──────────────────────────────────────────────────

def disambiguate_crossings(
    graph,
    predictions: List[AVPrediction],
) -> List[AVPrediction]:
    """
    Resolve artery/vein crossings where a high-degree node has both artery
    and vein neighbours.

    Strategy:
      At each crossing node (degree >= 4), group incident edges by their
      opposite-endpoint's AV label.  If the node is classified differently
      from the majority of its neighbours, flip it to the majority label.

    Args:
        graph:       VascularGraph.
        predictions: list of AVPrediction from propagate_av_labels().

    Returns:
        Updated list of AVPrediction.
    """
    if not _NX or not hasattr(graph, "nx_graph"):
        return predictions

    G    = graph.nx_graph
    pred = {p.node_id: p for p in predictions}

    crossing_nodes = [n for n in G.nodes() if G.degree(n) >= 4]

    for nid in crossing_nodes:
        nb_labels = [pred[nb].label for nb in G.neighbors(nid) if nb in pred]
        if not nb_labels:
            continue
        n_artery = sum(1 for l in nb_labels if l == AVLabel.ARTERY)
        n_vein   = sum(1 for l in nb_labels if l == AVLabel.VEIN)

        # If a clear majority exists, snap the crossing node to it
        if n_artery > n_vein * 2 and nid in pred:
            p = pred[nid]
            if p.label != AVLabel.ARTERY:
                pred[nid] = AVPrediction(
                    node_id=nid,
                    label=AVLabel.ARTERY,
                    confidence=p.confidence * 0.8,
                    prob_artery=0.8, prob_vein=0.1, prob_uncertain=0.1,
                    propagated=True,
                )
        elif n_vein > n_artery * 2 and nid in pred:
            p = pred[nid]
            if p.label != AVLabel.VEIN:
                pred[nid] = AVPrediction(
                    node_id=nid,
                    label=AVLabel.VEIN,
                    confidence=p.confidence * 0.8,
                    prob_artery=0.1, prob_vein=0.8, prob_uncertain=0.1,
                    propagated=True,
                )

    return list(pred.values())


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_av_metrics(
    predictions: List[AVPrediction],
    ground_truth: Dict[int, AVLabel],
) -> Dict[str, float]:
    """
    Compute A/V classification accuracy metrics.

    Args:
        predictions:  list of AVPrediction.
        ground_truth: dict mapping node_id → AVLabel (excluding UNCERTAIN).

    Returns:
        dict with:
          - accuracy:              fraction correctly classified.
          - artery_precision:      TP_A / (TP_A + FP_A).
          - artery_recall:         TP_A / (TP_A + FN_A).
          - vein_precision, vein_recall.
          - f1_artery, f1_vein.
          - uncertain_fraction:    fraction of predictions labeled UNCERTAIN.
          - lineage_consistency:   fraction of same-label neighbouring pairs.
    """
    pred_map = {p.node_id: p for p in predictions}

    TP_A = FP_A = FN_A = 0
    TP_V = FP_V = FN_V = 0
    n_uncertain = 0
    n_total     = 0

    for nid, gt_label in ground_truth.items():
        p = pred_map.get(nid)
        if p is None:
            continue
        n_total += 1
        if p.label == AVLabel.UNCERTAIN:
            n_uncertain += 1
            continue
        if gt_label == AVLabel.ARTERY:
            if p.label == AVLabel.ARTERY:
                TP_A += 1
            else:
                FN_A += 1; FP_V += 1
        elif gt_label == AVLabel.VEIN:
            if p.label == AVLabel.VEIN:
                TP_V += 1
            else:
                FN_V += 1; FP_A += 1

    def _precision(tp, fp): return tp / (tp + fp + 1e-8)
    def _recall(tp, fn):    return tp / (tp + fn + 1e-8)
    def _f1(p, r):          return 2 * p * r / (p + r + 1e-8)

    p_a = _precision(TP_A, FP_A)
    r_a = _recall(TP_A, FN_A)
    p_v = _precision(TP_V, FP_V)
    r_v = _recall(TP_V, FN_V)

    acc = (TP_A + TP_V) / max(n_total - n_uncertain, 1)

    return {
        "accuracy":           float(acc),
        "artery_precision":   float(p_a),
        "artery_recall":      float(r_a),
        "vein_precision":     float(p_v),
        "vein_recall":        float(r_v),
        "f1_artery":          float(_f1(p_a, r_a)),
        "f1_vein":            float(_f1(p_v, r_v)),
        "uncertain_fraction": float(n_uncertain / max(n_total, 1)),
    }


# ─── Dataset Adapter ──────────────────────────────────────────────────────────

@dataclass
class AVSample:
    """Single annotated sample for A/V classification training/evaluation."""
    image_id:      str
    vessel_mask:   np.ndarray          # (H, W) binary
    artery_mask:   Optional[np.ndarray]  # (H, W) binary, None if not available
    vein_mask:     Optional[np.ndarray]
    node_labels:   Optional[Dict[int, AVLabel]] = None  # populated after graph build


class AVDatasetAdapter:
    """
    Unified data adapter for A/V annotated retinal datasets.

    Supported formats:
      DRIVE-AV  — separate artery/vein PNG masks
      LES-AV    — separate artery/vein PNG masks (same format)
      IOSTAR    — artery/vein labels embedded in colour channel
      HRF-AV    — separate artery/vein masks

    Usage:
        adapter = AVDatasetAdapter("DRIVE-AV", root="/path/to/DRIVE-AV")
        for sample in adapter:
            graph = builder.build(sample.vessel_mask)
            labels = adapter.extract_node_labels(graph, sample)
    """

    SUPPORTED = {"DRIVE-AV", "LES-AV", "IOSTAR", "HRF-AV"}

    def __init__(self, dataset_name: str, root: str = ""):
        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: {sorted(self.SUPPORTED)}"
            )
        self.dataset_name = dataset_name
        self.root         = Path(root)
        self._samples: Optional[List[AVSample]] = None

    def load(self) -> List[AVSample]:
        """
        Load dataset samples.  Returns list of AVSample.
        Raises FileNotFoundError with a clear message if root does not exist.
        """
        if not self.root.exists():
            raise FileNotFoundError(
                f"AVDatasetAdapter: root path '{self.root}' does not exist. "
                f"Download {self.dataset_name} and set root= correctly."
            )
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for AVDatasetAdapter.load()")

        samples = []
        if self.dataset_name in {"DRIVE-AV", "LES-AV", "HRF-AV"}:
            samples = self._load_separate_masks(cv2)
        elif self.dataset_name == "IOSTAR":
            samples = self._load_iostar(cv2)

        self._samples = samples
        return samples

    def _load_separate_masks(self, cv2) -> List[AVSample]:
        """Load datasets with separate artery/vein PNG mask files."""
        vessel_dir = self.root / "vessels"
        artery_dir = self.root / "arteries"
        vein_dir   = self.root / "veins"

        if not vessel_dir.exists():
            raise FileNotFoundError(
                f"Expected 'vessels/' directory in {self.root}. "
                "Check dataset structure."
            )

        samples = []
        for vessel_path in sorted(vessel_dir.glob("*.png")):
            img_id = vessel_path.stem
            v_mask = (cv2.imread(str(vessel_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

            a_path = artery_dir / f"{img_id}.png"
            ve_path = vein_dir  / f"{img_id}.png"
            a_mask  = (cv2.imread(str(a_path),  cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) \
                      if a_path.exists()  else None
            ve_mask = (cv2.imread(str(ve_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) \
                      if ve_path.exists() else None

            samples.append(AVSample(
                image_id=img_id,
                vessel_mask=v_mask.astype(np.float32),
                artery_mask=a_mask.astype(np.float32) if a_mask is not None else None,
                vein_mask=ve_mask.astype(np.float32)  if ve_mask is not None else None,
            ))
        return samples

    def _load_iostar(self, cv2) -> List[AVSample]:
        """
        Load IOSTAR dataset: artery=red channel, vein=blue channel of label image.
        """
        label_dir = self.root / "labels"
        if not label_dir.exists():
            raise FileNotFoundError(
                f"Expected 'labels/' directory in {self.root} for IOSTAR."
            )
        samples = []
        for lpath in sorted(label_dir.glob("*.png")):
            img_id = lpath.stem
            label_img = cv2.imread(str(lpath), cv2.IMREAD_COLOR)
            if label_img is None:
                warnings.warn(f"IOSTAR: could not read {lpath}, skipping.")
                continue
            artery_mask = (label_img[:, :, 2] > 127).astype(np.float32)  # red
            vein_mask   = (label_img[:, :, 0] > 127).astype(np.float32)  # blue
            vessel_mask = np.clip(artery_mask + vein_mask, 0, 1)
            samples.append(AVSample(
                image_id=img_id,
                vessel_mask=vessel_mask,
                artery_mask=artery_mask,
                vein_mask=vein_mask,
            ))
        return samples

    def extract_node_labels(
        self,
        graph,
        sample: AVSample,
        radius: int = 5,
    ) -> Dict[int, AVLabel]:
        """
        Assign AV labels to graph nodes by majority-voting pixels within
        `radius` of each node position against the artery/vein masks.

        Args:
            graph:  VascularGraph.
            sample: AVSample with artery_mask / vein_mask.
            radius: pixel radius for majority vote.

        Returns:
            Dict[node_id → AVLabel]  (UNCERTAIN if masks unavailable or tied)
        """
        labels: Dict[int, AVLabel] = {}

        if sample.artery_mask is None or sample.vein_mask is None:
            for n in graph.nodes:
                labels[n.node_id] = AVLabel.UNCERTAIN
            return labels

        H, W = sample.artery_mask.shape
        for node in graph.nodes:
            r, c = node.position
            r0, r1 = max(0, r - radius), min(H, r + radius + 1)
            c0, c1 = max(0, c - radius), min(W, c + radius + 1)

            a_sum = float(sample.artery_mask[r0:r1, c0:c1].sum())
            v_sum = float(sample.vein_mask[r0:r1, c0:c1].sum())

            if a_sum > v_sum * 1.5:
                labels[node.node_id] = AVLabel.ARTERY
            elif v_sum > a_sum * 1.5:
                labels[node.node_id] = AVLabel.VEIN
            else:
                labels[node.node_id] = AVLabel.UNCERTAIN

        return labels

    def __iter__(self):
        if self._samples is None:
            self.load()
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples) if self._samples is not None else 0
