"""
Multi-scale Vessel Representation — Stage 1, Functional Vascular Characterization.

Retinal vasculature operates across three distinct morphological scales:

  LARGE     — main arteries and veins (optic-disc tributaries and main branches)
  MEDIUM    — secondary branching vessels
  CAPILLARY — fine terminal vessels and capillary-like structures

Scale is determined by edge caliber (mean_diameter in graph pixels).  Thresholds
are configurable; defaults are tuned for 3504×2336 Topcon camera images
(1 px ≈ 3.9 μm) at 45° FOV:
  LARGE     > 6.0 px  (≈ 23 μm — main arteries/veins)
  MEDIUM    3.0–6.0 px (≈ 12–23 μm — secondary branches)
  CAPILLARY < 3.0 px  (≈ < 12 μm — capillary-level)

Provides:
  VesselScale              — enum
  ScaleClassification      — per-edge and per-node scale assignments
  MultiScaleGraph          — VascularGraph subgraphs per scale + cross-scale edges
  MultiScaleMetrics        — per-scale vessel statistics
  MultiScaleVesselAnalyzer — classify → decompose → metrics → embedding

Design contracts:
  • Every output is a new object — inputs are never mutated.
  • Nodes are classified by the maximum scale of their incident edges.
  • Isolated nodes (no edges) are classified as CAPILLARY by convention.
  • All metric values are in [0, 1] or clearly documented otherwise.
  • No external ML dependencies — works with stdlib + numpy.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─── Scale enum ───────────────────────────────────────────────────────────────

class VesselScale(Enum):
    """Morphological scale of a retinal vessel segment."""
    LARGE     = "large"      # main arteries/veins
    MEDIUM    = "medium"     # secondary branches
    CAPILLARY = "capillary"  # fine terminal vessels

    @classmethod
    def from_diameter(
        cls,
        diameter:        float,
        large_thresh:    float = 6.0,
        capillary_thresh: float = 3.0,
    ) -> "VesselScale":
        """Classify a single caliber value into a scale tier."""
        if diameter > large_thresh:
            return cls.LARGE
        if diameter > capillary_thresh:
            return cls.MEDIUM
        return cls.CAPILLARY


# ─── Data containers ──────────────────────────────────────────────────────────

@dataclass
class ScaleClassification:
    """
    Per-edge and per-node scale assignments for a VascularGraph.

    Attributes:
        edge_scale:  dict mapping edge (src, dst) → VesselScale.
        node_scale:  dict mapping node_id → VesselScale  (max of incident edges).
        scale_counts: dict VesselScale → int counts.
        large_thresh:    caliber threshold used for LARGE classification.
        capillary_thresh: caliber threshold used for CAPILLARY classification.
    """
    edge_scale:       Dict[Tuple[int, int], VesselScale]
    node_scale:       Dict[int, VesselScale]
    scale_counts:     Dict[VesselScale, int]
    large_thresh:     float
    capillary_thresh: float

    def fraction(self, scale: VesselScale) -> float:
        """Fraction of edges assigned to this scale."""
        total = sum(self.scale_counts.values())
        return self.scale_counts.get(scale, 0) / max(total, 1)


@dataclass
class MultiScaleGraph:
    """
    VascularGraph decomposed into per-scale sub-structures.

    Attributes:
        scale_nodes:       VesselScale → list of node_ids at that scale.
        scale_edges:       VesselScale → list of (src, dst) at that scale.
        cross_scale_edges: list of (src_id, dst_id, src_scale, dst_scale) for
                           edges that cross scale boundaries (unusual; flagged).
        classification:    the ScaleClassification that produced this decomposition.
        source_graph:      reference to the original VascularGraph.
    """
    scale_nodes:       Dict[VesselScale, List[int]]
    scale_edges:       Dict[VesselScale, List[Tuple[int, int]]]
    cross_scale_edges: List[Tuple[int, int, VesselScale, VesselScale]]
    classification:    ScaleClassification
    source_graph:      object   # VascularGraph (avoiding circular import)


@dataclass
class MultiScaleMetrics:
    """
    Per-scale vessel statistics and cross-scale summary.

    All per-scale dicts have keys: VesselScale.LARGE, .MEDIUM, .CAPILLARY.

    Attributes:
        node_counts:         nodes per scale.
        edge_counts:         edges per scale.
        mean_tortuosity:     mean tortuosity per scale.
        mean_diameter:       mean caliber per scale.
        mean_length:         mean edge length per scale.
        density_fraction:    fraction of graph edges at each scale.
        tortuosity_spectrum: full distribution (std) of tortuosity per scale.
        connectivity_ratio:  edges / nodes ratio per scale.
        scale_embedding:     (D,) float32 feature vector concatenating per-scale stats.
        n_cross_scale_edges: number of edges crossing scale boundaries.
    """
    node_counts:         Dict[VesselScale, int]
    edge_counts:         Dict[VesselScale, int]
    mean_tortuosity:     Dict[VesselScale, float]
    mean_diameter:       Dict[VesselScale, float]
    mean_length:         Dict[VesselScale, float]
    density_fraction:    Dict[VesselScale, float]
    tortuosity_spectrum: Dict[VesselScale, float]   # std
    connectivity_ratio:  Dict[VesselScale, float]
    scale_embedding:     np.ndarray
    n_cross_scale_edges: int

    def to_dict(self) -> Dict[str, float]:
        """Flat dict of all scalar metrics for benchmark consumption."""
        out: Dict[str, float] = {}
        for scale in VesselScale:
            s = scale.value
            out[f"{s}_nodes"]         = float(self.node_counts.get(scale, 0))
            out[f"{s}_edges"]         = float(self.edge_counts.get(scale, 0))
            out[f"{s}_mean_tortuosity"] = float(self.mean_tortuosity.get(scale, float("nan")))
            out[f"{s}_mean_diameter"]  = float(self.mean_diameter.get(scale, float("nan")))
            out[f"{s}_mean_length"]    = float(self.mean_length.get(scale, float("nan")))
            out[f"{s}_density_frac"]   = float(self.density_fraction.get(scale, float("nan")))
            out[f"{s}_tortuosity_std"] = float(self.tortuosity_spectrum.get(scale, float("nan")))
            out[f"{s}_connectivity"]   = float(self.connectivity_ratio.get(scale, float("nan")))
        out["n_cross_scale_edges"] = float(self.n_cross_scale_edges)
        return out


# ─── Multi-scale Analyzer ────────────────────────────────────────────────────

class MultiScaleVesselAnalyzer:
    """
    Decomposes a VascularGraph into large / medium / capillary scale tiers.

    Usage:
        analyzer = MultiScaleVesselAnalyzer(large_thresh=6.0, capillary_thresh=3.0)
        classification = analyzer.classify(graph)
        msg             = analyzer.decompose(graph, classification)
        metrics         = analyzer.compute_metrics(msg)
        embedding       = metrics.scale_embedding  # (24,) float32 feature vector
    """

    def __init__(
        self,
        large_thresh:     float = 6.0,
        capillary_thresh: float = 3.0,
    ):
        if capillary_thresh >= large_thresh:
            raise ValueError(
                f"capillary_thresh ({capillary_thresh}) must be < "
                f"large_thresh ({large_thresh})."
            )
        self.large_thresh     = large_thresh
        self.capillary_thresh = capillary_thresh

    # ── Step 1: classify ─────────────────────────────────────────────────────

    def classify(self, graph) -> ScaleClassification:
        """
        Assign a VesselScale to each edge and node.

        Node scale = maximum scale of all incident edges (so a node joining
        a LARGE artery and MEDIUM branch is classified as LARGE).

        Args:
            graph: VascularGraph from graph_builder.

        Returns:
            ScaleClassification.
        """
        edge_scale: Dict[Tuple[int, int], VesselScale] = {}
        node_scale: Dict[int, VesselScale]              = {}
        scale_counts: Dict[VesselScale, int]            = defaultdict(int)

        # Scale ordering for max-merge
        _ORDER = {VesselScale.CAPILLARY: 0, VesselScale.MEDIUM: 1, VesselScale.LARGE: 2}
        _LEVEL = [VesselScale.CAPILLARY, VesselScale.MEDIUM, VesselScale.LARGE]

        for edge in graph.edges:
            diam  = edge.mean_diameter if np.isfinite(edge.mean_diameter) else 0.0
            scale = VesselScale.from_diameter(
                diam, self.large_thresh, self.capillary_thresh
            )
            edge_scale[(edge.src, edge.dst)] = scale
            scale_counts[scale] += 1

            # Update node_scale: take max
            for nid in (edge.src, edge.dst):
                current = node_scale.get(nid, VesselScale.CAPILLARY)
                if _ORDER[scale] > _ORDER[current]:
                    node_scale[nid] = scale

        # Isolated nodes not yet in node_scale
        for node in graph.nodes:
            if node.node_id not in node_scale:
                node_scale[node.node_id] = VesselScale.CAPILLARY

        return ScaleClassification(
            edge_scale=edge_scale,
            node_scale=node_scale,
            scale_counts=dict(scale_counts),
            large_thresh=self.large_thresh,
            capillary_thresh=self.capillary_thresh,
        )

    # ── Step 2: decompose ─────────────────────────────────────────────────────

    def decompose(
        self,
        graph,
        classification: Optional[ScaleClassification] = None,
    ) -> MultiScaleGraph:
        """
        Split the graph into per-scale node and edge sets.

        Cross-scale edges are rare but can occur at transition junctions;
        they are collected separately for inspection.

        Args:
            graph:          VascularGraph.
            classification: pre-computed ScaleClassification (computed if None).

        Returns:
            MultiScaleGraph.
        """
        if classification is None:
            classification = self.classify(graph)

        scale_nodes: Dict[VesselScale, List[int]] = {s: [] for s in VesselScale}
        scale_edges: Dict[VesselScale, List[Tuple[int, int]]] = {s: [] for s in VesselScale}
        cross: List[Tuple[int, int, VesselScale, VesselScale]] = []

        for nid, ns in classification.node_scale.items():
            scale_nodes[ns].append(nid)

        for (src, dst), es in classification.edge_scale.items():
            ns_src = classification.node_scale.get(src, VesselScale.CAPILLARY)
            ns_dst = classification.node_scale.get(dst, VesselScale.CAPILLARY)
            scale_edges[es].append((src, dst))
            if ns_src != ns_dst:
                cross.append((src, dst, ns_src, ns_dst))

        return MultiScaleGraph(
            scale_nodes=scale_nodes,
            scale_edges=scale_edges,
            cross_scale_edges=cross,
            classification=classification,
            source_graph=graph,
        )

    # ── Step 3: compute metrics ───────────────────────────────────────────────

    def compute_metrics(self, msg: MultiScaleGraph) -> MultiScaleMetrics:
        """
        Compute per-scale statistics and build a scale-aware embedding.

        Args:
            msg: MultiScaleGraph from decompose().

        Returns:
            MultiScaleMetrics with per-scale stats and embedding vector.
        """
        graph = msg.source_graph
        # Build edge lookup
        edge_map = {(e.src, e.dst): e for e in graph.edges}
        edge_map.update({(e.dst, e.src): e for e in graph.edges})

        node_counts:         Dict[VesselScale, int]   = {}
        edge_counts:         Dict[VesselScale, int]   = {}
        mean_tortuosity:     Dict[VesselScale, float] = {}
        mean_diameter:       Dict[VesselScale, float] = {}
        mean_length:         Dict[VesselScale, float] = {}
        tortuosity_spectrum: Dict[VesselScale, float] = {}
        connectivity_ratio:  Dict[VesselScale, float] = {}
        density_fraction:    Dict[VesselScale, float] = {}

        total_edges = max(sum(len(v) for v in msg.scale_edges.values()), 1)

        for scale in VesselScale:
            n_nodes = len(msg.scale_nodes[scale])
            n_edges = len(msg.scale_edges[scale])
            node_counts[scale] = n_nodes
            edge_counts[scale] = n_edges
            density_fraction[scale] = n_edges / total_edges

            tortosities = []
            diameters   = []
            lengths     = []

            for src, dst in msg.scale_edges[scale]:
                e = edge_map.get((src, dst))
                if e is None:
                    continue
                if np.isfinite(e.tortuosity):
                    tortosities.append(e.tortuosity)
                if np.isfinite(e.mean_diameter) and e.mean_diameter > 0:
                    diameters.append(e.mean_diameter)
                if np.isfinite(e.length) and e.length > 0:
                    lengths.append(e.length)

            mean_tortuosity[scale]     = float(np.mean(tortosities))  if tortosities else float("nan")
            tortuosity_spectrum[scale] = float(np.std(tortosities))   if tortosities else float("nan")
            mean_diameter[scale]       = float(np.mean(diameters))    if diameters   else float("nan")
            mean_length[scale]         = float(np.mean(lengths))      if lengths     else float("nan")
            connectivity_ratio[scale]  = n_edges / max(n_nodes, 1)

        # Scale-aware embedding: 8 stats × 3 scales = 24 dimensions
        embedding_parts = []
        for scale in [VesselScale.LARGE, VesselScale.MEDIUM, VesselScale.CAPILLARY]:
            embedding_parts.extend([
                float(node_counts[scale]) / max(len(graph.nodes), 1),
                float(edge_counts[scale]) / max(len(graph.edges), 1),
                _safe(mean_tortuosity[scale]),
                _safe(tortuosity_spectrum[scale]),
                _safe(mean_diameter[scale]) / max(self.large_thresh, 1),
                _safe(mean_length[scale])   / 100.0,
                density_fraction[scale],
                connectivity_ratio[scale] / 4.0,   # normalise by typical max degree
            ])

        embedding = np.array(embedding_parts, dtype=np.float32)
        # Replace any remaining NaN with 0 (safe default for embedding)
        embedding = np.where(np.isfinite(embedding), embedding, 0.0).astype(np.float32)

        return MultiScaleMetrics(
            node_counts=node_counts,
            edge_counts=edge_counts,
            mean_tortuosity=mean_tortuosity,
            mean_diameter=mean_diameter,
            mean_length=mean_length,
            density_fraction=density_fraction,
            tortuosity_spectrum=tortuosity_spectrum,
            connectivity_ratio=connectivity_ratio,
            scale_embedding=embedding,
            n_cross_scale_edges=len(msg.cross_scale_edges),
        )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def analyze(self, graph) -> Tuple[MultiScaleGraph, MultiScaleMetrics]:
        """Convenience: classify → decompose → metrics in one call."""
        clf     = self.classify(graph)
        msg     = self.decompose(graph, clf)
        metrics = self.compute_metrics(msg)
        return msg, metrics


def _safe(v: float, default: float = 0.0) -> float:
    return float(v) if np.isfinite(v) else default


# ─── Capillary rarefaction detection ─────────────────────────────────────────

def detect_capillary_rarefaction(
    msg:        MultiScaleGraph,
    mask:       np.ndarray,
    cell_size:  int   = 32,
    low_thresh: float = 0.05,
) -> Dict[str, float]:
    """
    Estimate capillary rarefaction — reduction in fine vessel density.

    Divides the image into cell_size × cell_size tiles and counts capillary
    pixels.  Cells below low_thresh are flagged as rarefied.

    Args:
        msg:        MultiScaleGraph (capillary edges used).
        mask:       (H, W) binary float32 vessel mask.
        cell_size:  grid cell size in pixels.
        low_thresh: capillary density fraction below which a cell is rarefied.

    Returns:
        dict with:
          rarefaction_index:     fraction of non-empty cells that are rarefied.
          capillary_density:     overall capillary pixel fraction.
          rarefied_cell_count:   integer count of rarefied cells.
          total_cell_count:      total non-empty cells analysed.
    """
    if mask is None or mask.size == 0:
        return {k: float("nan") for k in
                ["rarefaction_index", "capillary_density",
                 "rarefied_cell_count", "total_cell_count"]}

    H, W = mask.shape
    binary = (mask > 0.5).astype(np.float32)

    # Build capillary-only mask from graph node positions
    cap_mask = np.zeros((H, W), dtype=np.float32)
    graph    = msg.source_graph
    node_map = {n.node_id: n for n in graph.nodes}
    cap_nodes = set(msg.scale_nodes.get(VesselScale.CAPILLARY, []))

    for nid in cap_nodes:
        node = node_map.get(nid)
        if node is None:
            continue
        r, c = int(node.position[0]), int(node.position[1])
        if 0 <= r < H and 0 <= c < W:
            cap_mask[r, c] = 1.0

    # Grid analysis
    n_rows = H // cell_size
    n_cols = W // cell_size
    rarefied = 0
    total    = 0

    for ri in range(n_rows):
        for ci in range(n_cols):
            r0, r1 = ri * cell_size, (ri + 1) * cell_size
            c0, c1 = ci * cell_size, (ci + 1) * cell_size
            cell_binary = binary[r0:r1, c0:c1]
            if cell_binary.sum() < 1:
                continue  # empty cell — no vessels at all, skip
            total += 1
            cell_cap = cap_mask[r0:r1, c0:c1]
            cap_frac = float(cell_cap.sum() / max(cell_cap.size, 1))
            if cap_frac < low_thresh:
                rarefied += 1

    return {
        "rarefaction_index":   float(rarefied / max(total, 1)),
        "capillary_density":   float(cap_mask.sum() / max(binary.sum(), 1)),
        "rarefied_cell_count": float(rarefied),
        "total_cell_count":    float(total),
    }
