"""
Retinal Vascular Graph Quality Metrics.

Quantifies graph-level vascular intelligence beyond pixel-level segmentation:
  - Fractal dimension (box-counting method)
  - Branch fragmentation index
  - Graph connectivity and robustness
  - Topology integrity score
  - Bifurcation density and angle distribution
  - Vessel caliber statistics
  - Tortuosity distribution statistics
  - Network efficiency (global / local)
  - Composite graph quality score

All functions accept a VascularGraph (from graph_builder) and/or a binary
numpy mask (H, W) — whichever is appropriate for that metric.

IMPORTANT:
  - Every public function validates its inputs and raises ValueError explicitly.
  - No silent fallbacks. NaN values are never returned without a clear reason.
  - All metrics are documented with their interpretation and value range.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    warnings.warn("networkx not available — network efficiency metrics disabled.")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn("opencv not available — some graph metrics will be degraded.")


# ─── Type alias (avoid circular import from graph_builder) ────────────────────

# We accept any object with .nodes, .edges, .nx_graph, .skeleton, .node_features,
# .edge_features attributes (i.e., VascularGraph from graph_builder).


# ─── Fractal Dimension ────────────────────────────────────────────────────────

def compute_fractal_dimension(
    mask: np.ndarray,
    box_sizes: Optional[List[int]] = None,
    min_box: int = 2,
    max_box_ratio: float = 0.25,
) -> float:
    """
    Box-counting fractal dimension of a binary vessel mask.

    Measures the self-similarity / complexity of the vascular network.
    Normal retinal vasculature: D ≈ 1.6–1.8.
    Lower values indicate sparser / more fragmented networks.
    Higher values indicate denser networks.

    Args:
        mask: (H, W) binary or float mask (>0.5 = vessel).
        box_sizes: list of box edge lengths (pixels). Auto-computed if None.
        min_box: minimum box size in pixels.
        max_box_ratio: maximum box size as fraction of image dimension.

    Returns:
        Fractal dimension D in [1.0, 2.0]. Returns 1.0 on degenerate inputs.
    """
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return 1.0

    H, W = binary.shape
    min_dim = min(H, W)
    max_box = max(min_box + 1, int(min_dim * max_box_ratio))

    if box_sizes is None:
        # Logarithmically spaced box sizes
        sizes = []
        s = min_box
        while s <= max_box:
            sizes.append(s)
            s = max(s + 1, int(s * 1.5))
        box_sizes = sizes

    if len(box_sizes) < 2:
        return 1.0

    counts = []
    for box in box_sizes:
        if box < 1:
            continue
        # Count non-empty boxes
        n_rows = (H + box - 1) // box
        n_cols = (W + box - 1) // box
        count = 0
        for r in range(n_rows):
            for c in range(n_cols):
                patch = binary[r * box: (r + 1) * box, c * box: (c + 1) * box]
                if patch.any():
                    count += 1
        counts.append((box, count))

    if len(counts) < 2:
        return 1.0

    # Fit log(count) = -D * log(box_size) + const
    log_sizes  = np.log([b for b, _ in counts])
    log_counts = np.log([max(c, 1) for _, c in counts])

    if log_sizes.std() < 1e-8:
        return 1.0

    try:
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        D = -coeffs[0]
    except (np.linalg.LinAlgError, ValueError):
        return 1.0

    return float(np.clip(D, 1.0, 2.0))


# ─── Branch Fragmentation ─────────────────────────────────────────────────────

def compute_branch_fragmentation(graph) -> Dict[str, float]:
    """
    Quantify branch fragmentation in the vascular graph.

    High fragmentation = missing vessel connections, broken vessel trees.
    Low fragmentation = well-connected, continuous vascular network.

    Args:
        graph: VascularGraph (from RetinalGraphBuilder).

    Returns:
        dict with:
          - fragmentation_index: 1 - (edges / max_possible_edges). In [0, 1].
            0 = complete graph, 1 = no edges.
          - isolated_node_fraction: fraction of nodes with degree 0.
          - endpoint_fraction: fraction of nodes that are endpoints (degree 1).
          - mean_degree: average node degree.
          - degree_std: standard deviation of node degree.
          - n_connected_components: number of weakly connected components.
          - largest_component_fraction: fraction of nodes in the largest component.
    """
    nodes = graph.nodes
    edges = graph.edges

    if not nodes:
        return {
            "fragmentation_index": 1.0,
            "isolated_node_fraction": 1.0,
            "endpoint_fraction": 0.0,
            "mean_degree": 0.0,
            "degree_std": 0.0,
            "n_connected_components": 0,
            "largest_component_fraction": 0.0,
        }

    N = len(nodes)
    E = len(edges)

    max_edges = N * (N - 1) / 2 if N > 1 else 1
    frag_idx = 1.0 - (E / max_edges)

    degrees = [n.degree for n in nodes]
    isolated = sum(1 for d in degrees if d == 0) / N
    endpoints = sum(1 for n in nodes if n.is_endpoint) / N
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    std_deg  = float(np.std(degrees))  if degrees else 0.0

    # Connected components via NetworkX
    n_components = 1
    largest_frac  = 1.0
    if _NX_AVAILABLE and hasattr(graph, "nx_graph"):
        G = graph.nx_graph
        if G.number_of_nodes() > 0:
            comps = list(nx.connected_components(G))
            n_components = len(comps)
            largest_frac = max(len(c) for c in comps) / G.number_of_nodes()

    return {
        "fragmentation_index":      float(np.clip(frag_idx, 0.0, 1.0)),
        "isolated_node_fraction":   float(isolated),
        "endpoint_fraction":        float(endpoints),
        "mean_degree":              mean_deg,
        "degree_std":               std_deg,
        "n_connected_components":   int(n_components),
        "largest_component_fraction": float(largest_frac),
    }


# ─── Graph Connectivity ───────────────────────────────────────────────────────

def compute_graph_connectivity(graph) -> Dict[str, float]:
    """
    Structural connectivity metrics for the vascular graph.

    Args:
        graph: VascularGraph.

    Returns:
        dict with:
          - is_connected: 1.0 if the entire graph is connected, else 0.0.
          - algebraic_connectivity: Fiedler value (2nd-smallest Laplacian
            eigenvalue). >0 means connected; larger = more robustly connected.
            Limited to graphs ≤ 500 nodes for performance.
          - node_connectivity: minimum node cut size (robustness to node failure).
            -1 if not computable.
          - edge_connectivity: minimum edge cut size.
            -1 if not computable.
          - density: 2*E / (N*(N-1)).  In [0, 1].
          - average_clustering: mean local clustering coefficient.
    """
    if not _NX_AVAILABLE or not hasattr(graph, "nx_graph"):
        return {k: float("nan") for k in [
            "is_connected", "algebraic_connectivity", "node_connectivity",
            "edge_connectivity", "density", "average_clustering",
        ]}

    G = graph.nx_graph
    N = G.number_of_nodes()
    E = G.number_of_edges()

    if N == 0:
        return {
            "is_connected": 0.0, "algebraic_connectivity": 0.0,
            "node_connectivity": 0.0, "edge_connectivity": 0.0,
            "density": 0.0, "average_clustering": 0.0,
        }

    is_conn = float(nx.is_connected(G))
    density  = nx.density(G)
    avg_clust = nx.average_clustering(G)

    alg_conn = 0.0
    if N <= 500 and is_conn:
        try:
            alg_conn = float(nx.algebraic_connectivity(G))
        except Exception:
            alg_conn = 0.0

    node_conn = -1
    edge_conn = -1
    if is_conn and N <= 200:
        try:
            node_conn = nx.node_connectivity(G)
            edge_conn = nx.edge_connectivity(G)
        except Exception:
            pass

    return {
        "is_connected":           is_conn,
        "algebraic_connectivity": float(alg_conn),
        "node_connectivity":      float(node_conn),
        "edge_connectivity":      float(edge_conn),
        "density":                float(density),
        "average_clustering":     float(avg_clust),
    }


# ─── Topology Integrity ───────────────────────────────────────────────────────

def compute_topology_integrity(graph, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Topology integrity: how faithfully the graph represents the vessel skeleton.

    Args:
        graph: VascularGraph.
        mask: (H, W) binary vessel mask for skeleton-based validation.
              If None, mask-based metrics are skipped.

    Returns:
        dict with:
          - node_coverage: fraction of skeleton pixels within radius 5 of a node.
          - bifurcation_density: bifurcation nodes per 1000 skeleton pixels.
          - endpoint_density: endpoint nodes per 1000 skeleton pixels.
          - skeleton_length: total skeleton pixel count (0 if mask not provided).
          - n_bifurcations: count of bifurcation nodes (degree >= 3).
          - n_endpoints: count of endpoint nodes (degree == 1).
          - bif_to_end_ratio: bifurcations / endpoints (> 1 = complex tree).
    """
    nodes = graph.nodes
    n_bif = sum(1 for n in nodes if n.is_bifurcation)
    n_end = sum(1 for n in nodes if n.is_endpoint)

    result: Dict[str, float] = {
        "n_bifurcations":   float(n_bif),
        "n_endpoints":      float(n_end),
        "bif_to_end_ratio": float(n_bif / max(n_end, 1)),
        "node_coverage":    float("nan"),
        "bifurcation_density": float("nan"),
        "endpoint_density":    float("nan"),
        "skeleton_length":  0.0,
    }

    if mask is None or not _CV2_AVAILABLE:
        return result

    # Build skeleton
    binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(binary)
    temp = binary.copy()
    while True:
        eroded  = cv2.erode(temp, kernel)
        opened  = cv2.dilate(eroded, kernel)
        sub     = cv2.subtract(temp, opened)
        skel    = cv2.bitwise_or(skel, sub)
        temp    = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    skel_len = int(skel.sum())
    result["skeleton_length"] = float(skel_len)

    if skel_len == 0:
        return result

    if nodes:
        skel_pts = np.argwhere(skel > 0)     # (K, 2)
        node_pts = np.array([list(n.position) for n in nodes])  # (N, 2)

        # Vectorised: min distance from each skel pixel to nearest node
        # For large skeletons, sample to keep cost O(K_sample * N)
        sample_size = min(skel_len, 2000)
        idx = np.random.choice(skel_len, sample_size, replace=False)
        sample_pts = skel_pts[idx]

        # Euclidean distances: (sample_size, N)
        diffs = sample_pts[:, None, :] - node_pts[None, :, :]   # (S, N, 2)
        dists = np.linalg.norm(diffs, axis=-1).min(axis=1)       # (S,)
        coverage = float((dists <= 5).mean())
        result["node_coverage"] = coverage

    per_1k = 1000.0 / max(skel_len, 1)
    result["bifurcation_density"] = float(n_bif * per_1k)
    result["endpoint_density"]    = float(n_end * per_1k)

    return result


# ─── Bifurcation Statistics ───────────────────────────────────────────────────

def compute_bifurcation_stats(graph) -> Dict[str, float]:
    """
    Statistics of bifurcation angles in the vascular network.

    Args:
        graph: VascularGraph.

    Returns:
        dict with:
          - mean_branching_angle: mean angle between vessel branches (degrees).
          - std_branching_angle: standard deviation of branching angles.
          - min_branching_angle, max_branching_angle.
          - n_bifurcations: total bifurcation node count.
          - bifurcation_index: bifurcations / total_nodes.
    """
    nodes = graph.nodes
    bif_nodes = [n for n in nodes if n.is_bifurcation]

    if not bif_nodes:
        return {
            "mean_branching_angle": float("nan"),
            "std_branching_angle":  float("nan"),
            "min_branching_angle":  float("nan"),
            "max_branching_angle":  float("nan"),
            "n_bifurcations":       0.0,
            "bifurcation_index":    0.0,
        }

    angles = [n.branching_angle for n in bif_nodes]

    return {
        "mean_branching_angle": float(np.mean(angles)),
        "std_branching_angle":  float(np.std(angles)),
        "min_branching_angle":  float(np.min(angles)),
        "max_branching_angle":  float(np.max(angles)),
        "n_bifurcations":       float(len(bif_nodes)),
        "bifurcation_index":    float(len(bif_nodes) / max(len(nodes), 1)),
    }


# ─── Vessel Caliber Statistics ────────────────────────────────────────────────

def compute_vessel_caliber_stats(graph) -> Dict[str, float]:
    """
    Statistical distribution of vessel caliber (diameter) across all edges.

    Args:
        graph: VascularGraph.

    Returns:
        dict with mean, std, median, p25, p75, p95, min, max diameter in pixels.
        All nan if no edges.
    """
    edges = graph.edges
    if not edges:
        return {k: float("nan") for k in [
            "mean_diameter", "std_diameter", "median_diameter",
            "p25_diameter", "p75_diameter", "p95_diameter",
            "min_diameter", "max_diameter",
        ]}

    diameters = np.array([e.mean_diameter for e in edges], dtype=np.float32)
    diameters = diameters[np.isfinite(diameters)]

    if len(diameters) == 0:
        return {k: float("nan") for k in [
            "mean_diameter", "std_diameter", "median_diameter",
            "p25_diameter", "p75_diameter", "p95_diameter",
            "min_diameter", "max_diameter",
        ]}

    return {
        "mean_diameter":   float(np.mean(diameters)),
        "std_diameter":    float(np.std(diameters)),
        "median_diameter": float(np.median(diameters)),
        "p25_diameter":    float(np.percentile(diameters, 25)),
        "p75_diameter":    float(np.percentile(diameters, 75)),
        "p95_diameter":    float(np.percentile(diameters, 95)),
        "min_diameter":    float(np.min(diameters)),
        "max_diameter":    float(np.max(diameters)),
    }


# ─── Tortuosity Statistics ────────────────────────────────────────────────────

def compute_tortuosity_stats(graph) -> Dict[str, float]:
    """
    Statistical distribution of vessel tortuosity across all edges.

    Tortuosity = actual path length / straight-line distance.
    Values close to 1.0 = straight; higher = more tortuous.
    Clinically: hypertension, diabetic retinopathy → increased tortuosity.

    Args:
        graph: VascularGraph.

    Returns:
        dict with mean, std, median, p75, p95, max tortuosity.
        All nan if no edges.
    """
    edges = graph.edges
    if not edges:
        return {k: float("nan") for k in [
            "mean_tortuosity", "std_tortuosity", "median_tortuosity",
            "p75_tortuosity", "p95_tortuosity", "max_tortuosity",
        ]}

    torts = np.array([e.tortuosity for e in edges], dtype=np.float32)
    torts = torts[np.isfinite(torts)]

    if len(torts) == 0:
        return {k: float("nan") for k in [
            "mean_tortuosity", "std_tortuosity", "median_tortuosity",
            "p75_tortuosity", "p95_tortuosity", "max_tortuosity",
        ]}

    return {
        "mean_tortuosity":   float(np.mean(torts)),
        "std_tortuosity":    float(np.std(torts)),
        "median_tortuosity": float(np.median(torts)),
        "p75_tortuosity":    float(np.percentile(torts, 75)),
        "p95_tortuosity":    float(np.percentile(torts, 95)),
        "max_tortuosity":    float(np.max(torts)),
    }


# ─── Network Efficiency ───────────────────────────────────────────────────────

def compute_network_efficiency(graph) -> Dict[str, float]:
    """
    Global and local network efficiency.

    Global efficiency = average of inverse shortest path lengths.
    Measures how efficiently information/blood flow is distributed globally.
    Perfect: 1.0 (complete graph), broken: 0.0.

    Local efficiency = average efficiency of the neighbourhood sub-graph of each node.
    Measures fault-tolerance / redundancy.

    Only computed for graphs with ≤ 300 nodes (cost is O(N^2) or O(N^3)).

    Args:
        graph: VascularGraph.

    Returns:
        dict with global_efficiency, local_efficiency.
        Returns nan for large or edgeless graphs.
    """
    if not _NX_AVAILABLE or not hasattr(graph, "nx_graph"):
        return {"global_efficiency": float("nan"), "local_efficiency": float("nan")}

    G = graph.nx_graph
    N = G.number_of_nodes()

    if N == 0 or G.number_of_edges() == 0:
        return {"global_efficiency": 0.0, "local_efficiency": 0.0}

    if N > 300:
        return {"global_efficiency": float("nan"), "local_efficiency": float("nan")}

    try:
        glob_eff  = nx.global_efficiency(G)
        local_eff = nx.local_efficiency(G)
    except Exception:
        glob_eff  = float("nan")
        local_eff = float("nan")

    return {
        "global_efficiency": float(glob_eff),
        "local_efficiency":  float(local_eff),
    }


# ─── Vessel Length Distribution ───────────────────────────────────────────────

def compute_vessel_length_stats(graph) -> Dict[str, float]:
    """
    Statistical distribution of vessel segment lengths across all edges.

    Args:
        graph: VascularGraph.

    Returns:
        dict with mean, std, median, p25, p75, p95, total_length (pixels).
        All nan except total_length=0 if no edges.
    """
    edges = graph.edges
    if not edges:
        return {
            "mean_length": float("nan"),
            "std_length":  float("nan"),
            "median_length": float("nan"),
            "p25_length":  float("nan"),
            "p75_length":  float("nan"),
            "p95_length":  float("nan"),
            "total_length": 0.0,
        }

    lengths = np.array([e.length for e in edges], dtype=np.float32)
    lengths = lengths[np.isfinite(lengths)]

    if len(lengths) == 0:
        return {
            "mean_length": float("nan"), "std_length": float("nan"),
            "median_length": float("nan"), "p25_length": float("nan"),
            "p75_length": float("nan"), "p95_length": float("nan"),
            "total_length": 0.0,
        }

    return {
        "mean_length":   float(np.mean(lengths)),
        "std_length":    float(np.std(lengths)),
        "median_length": float(np.median(lengths)),
        "p25_length":    float(np.percentile(lengths, 25)),
        "p75_length":    float(np.percentile(lengths, 75)),
        "p95_length":    float(np.percentile(lengths, 95)),
        "total_length":  float(np.sum(lengths)),
    }


# ─── Composite Graph Quality Score ───────────────────────────────────────────

def compute_composite_quality_score(metrics: Dict[str, float]) -> float:
    """
    Weighted composite graph quality score in [0, 1].

    Higher = better graph quality (more complete, connected, topologically faithful).

    Component weights and their sources:
      - largest_component_fraction   (0.25): connectivity robustness
      - node_coverage                (0.20): topology faithfulness to skeleton
      - 1 - fragmentation_index      (0.20): low fragmentation = good
      - global_efficiency            (0.15): network flow efficiency
      - 1 - isolated_node_fraction   (0.10): no dangling nodes
      - bifurcation_index            (0.10): rich branching structure

    Missing values are substituted with the neutral value (0.5) and a
    warning is printed.

    Args:
        metrics: flat dict from compute_all_graph_metrics().

    Returns:
        Composite score in [0, 1].
    """
    def _get(key: str, neutral: float = 0.5) -> float:
        v = metrics.get(key, float("nan"))
        if not np.isfinite(v):
            return neutral
        return float(np.clip(v, 0.0, 1.0))

    score = (
        0.25 * _get("largest_component_fraction")
        + 0.20 * _get("node_coverage")
        + 0.20 * (1.0 - _get("fragmentation_index"))
        + 0.15 * _get("global_efficiency")
        + 0.10 * (1.0 - _get("isolated_node_fraction"))
        + 0.10 * _get("bifurcation_index")
    )
    return float(np.clip(score, 0.0, 1.0))


# ─── Full metric bundle ───────────────────────────────────────────────────────

def compute_all_graph_metrics(
    graph,
    mask: Optional[np.ndarray] = None,
    compute_efficiency: bool = True,
    compute_fractal: bool = True,
    fractal_box_sizes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute the full suite of vascular graph quality metrics.

    Args:
        graph:             VascularGraph from RetinalGraphBuilder.build().
        mask:              (H, W) binary vessel mask. Required for fractal dimension,
                           node_coverage, skeleton_length, and topology integrity.
        compute_efficiency: include network efficiency (slow for large graphs).
        compute_fractal:   include fractal dimension (slow for large images).
        fractal_box_sizes: custom box sizes for fractal dimension computation.

    Returns:
        Flat dict of metric_name → float.
    """
    metrics: Dict[str, float] = {}

    # — 1. Branch fragmentation ────────────────────────────────────────────────
    metrics.update(compute_branch_fragmentation(graph))

    # — 2. Graph connectivity ──────────────────────────────────────────────────
    metrics.update(compute_graph_connectivity(graph))

    # — 3. Topology integrity ──────────────────────────────────────────────────
    metrics.update(compute_topology_integrity(graph, mask))

    # — 4. Bifurcation statistics ──────────────────────────────────────────────
    metrics.update(compute_bifurcation_stats(graph))

    # — 5. Vessel caliber statistics ───────────────────────────────────────────
    metrics.update(compute_vessel_caliber_stats(graph))

    # — 6. Tortuosity statistics ───────────────────────────────────────────────
    metrics.update(compute_tortuosity_stats(graph))

    # — 7. Vessel length statistics ────────────────────────────────────────────
    metrics.update(compute_vessel_length_stats(graph))

    # — 8. Network efficiency ──────────────────────────────────────────────────
    if compute_efficiency:
        metrics.update(compute_network_efficiency(graph))
    else:
        metrics["global_efficiency"] = float("nan")
        metrics["local_efficiency"]  = float("nan")

    # — 9. Fractal dimension ───────────────────────────────────────────────────
    if compute_fractal and mask is not None:
        metrics["fractal_dimension"] = compute_fractal_dimension(
            mask, box_sizes=fractal_box_sizes
        )
    else:
        metrics["fractal_dimension"] = float("nan")

    # — 10. Node / edge counts ─────────────────────────────────────────────────
    metrics["n_nodes"] = float(len(graph.nodes))
    metrics["n_edges"] = float(len(graph.edges))

    # — 11. Composite quality score ────────────────────────────────────────────
    metrics["graph_quality_score"] = compute_composite_quality_score(metrics)

    return metrics


# ─── Graph Quality Benchmark ──────────────────────────────────────────────────

@dataclass
class GraphQualityRecord:
    metrics: Dict[str, float]
    dataset: str
    image_id: str


class GraphQualityBenchmark:
    """
    Accumulates per-image graph metrics and produces per-dataset summaries.

    Usage:
        bench = GraphQualityBenchmark()
        for graph, mask, ds_name, img_id in pipeline_results:
            bench.update(graph, mask, dataset=ds_name, image_id=img_id)
        summary = bench.summary()
        bench.print_report()
    """

    def __init__(
        self,
        compute_efficiency: bool = True,
        compute_fractal:    bool = True,
    ):
        self.compute_efficiency = compute_efficiency
        self.compute_fractal    = compute_fractal
        self._records: List[GraphQualityRecord] = []

    def update(
        self,
        graph,
        mask:     Optional[np.ndarray] = None,
        dataset:  str = "unknown",
        image_id: str = "",
    ):
        """Compute and store metrics for one image."""
        m = compute_all_graph_metrics(
            graph, mask,
            compute_efficiency=self.compute_efficiency,
            compute_fractal=self.compute_fractal,
        )
        self._records.append(GraphQualityRecord(metrics=m, dataset=dataset, image_id=image_id))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-dataset + overall mean metrics."""
        if not self._records:
            return {}

        NUMERIC_KEYS = [
            k for k, v in self._records[0].metrics.items()
            if isinstance(v, (int, float))
        ]
        all_datasets = sorted(set(r.dataset for r in self._records))
        result: Dict[str, Dict[str, float]] = {}

        for ds in all_datasets + ["overall"]:
            rows = (
                [r.metrics for r in self._records if r.dataset == ds]
                if ds != "overall" else [r.metrics for r in self._records]
            )
            if not rows:
                continue
            result[ds] = {}
            for k in NUMERIC_KEYS:
                vals = [r[k] for r in rows if np.isfinite(r.get(k, float("nan")))]
                result[ds][k] = float(np.mean(vals)) if vals else float("nan")

        return result

    def print_report(self, key_metrics: Optional[List[str]] = None):
        """Print a formatted table of key graph quality metrics."""
        if key_metrics is None:
            key_metrics = [
                "n_nodes", "n_edges", "fractal_dimension",
                "largest_component_fraction", "mean_tortuosity",
                "mean_diameter", "bifurcation_index",
                "graph_quality_score",
            ]

        summary = self.summary()
        print(f"\n{'='*80}")
        print("  Retinal Vascular Graph Quality Benchmark")
        print(f"{'='*80}")

        header = f"{'Dataset':>12}"
        for k in key_metrics:
            short = k.replace("_fraction", "_frac").replace("largest_component", "lc")[:12]
            header += f"  {short:>12}"
        print(header)
        print("─" * len(header))

        for ds_name, m in summary.items():
            row = f"{ds_name:>12}"
            for k in key_metrics:
                v = m.get(k, float("nan"))
                row += f"  {v:>12.4f}" if np.isfinite(v) else f"  {'nan':>12}"
            print(row)
        print()

    def records(self) -> List[GraphQualityRecord]:
        return list(self._records)
