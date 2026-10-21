"""
Graph Pipeline Integration Tests.

Tests cover:
  1. Graph metrics (graph_metrics.py)
  2. Branch hierarchy (branch_hierarchy.py)
  3. End-to-end graph pipeline (graph_pipeline.py)
  4. Edge cases: empty graph, no-edge graph, single-node graph

All tests use synthetic data only — no real dataset files required.
Uses only standard library + numpy for the graph skeleton tests,
and torch only where strictly needed.

Run with:
    python -m pytest tests/test_graph_pipeline.py -v
or directly:
    python tests/test_graph_pipeline.py
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

import numpy as np


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _passed(name: str):
    print(f"  PASS  {name}")


def _failed(name: str, exc: Exception):
    print(f"  FAIL  {name}: {exc}")
    traceback.print_exc()
    return False


def _make_synthetic_mask(H: int = 64, W: int = 64, vessel_frac: float = 0.12) -> np.ndarray:
    """Create a synthetic binary mask with simple vessel-like structures."""
    rng  = np.random.default_rng(seed=42)
    mask = np.zeros((H, W), dtype=np.float32)
    # Horizontal and vertical lines as vessel proxies
    for row in [H // 4, H // 2, 3 * H // 4]:
        mask[row, :] = 1.0
    for col in [W // 4, W // 2, 3 * W // 4]:
        mask[:, col] = 1.0
    # Add random noise vessels
    noise = rng.random((H, W)) < (vessel_frac / 4)
    mask = np.clip(mask + noise.astype(np.float32), 0.0, 1.0)
    return mask


def _make_synthetic_prob_map(H: int = 64, W: int = 64) -> np.ndarray:
    """Probability map with gradient — non-binary so postprocessing exercises thresholding."""
    mask = _make_synthetic_mask(H, W)
    rng  = np.random.default_rng(seed=7)
    noise = rng.random((H, W)).astype(np.float32) * 0.15
    return np.clip(mask * 0.85 + noise, 0.0, 1.0)


def _make_vascular_graph(n_nodes: int = 15, n_edges: int = 20, seed: int = 0):
    """Build a synthetic VascularGraph from scratch via RetinalGraphBuilder."""
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder
    mask    = _make_synthetic_mask()
    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    graph   = builder.build(mask, intensity_image=mask)
    return graph


# ─── 1. Fractal Dimension ─────────────────────────────────────────────────────

def test_fractal_dimension_range():
    """Fractal dimension must be in [1.0, 2.0] for non-trivial masks."""
    from rapid_twin.graph.graph_metrics import compute_fractal_dimension

    mask = _make_synthetic_mask(128, 128)
    D = compute_fractal_dimension(mask)
    assert 1.0 <= D <= 2.0, f"Fractal dimension {D} out of [1, 2]"
    _passed("fractal_dimension_range")


def test_fractal_dimension_empty_mask():
    """Empty mask must return 1.0 without error."""
    from rapid_twin.graph.graph_metrics import compute_fractal_dimension

    empty = np.zeros((64, 64), dtype=np.float32)
    D = compute_fractal_dimension(empty)
    assert D == 1.0, f"Expected 1.0 for empty mask, got {D}"
    _passed("fractal_dimension_empty_mask")


def test_fractal_dimension_full_mask():
    """Full mask (all vessels) should give D close to 2.0."""
    from rapid_twin.graph.graph_metrics import compute_fractal_dimension

    full = np.ones((64, 64), dtype=np.float32)
    D = compute_fractal_dimension(full)
    assert D >= 1.8, f"Full mask fractal dim should be near 2, got {D}"
    _passed("fractal_dimension_full_mask")


# ─── 2. Graph Fragmentation ───────────────────────────────────────────────────

def test_branch_fragmentation_normal_graph():
    """Fragmentation metrics must be in valid ranges for a non-trivial graph."""
    from rapid_twin.graph.graph_metrics import compute_branch_fragmentation

    graph = _make_vascular_graph()
    metrics = compute_branch_fragmentation(graph)

    assert "fragmentation_index"        in metrics
    assert "isolated_node_fraction"     in metrics
    assert "n_connected_components"     in metrics
    assert "largest_component_fraction" in metrics

    fi = metrics["fragmentation_index"]
    assert 0.0 <= fi <= 1.0, f"fragmentation_index out of [0,1]: {fi}"

    lf = metrics["largest_component_fraction"]
    assert 0.0 <= lf <= 1.0, f"largest_component_fraction out of [0,1]: {lf}"

    _passed("branch_fragmentation_normal_graph")


def test_branch_fragmentation_empty_graph():
    """Fragmentation metrics on empty graph must not crash."""
    from rapid_twin.graph.graph_metrics import compute_branch_fragmentation
    from rapid_twin.graph.graph_builder import VascularGraph
    import networkx as nx

    empty_graph = VascularGraph([], [], nx.Graph(), np.zeros((64, 64), np.uint8),
                                np.zeros((0, 16), np.float32),
                                np.zeros((0, 8), np.float32),
                                np.zeros((2, 0), np.int64))
    metrics = compute_branch_fragmentation(empty_graph)
    assert metrics["fragmentation_index"] == 1.0
    _passed("branch_fragmentation_empty_graph")


# ─── 3. Topology Integrity ────────────────────────────────────────────────────

def test_topology_integrity_with_mask():
    """Topology integrity with mask must produce finite, bounded metrics."""
    from rapid_twin.graph.graph_metrics import compute_topology_integrity

    graph = _make_vascular_graph()
    mask  = _make_synthetic_mask()
    metrics = compute_topology_integrity(graph, mask)

    assert "n_bifurcations"     in metrics
    assert "n_endpoints"        in metrics
    assert "bif_to_end_ratio"   in metrics
    assert "skeleton_length"    in metrics

    assert metrics["skeleton_length"] >= 0
    assert metrics["n_bifurcations"]  >= 0
    assert metrics["n_endpoints"]     >= 0
    _passed("topology_integrity_with_mask")


def test_topology_integrity_no_mask():
    """Topology integrity without mask must skip mask-based metrics (return nan)."""
    from rapid_twin.graph.graph_metrics import compute_topology_integrity

    graph = _make_vascular_graph()
    metrics = compute_topology_integrity(graph, mask=None)

    assert math.isnan(metrics["node_coverage"]), \
        "node_coverage should be nan when mask is None"
    _passed("topology_integrity_no_mask")


# ─── 4. Vessel Caliber and Tortuosity ─────────────────────────────────────────

def test_vessel_caliber_stats():
    """Caliber stats must be finite positive values for a non-empty graph."""
    from rapid_twin.graph.graph_metrics import compute_vessel_caliber_stats

    graph = _make_vascular_graph()
    if not graph.edges:
        _passed("vessel_caliber_stats (no edges — skipped)")
        return

    metrics = compute_vessel_caliber_stats(graph)
    for k, v in metrics.items():
        if math.isnan(v):
            continue
        assert v >= 0, f"{k} must be non-negative, got {v}"
    assert metrics["mean_diameter"] > 0
    _passed("vessel_caliber_stats")


def test_tortuosity_stats_range():
    """Tortuosity values must be >= 1.0 (actual path >= straight-line distance)."""
    from rapid_twin.graph.graph_metrics import compute_tortuosity_stats

    graph = _make_vascular_graph()
    if not graph.edges:
        _passed("tortuosity_stats_range (no edges — skipped)")
        return

    metrics = compute_tortuosity_stats(graph)
    mean_t = metrics["mean_tortuosity"]
    if not math.isnan(mean_t):
        assert mean_t >= 1.0, f"mean_tortuosity {mean_t} < 1.0 (impossible)"
    _passed("tortuosity_stats_range")


# ─── 5. Network Efficiency ────────────────────────────────────────────────────

def test_network_efficiency_range():
    """Global and local efficiency must be in [0, 1]."""
    from rapid_twin.graph.graph_metrics import compute_network_efficiency

    graph = _make_vascular_graph()
    metrics = compute_network_efficiency(graph)

    for k, v in metrics.items():
        if not math.isnan(v):
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
    _passed("network_efficiency_range")


# ─── 6. Composite Quality Score ───────────────────────────────────────────────

def test_composite_quality_score_range():
    """Composite quality score must be in [0, 1]."""
    from rapid_twin.graph.graph_metrics import (
        compute_all_graph_metrics, compute_composite_quality_score
    )

    graph   = _make_vascular_graph()
    mask    = _make_synthetic_mask()
    metrics = compute_all_graph_metrics(
        graph, mask, compute_efficiency=True, compute_fractal=True
    )

    score = metrics["graph_quality_score"]
    assert 0.0 <= score <= 1.0, f"graph_quality_score={score} out of [0,1]"
    _passed("composite_quality_score_range")


def test_composite_score_all_nan_inputs():
    """Composite score with all-nan inputs must not crash and return [0,1]."""
    from rapid_twin.graph.graph_metrics import compute_composite_quality_score

    nan_metrics = {k: float("nan") for k in [
        "largest_component_fraction", "node_coverage", "fragmentation_index",
        "global_efficiency", "isolated_node_fraction", "bifurcation_index",
    ]}
    score = compute_composite_quality_score(nan_metrics)
    assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] with nan inputs"
    _passed("composite_score_all_nan_inputs")


# ─── 7. compute_all_graph_metrics ────────────────────────────────────────────

def test_all_graph_metrics_keys():
    """compute_all_graph_metrics must return all expected keys."""
    from rapid_twin.graph.graph_metrics import compute_all_graph_metrics

    graph   = _make_vascular_graph()
    mask    = _make_synthetic_mask()
    metrics = compute_all_graph_metrics(
        graph, mask, compute_efficiency=True, compute_fractal=True
    )

    required_keys = [
        "fragmentation_index", "isolated_node_fraction", "n_connected_components",
        "largest_component_fraction", "n_bifurcations", "n_endpoints",
        "bif_to_end_ratio", "mean_diameter", "mean_tortuosity",
        "mean_length", "total_length", "n_nodes", "n_edges",
        "graph_quality_score", "fractal_dimension",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    _passed("all_graph_metrics_keys")


def test_all_graph_metrics_no_mask():
    """compute_all_graph_metrics without mask must run without error."""
    from rapid_twin.graph.graph_metrics import compute_all_graph_metrics

    graph   = _make_vascular_graph()
    metrics = compute_all_graph_metrics(graph, mask=None,
                                        compute_efficiency=False,
                                        compute_fractal=False)
    assert "graph_quality_score" in metrics
    _passed("all_graph_metrics_no_mask")


# ─── 8. Branch Hierarchy ──────────────────────────────────────────────────────

def test_build_branch_hierarchy_normal():
    """Branch hierarchy must produce a valid tree with correct node count."""
    from rapid_twin.graph.branch_hierarchy import build_branch_hierarchy

    graph = _make_vascular_graph()
    if not graph.nodes:
        _passed("build_branch_hierarchy_normal (empty graph — skipped)")
        return

    tree = build_branch_hierarchy(graph)
    assert tree.is_valid, "BranchTree.is_valid must be True for non-empty graph"
    assert tree.root_id in tree.nodes, "Root must appear in tree.nodes"
    assert tree.max_depth >= 0
    assert tree.n_leaves >= 0

    # Every node in tree.nodes must have a valid depth
    for nid, bn in tree.nodes.items():
        assert bn.depth >= 0, f"Node {nid} has negative depth {bn.depth}"

    _passed("build_branch_hierarchy_normal")


def test_build_branch_hierarchy_empty_graph():
    """Branch hierarchy on empty graph must return is_valid=False."""
    from rapid_twin.graph.branch_hierarchy import build_branch_hierarchy
    from rapid_twin.graph.graph_builder import VascularGraph
    import networkx as nx

    empty = VascularGraph([], [], nx.Graph(), np.zeros((32, 32), np.uint8),
                          np.zeros((0, 16), np.float32),
                          np.zeros((0, 8), np.float32),
                          np.zeros((2, 0), np.int64))
    tree = build_branch_hierarchy(empty)
    assert not tree.is_valid, "Empty graph must produce is_valid=False tree"
    _passed("build_branch_hierarchy_empty_graph")


def test_annotate_branch_levels():
    """Level map must contain all node ids, with root at depth 0."""
    from rapid_twin.graph.branch_hierarchy import (
        build_branch_hierarchy, annotate_branch_levels
    )

    graph = _make_vascular_graph()
    if not graph.nodes:
        _passed("annotate_branch_levels (empty graph — skipped)")
        return

    tree      = build_branch_hierarchy(graph)
    level_map = annotate_branch_levels(graph, tree)

    all_ids = {n.node_id for n in graph.nodes}
    assert set(level_map.keys()) == all_ids, (
        f"level_map is missing node ids: {all_ids - set(level_map.keys())}"
    )
    assert level_map[tree.root_id] == 0, "Root must have depth 0"
    _passed("annotate_branch_levels")


def test_hierarchy_stats_valid():
    """compute_hierarchy_stats must return finite positive values."""
    from rapid_twin.graph.branch_hierarchy import (
        build_branch_hierarchy, compute_hierarchy_stats
    )

    graph = _make_vascular_graph()
    tree  = build_branch_hierarchy(graph)
    stats = compute_hierarchy_stats(tree)

    required = [
        "max_depth", "mean_depth", "std_depth", "n_leaves",
        "n_internal", "branching_ratio", "asymmetry_index",
        "root_subtree_size", "depth_entropy", "is_valid",
    ]
    for k in required:
        assert k in stats, f"Missing hierarchy stat: {k}"
        v = stats[k]
        assert math.isfinite(v), f"{k}={v} is not finite"

    if tree.is_valid:
        assert stats["is_valid"] == 1.0
        assert stats["max_depth"] >= 0
        assert stats["n_leaves"] >= 0

    _passed("hierarchy_stats_valid")


def test_analyse_branch_hierarchy_returns_three():
    """analyse_branch_hierarchy must return (tree, level_map, stats)."""
    from rapid_twin.graph.branch_hierarchy import analyse_branch_hierarchy

    graph = _make_vascular_graph()
    result = analyse_branch_hierarchy(graph)
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    tree, level_map, stats = result
    assert isinstance(level_map, dict)
    assert isinstance(stats, dict)
    _passed("analyse_branch_hierarchy_returns_three")


# ─── 9. End-to-End Graph Pipeline ─────────────────────────────────────────────

def test_pipeline_basic_run():
    """VascularGraphPipeline.run on a synthetic prob map must succeed."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    prob = _make_synthetic_prob_map(64, 64)
    pipeline = VascularGraphPipeline(min_branch_length=2, max_nodes=256, verbose=False)
    result = pipeline.run(prob)

    assert result.success, f"Pipeline failed: {result.error}"
    assert result.binary_mask is not None
    assert result.binary_mask.shape == (64, 64)
    assert result.binary_mask.dtype == np.float32
    assert set(result.binary_mask.flatten().tolist()).issubset({0.0, 1.0}), \
        "binary_mask must be strictly binary {0.0, 1.0}"

    assert result.vascular_graph is not None
    assert "graph_quality_score" in result.graph_metrics
    assert 0.0 <= result.graph_metrics["graph_quality_score"] <= 1.0

    _passed("pipeline_basic_run")


def test_pipeline_with_intensity():
    """Pipeline with intensity image must produce the same shape output."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    prob      = _make_synthetic_prob_map(64, 64)
    intensity = np.random.rand(64, 64).astype(np.float32)
    pipeline  = VascularGraphPipeline(min_branch_length=2, max_nodes=256)
    result    = pipeline.run(prob, intensity_image=intensity)

    assert result.success, f"Pipeline with intensity failed: {result.error}"
    assert result.binary_mask.shape == (64, 64)
    _passed("pipeline_with_intensity")


def test_pipeline_empty_mask():
    """Pipeline on all-zero prob map must succeed with is_valid=False graph metrics."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    empty_prob = np.zeros((64, 64), dtype=np.float32)
    pipeline   = VascularGraphPipeline(min_branch_length=2, max_nodes=256)
    result     = pipeline.run(empty_prob)

    assert result.success, f"Pipeline failed on empty mask: {result.error}"
    assert result.n_nodes == 0
    _passed("pipeline_empty_mask")


def test_pipeline_invalid_input_ndim():
    """Pipeline must fail cleanly on 3-D prob_map instead of crashing."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    bad_input = np.zeros((3, 64, 64), dtype=np.float32)
    pipeline  = VascularGraphPipeline()
    result    = pipeline.run(bad_input)

    assert not result.success, "Pipeline should report failure on 3-D input"
    assert result.error is not None
    _passed("pipeline_invalid_input_ndim")


def test_pipeline_timing_populated():
    """Timing dict must have entries for completed stages."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    prob     = _make_synthetic_prob_map(64, 64)
    pipeline = VascularGraphPipeline(min_branch_length=2, verbose=False)
    result   = pipeline.run(prob)

    assert result.success
    assert "postprocess"    in result.timing
    assert "graph_build"    in result.timing
    assert "total"          in result.timing
    assert result.timing["total"] > 0
    _passed("pipeline_timing_populated")


def test_pipeline_with_gnn():
    """Pipeline with GNN model must produce a finite embedding tensor."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline
    from rapid_twin.models.gnn import RetinalGNN
    import torch

    prob     = _make_synthetic_prob_map(64, 64)
    gnn      = RetinalGNN(node_in_dim=16, edge_in_dim=8, hidden_dim=32, out_dim=64,
                          num_heads=2, num_layers=2)
    pipeline = VascularGraphPipeline(min_branch_length=2, max_nodes=256)
    result   = pipeline.run(prob, gnn_model=gnn, device="cpu")

    assert result.success
    assert result.gnn_embedding is not None
    emb = result.gnn_embedding
    assert tuple(emb.shape) == (1, 64), f"Expected (1, 64), got {tuple(emb.shape)}"
    assert torch.isfinite(emb).all(), "GNN embedding contains non-finite values"
    _passed("pipeline_with_gnn")


def test_pipeline_batch_mode():
    """run_batch must return a list of results with matching length."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    probs    = [_make_synthetic_prob_map(64, 64) for _ in range(3)]
    pipeline = VascularGraphPipeline(min_branch_length=2)
    results  = pipeline.run_batch(probs)

    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.success, f"Batch result {i} failed: {r.error}"
    _passed("pipeline_batch_mode")


def test_pipeline_print_result():
    """print_result must not raise on a successful result."""
    from rapid_twin.graph.graph_pipeline import VascularGraphPipeline

    prob     = _make_synthetic_prob_map(64, 64)
    pipeline = VascularGraphPipeline(min_branch_length=2)
    result   = pipeline.run(prob)

    VascularGraphPipeline.print_result(result)   # must not raise
    _passed("pipeline_print_result")


# ─── 10. GraphQualityBenchmark ────────────────────────────────────────────────

def test_graph_quality_benchmark():
    """GraphQualityBenchmark must accumulate and summarise correctly."""
    from rapid_twin.graph.graph_metrics import GraphQualityBenchmark

    bench = GraphQualityBenchmark(compute_efficiency=False, compute_fractal=False)
    mask  = _make_synthetic_mask()

    for i in range(3):
        graph = _make_vascular_graph(seed=i)
        bench.update(graph, mask=mask, dataset="synthetic", image_id=f"img{i}")

    assert len(bench.records()) == 3

    summary = bench.summary()
    assert "synthetic" in summary
    assert "overall"   in summary
    assert "graph_quality_score" in summary["overall"]

    q_score = summary["overall"]["graph_quality_score"]
    assert 0.0 <= q_score <= 1.0, f"quality score {q_score} out of [0,1]"

    bench.print_report()
    _passed("graph_quality_benchmark")


# ─── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_fractal_dimension_range,
    test_fractal_dimension_empty_mask,
    test_fractal_dimension_full_mask,
    test_branch_fragmentation_normal_graph,
    test_branch_fragmentation_empty_graph,
    test_topology_integrity_with_mask,
    test_topology_integrity_no_mask,
    test_vessel_caliber_stats,
    test_tortuosity_stats_range,
    test_network_efficiency_range,
    test_composite_quality_score_range,
    test_composite_score_all_nan_inputs,
    test_all_graph_metrics_keys,
    test_all_graph_metrics_no_mask,
    test_build_branch_hierarchy_normal,
    test_build_branch_hierarchy_empty_graph,
    test_annotate_branch_levels,
    test_hierarchy_stats_valid,
    test_analyse_branch_hierarchy_returns_three,
    test_pipeline_basic_run,
    test_pipeline_with_intensity,
    test_pipeline_empty_mask,
    test_pipeline_invalid_input_ndim,
    test_pipeline_timing_populated,
    test_pipeline_with_gnn,
    test_pipeline_batch_mode,
    test_pipeline_print_result,
    test_graph_quality_benchmark,
]


def run_all_tests():
    passed = 0
    failed = 0
    print(f"\n{'='*65}")
    print("  RAPiD-Twin Graph Pipeline Test Suite")
    print(f"{'='*65}\n")

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            _failed(test_fn.__name__, exc)
            failed += 1

    print(f"\n{'='*65}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"{'='*65}\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
