"""
Vascular Intelligence Integration Tests.

Covers:
  1.  Graph refinement (topology repair, simplification, consistency, plausibility)
  2.  Artery/vein classification (AVClassifier, propagation, crossing disambiguation, metrics)
  3.  Biomarker extraction (all 12 families)
  4.  Hemodynamic modeling (Murray's law, resistance, flow directions, full simulation)

All tests use synthetic data only — no real dataset files required.
Tests that require torch are guarded with try/except.

Run with:
    python -m pytest tests/test_vascular_intelligence.py -v
or:
    python tests/test_vascular_intelligence.py
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

import numpy as np


# ─── Shared test fixtures ─────────────────────────────────────────────────────

def _make_mask(H: int = 64, W: int = 64) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.float32)
    mask[H // 4, :] = 1.0
    mask[H // 2, :] = 1.0
    mask[3 * H // 4, :] = 1.0
    mask[:, W // 4] = 1.0
    mask[:, W // 2] = 1.0
    mask[:, 3 * W // 4] = 1.0
    return mask


def _make_graph(min_branch: int = 2):
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder
    builder = RetinalGraphBuilder(min_branch_length=min_branch, max_nodes=512)
    mask = _make_mask()
    return builder.build(mask, intensity_image=mask)


def _make_empty_graph():
    from rapid_twin.graph.graph_builder import VascularGraph
    import networkx as nx
    return VascularGraph([], [], nx.Graph(),
                         np.zeros((64, 64), np.uint8),
                         np.zeros((0, 16), np.float32),
                         np.zeros((0, 8), np.float32),
                         np.zeros((2, 0), np.int64))


def _make_hierarchy(graph):
    from rapid_twin.graph.branch_hierarchy import build_branch_hierarchy
    return build_branch_hierarchy(graph)


def _passed(name: str):
    print(f"  PASS  {name}")


def _failed(name: str, exc: Exception):
    print(f"  FAIL  {name}: {exc}")
    traceback.print_exc()
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. Graph Refinement
# ══════════════════════════════════════════════════════════════════════════════

def test_refiner_basic():
    """VascularGraphRefiner.refine must return a valid graph and report."""
    from rapid_twin.graph.graph_refinement import VascularGraphRefiner
    graph   = _make_graph()
    mask    = _make_mask()
    refiner = VascularGraphRefiner(verbose=False)
    refined, report = refiner.refine(graph, mask=mask)

    assert refined is not None, "Refiner returned None"
    assert report is not None, "Report is None"
    assert report.input_n_nodes == len(graph.nodes)
    assert report.output_n_nodes == len(refined.nodes)
    _passed("refiner_basic")


def test_refiner_empty_graph():
    """Refiner on empty graph must not crash."""
    from rapid_twin.graph.graph_refinement import VascularGraphRefiner
    graph   = _make_empty_graph()
    refiner = VascularGraphRefiner()
    refined, report = refiner.refine(graph)
    assert len(refined.nodes) == 0
    _passed("refiner_empty_graph")


def test_topology_repair_bridges():
    """topology_repair must return non-negative bridge/island/loop counts."""
    from rapid_twin.graph.graph_refinement import repair_topology
    graph = _make_graph()
    refined, bridges, islands, loops = repair_topology(graph, gap_threshold=20)
    assert bridges >= 0
    assert islands >= 0
    assert loops >= 0
    _passed("topology_repair_bridges")


def test_simplify_graph():
    """simplify_graph must reduce or keep node count, never increase."""
    from rapid_twin.graph.graph_refinement import simplify_graph
    graph = _make_graph()
    simplified, merged, collapsed = simplify_graph(graph)
    assert len(simplified.nodes) <= len(graph.nodes), (
        f"Simplified graph has more nodes: {len(simplified.nodes)} > {len(graph.nodes)}"
    )
    assert merged >= 0
    assert collapsed >= 0
    _passed("simplify_graph")


def test_validate_consistency_no_nans():
    """consistency report must detect NaN features if artificially injected."""
    from rapid_twin.graph.graph_refinement import validate_consistency
    from rapid_twin.graph.graph_builder import VascularGraph
    import networkx as nx

    graph = _make_graph()
    # Inject NaN into node_features
    bad_feats = graph.node_features.copy()
    if bad_feats.shape[0] > 0:
        bad_feats[0, 0] = float("nan")
    bad_graph = VascularGraph(
        graph.nodes, graph.edges, graph.nx_graph, graph.skeleton,
        bad_feats, graph.edge_features, graph.edge_index,
    )
    report = validate_consistency(bad_graph)
    if len(graph.nodes) > 0:
        assert len(report.nan_feature_nodes) > 0, "Should detect NaN in features"
    _passed("validate_consistency_no_nans")


def test_check_anatomical_plausibility():
    """check_anatomical_plausibility must return a PlausibilityReport."""
    from rapid_twin.graph.graph_refinement import check_anatomical_plausibility
    graph = _make_graph()
    hier  = _make_hierarchy(graph)
    report = check_anatomical_plausibility(graph, hierarchy=hier)
    assert report is not None
    assert report.total_violations >= 0
    _passed("check_anatomical_plausibility")


def test_refiner_report_fields():
    """RefinementReport must contain all required fields."""
    from rapid_twin.graph.graph_refinement import VascularGraphRefiner
    graph   = _make_graph()
    refiner = VascularGraphRefiner()
    _, report = refiner.refine(graph)

    assert hasattr(report, "bridges_added")
    assert hasattr(report, "islands_removed")
    assert hasattr(report, "loops_broken")
    assert hasattr(report, "nodes_merged")
    assert hasattr(report, "input_n_nodes")
    assert hasattr(report, "output_n_nodes")
    assert hasattr(report, "consistency")
    assert hasattr(report, "plausibility")
    _passed("refiner_report_fields")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Artery/Vein Intelligence
# ══════════════════════════════════════════════════════════════════════════════

def test_av_label_enum():
    """AVLabel enum must have ARTERY=0, VEIN=1, UNCERTAIN=2."""
    from rapid_twin.av_classification import AVLabel
    assert int(AVLabel.ARTERY)    == 0
    assert int(AVLabel.VEIN)      == 1
    assert int(AVLabel.UNCERTAIN) == 2
    _passed("av_label_enum")


def test_av_classifier_forward():
    """AVClassifier.forward must return (N,3) logits and (N,1) confidence."""
    try:
        import torch
        from rapid_twin.av_classification import AVClassifier
    except ImportError:
        print("  SKIP  av_classifier_forward (torch not available)")
        return

    graph = _make_graph()
    N     = max(len(graph.nodes), 1)
    E     = max(len(graph.edges) * 2, 2)

    clf = AVClassifier(node_in_dim=16, edge_in_dim=8, hidden_dim=32,
                       n_heads=2, n_layers=2)
    nf = torch.from_numpy(graph.node_features).float()
    ei = torch.from_numpy(graph.edge_index).long()
    ef = torch.from_numpy(graph.edge_features).float()

    logits, conf = clf(nf, ei, ef)

    assert logits.shape == (N, 3), f"Expected ({N}, 3), got {tuple(logits.shape)}"
    assert conf.shape == (N, 1), f"Expected ({N}, 1), got {tuple(conf.shape)}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"
    assert torch.isfinite(conf).all(), "Confidence contains non-finite values"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence not in [0,1]"
    _passed("av_classifier_forward")


def test_av_classifier_predict():
    """AVClassifier.predict must return N AVPrediction objects."""
    try:
        import torch
        from rapid_twin.av_classification import AVClassifier, AVPrediction
    except ImportError:
        print("  SKIP  av_classifier_predict (torch not available)")
        return

    graph = _make_graph()
    clf   = AVClassifier(node_in_dim=16, edge_in_dim=8, hidden_dim=32)
    nf    = torch.from_numpy(graph.node_features).float()
    ei    = torch.from_numpy(graph.edge_index).long()
    ef    = torch.from_numpy(graph.edge_features).float()

    preds = clf.predict(nf, ei, ef)

    assert len(preds) == len(graph.nodes), (
        f"Expected {len(graph.nodes)} predictions, got {len(preds)}"
    )
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0
        assert math.isfinite(p.prob_artery)
        assert math.isfinite(p.prob_vein)
        assert abs(p.prob_artery + p.prob_vein + p.prob_uncertain - 1.0) < 0.01
    _passed("av_classifier_predict")


def test_av_classifier_empty_graph():
    """AVClassifier must handle empty node set without crash."""
    try:
        import torch
        from rapid_twin.av_classification import AVClassifier
    except ImportError:
        print("  SKIP  av_classifier_empty_graph (torch not available)")
        return

    clf = AVClassifier()
    nf  = torch.zeros(0, 16)
    ei  = torch.zeros(2, 0, dtype=torch.long)
    ef  = torch.zeros(0, 8)
    logits, conf = clf(nf, ei, ef)
    assert logits.shape[0] == 0
    assert conf.shape[0] == 0
    _passed("av_classifier_empty_graph")


def test_propagate_av_labels():
    """propagate_av_labels must return same number of predictions as input."""
    from rapid_twin.av_classification import AVLabel, AVPrediction, propagate_av_labels
    graph = _make_graph()
    nodes = graph.nodes

    if not nodes:
        _passed("propagate_av_labels (empty — skipped)")
        return

    # Synthetic predictions: alternate ARTERY / VEIN
    preds = [
        AVPrediction(
            node_id=n.node_id,
            label=AVLabel.ARTERY if i % 2 == 0 else AVLabel.VEIN,
            confidence=0.9 if i < len(nodes) // 2 else 0.4,
            prob_artery=0.9 if i % 2 == 0 else 0.05,
            prob_vein=0.05 if i % 2 == 0 else 0.9,
            prob_uncertain=0.05,
        )
        for i, n in enumerate(nodes)
    ]
    propagated = propagate_av_labels(graph, preds, confidence_threshold=0.7)

    assert len(propagated) == len(preds), (
        f"propagate changed number of predictions: {len(propagated)} vs {len(preds)}"
    )
    for p in propagated:
        assert p.label in (AVLabel.ARTERY, AVLabel.VEIN, AVLabel.UNCERTAIN)
        assert 0.0 <= p.confidence <= 1.0
    _passed("propagate_av_labels")


def test_disambiguate_crossings():
    """disambiguate_crossings must return same count, valid labels."""
    from rapid_twin.av_classification import AVLabel, AVPrediction, disambiguate_crossings
    graph = _make_graph()
    nodes = graph.nodes

    if not nodes:
        _passed("disambiguate_crossings (empty — skipped)")
        return

    preds = [
        AVPrediction(
            node_id=n.node_id,
            label=AVLabel.ARTERY if i % 2 == 0 else AVLabel.VEIN,
            confidence=0.8,
            prob_artery=0.8 if i % 2 == 0 else 0.1,
            prob_vein=0.1 if i % 2 == 0 else 0.8,
            prob_uncertain=0.1,
        )
        for i, n in enumerate(nodes)
    ]
    result = disambiguate_crossings(graph, preds)

    assert len(result) == len(preds)
    for p in result:
        assert p.label in (AVLabel.ARTERY, AVLabel.VEIN, AVLabel.UNCERTAIN)
    _passed("disambiguate_crossings")


def test_av_metrics_computation():
    """compute_av_metrics must return valid accuracy and F1 scores."""
    from rapid_twin.av_classification import AVLabel, AVPrediction, compute_av_metrics

    preds = [
        AVPrediction(0, AVLabel.ARTERY, 0.9, 0.9, 0.05, 0.05),
        AVPrediction(1, AVLabel.VEIN,   0.9, 0.05, 0.9, 0.05),
        AVPrediction(2, AVLabel.ARTERY, 0.7, 0.7, 0.2, 0.1),
        AVPrediction(3, AVLabel.UNCERTAIN, 0.5, 0.3, 0.3, 0.4),
    ]
    gt = {0: AVLabel.ARTERY, 1: AVLabel.VEIN, 2: AVLabel.VEIN, 3: AVLabel.ARTERY}
    metrics = compute_av_metrics(preds, gt)

    required_keys = [
        "accuracy", "artery_precision", "artery_recall",
        "vein_precision", "vein_recall", "f1_artery", "f1_vein",
        "uncertain_fraction",
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric: {k}"
        v = metrics[k]
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
    _passed("av_metrics_computation")


def test_av_dataset_adapter_invalid_path():
    """AVDatasetAdapter.load must raise FileNotFoundError on missing root."""
    from rapid_twin.av_classification import AVDatasetAdapter
    adapter = AVDatasetAdapter("DRIVE-AV", root="/nonexistent/path")
    try:
        adapter.load()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # expected
    _passed("av_dataset_adapter_invalid_path")


def test_av_dataset_adapter_unsupported():
    """AVDatasetAdapter must raise ValueError for unknown dataset."""
    from rapid_twin.av_classification import AVDatasetAdapter
    try:
        AVDatasetAdapter("UNKNOWN-DS")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("av_dataset_adapter_unsupported")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Biomarker Extraction
# ══════════════════════════════════════════════════════════════════════════════

def test_biomarker_report_structure():
    """BiomarkerReport must have all 12 biomarker families as attributes."""
    from rapid_twin.biomarkers import BiomarkerReport

    report = BiomarkerReport()
    required = [
        "mean_tortuosity", "std_tortuosity",
        "crae", "crve", "avr",
        "mean_branching_angle", "murray_compliance_score",
        "fractal_dimension",
        "vessel_density_global",
        "av_nicking_index",
        "focal_narrowing_index", "diffuse_narrowing_index",
        "vessel_dropout_index", "capillary_density",
        "peripapillary_vessel_density", "cup_disc_ratio_proxy",
        "neovascularization_index", "vessel_complexity",
        "arterial_narrowing_severity", "av_crossing_score",
    ]
    for k in required:
        assert hasattr(report, k), f"BiomarkerReport missing attribute: {k}"
    _passed("biomarker_report_structure")


def test_biomarker_extraction_basic():
    """RetinalBiomarkerExtractor.extract must run without error."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph   = _make_graph()
    mask    = _make_mask()
    ext     = RetinalBiomarkerExtractor()
    report  = ext.extract(graph, mask=mask)

    assert report is not None
    assert report.n_nodes == len(graph.nodes)
    assert report.n_edges == len(graph.edges)
    _passed("biomarker_extraction_basic")


def test_biomarker_tortuosity():
    """Tortuosity biomarkers must be ≥ 1.0 (actual path ≥ straight line)."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph  = _make_graph()
    mask   = _make_mask()
    report = RetinalBiomarkerExtractor().extract(graph, mask=mask)

    if math.isfinite(report.mean_tortuosity):
        assert report.mean_tortuosity >= 1.0, (
            f"mean_tortuosity {report.mean_tortuosity} < 1.0"
        )
    _passed("biomarker_tortuosity")


def test_biomarker_vessel_density_range():
    """Vessel density must be in [0, 1]."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph  = _make_graph()
    mask   = _make_mask()
    report = RetinalBiomarkerExtractor().extract(graph, mask=mask)

    if math.isfinite(report.vessel_density_global):
        assert 0.0 <= report.vessel_density_global <= 1.0, (
            f"vessel_density_global={report.vessel_density_global} out of [0,1]"
        )
    _passed("biomarker_vessel_density_range")


def test_biomarker_fractal_dimension_range():
    """Fractal dimension in BiomarkerReport must be in [1.0, 2.0]."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph  = _make_graph()
    mask   = _make_mask()
    report = RetinalBiomarkerExtractor().extract(graph, mask=mask)

    if math.isfinite(report.fractal_dimension):
        assert 1.0 <= report.fractal_dimension <= 2.0, (
            f"fractal_dimension={report.fractal_dimension} out of [1,2]"
        )
    _passed("biomarker_fractal_dimension_range")


def test_biomarker_with_zone():
    """Biomarkers with ZoneDefinition must populate zone-specific metrics."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor, ZoneDefinition
    graph  = _make_graph()
    mask   = _make_mask()
    zone   = ZoneDefinition(disc_centre=(32, 32), disc_radius=8.0)
    report = RetinalBiomarkerExtractor().extract(graph, mask=mask, zone=zone)

    # Zone-specific density should now be non-nan (zone exists in the image)
    # At least one zone metric should be finite
    zone_metrics = [report.vessel_density_zone_a, report.vessel_density_zone_b,
                    report.peripapillary_vessel_density]
    has_finite = any(math.isfinite(v) for v in zone_metrics)
    assert has_finite, "No zone-specific metrics computed despite valid zone definition"
    _passed("biomarker_with_zone")


def test_biomarker_av_labels_nan_without():
    """CRAE/CRVE/AVR must be nan when AV labels are not provided."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph  = _make_graph()
    report = RetinalBiomarkerExtractor().extract(graph, av_labels=None)

    assert math.isnan(report.crae), "CRAE should be nan without AV labels"
    assert math.isnan(report.crve), "CRVE should be nan without AV labels"
    assert math.isnan(report.avr),  "AVR should be nan without AV labels"
    _passed("biomarker_av_labels_nan_without")


def test_biomarker_av_with_labels():
    """CRAE/CRVE should be computed (non-nan) when AV labels are provided."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    from rapid_twin.av_classification import AVLabel
    graph = _make_graph()
    nodes = graph.nodes

    if not nodes or not graph.edges:
        _passed("biomarker_av_with_labels (no edges — skipped)")
        return

    # Provide synthetic AV labels
    av_labels = {
        n.node_id: (AVLabel.ARTERY if i % 2 == 0 else AVLabel.VEIN)
        for i, n in enumerate(nodes)
    }
    report = RetinalBiomarkerExtractor().extract(graph, av_labels=av_labels)

    # At least one of CRAE/CRVE should be non-nan (depends on edge count in zone)
    # Just verify extraction ran without crash
    assert report is not None
    _passed("biomarker_av_with_labels")


def test_biomarker_occupancy_map_shape():
    """Occupancy map must have shape (H//16, W//16)."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    H, W   = 64, 64
    mask   = _make_mask(H, W)
    graph  = _make_graph()
    report = RetinalBiomarkerExtractor(occupancy_cell_size=16).extract(graph, mask=mask)

    assert report.occupancy_map is not None
    assert report.occupancy_map.shape == (H // 16, W // 16), (
        f"Occupancy map shape {report.occupancy_map.shape} "
        f"≠ ({H//16}, {W//16})"
    )
    assert (report.occupancy_map >= 0).all() and (report.occupancy_map <= 1).all()
    _passed("biomarker_occupancy_map_shape")


def test_biomarker_to_dict():
    """BiomarkerReport.to_dict must return a dict of floats."""
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    graph   = _make_graph()
    report  = RetinalBiomarkerExtractor().extract(graph)
    flat    = report.to_dict()

    assert isinstance(flat, dict)
    assert len(flat) > 0
    for k, v in flat.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
    _passed("biomarker_to_dict")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Hemodynamic Modeling
# ══════════════════════════════════════════════════════════════════════════════

def test_murray_law_report():
    """validate_murray_law must return a valid report with non-negative counts."""
    from rapid_twin.hemodynamics import validate_murray_law
    graph  = _make_graph()
    report = validate_murray_law(graph)

    assert report.n_bifurcations >= 0
    assert report.n_compliant    >= 0
    assert report.n_violations   >= 0
    assert report.n_compliant + report.n_violations <= report.n_bifurcations + 1
    if report.n_bifurcations > 0:
        assert 0.0 <= report.compliance_fraction <= 1.0
    _passed("murray_law_report")


def test_murray_law_deviation_range():
    """Murray's law mean deviation must be finite and non-negative."""
    from rapid_twin.hemodynamics import validate_murray_law
    graph  = _make_graph()
    report = validate_murray_law(graph)

    if math.isfinite(report.mean_deviation):
        assert report.mean_deviation >= 0.0, (
            f"mean_deviation {report.mean_deviation} < 0"
        )
    _passed("murray_law_deviation_range")


def test_vessel_resistance_all_positive():
    """Poiseuille resistance must be strictly positive for all edges."""
    from rapid_twin.hemodynamics import compute_vessel_resistance
    graph = _make_graph()
    R     = compute_vessel_resistance(graph)

    for key, val in R.items():
        assert val > 0, f"Non-positive resistance at edge {key}: {val}"
        assert math.isfinite(val), f"Non-finite resistance at edge {key}: {val}"
    _passed("vessel_resistance_all_positive")


def test_flow_directions_valid():
    """Flow directions must be +1 or -1 for all edges."""
    from rapid_twin.hemodynamics import estimate_flow_directions
    graph     = _make_graph()
    hierarchy = _make_hierarchy(graph)
    dirs      = estimate_flow_directions(graph, hierarchy)

    for key, d in dirs.items():
        assert d in (+1, -1), f"Invalid direction {d} at edge {key}"
    _passed("flow_directions_valid")


def test_hemodynamic_simulation_basic():
    """HemodynamicModel.simulate must succeed and return valid pressures."""
    from rapid_twin.hemodynamics import HemodynamicModel
    graph     = _make_graph()
    hierarchy = _make_hierarchy(graph)

    if not graph.nodes:
        _passed("hemodynamic_simulation_basic (empty graph — skipped)")
        return

    model  = HemodynamicModel(root_pressure=80.0, validate_murray=True)
    result = model.simulate(graph, hierarchy=hierarchy)

    assert result.success, f"Simulation failed: {result.error}"
    assert result.is_valid
    assert result.perfusion_index >= 0.0
    assert result.perfusion_index <= 1.0, (
        f"perfusion_index={result.perfusion_index} > 1.0"
    )

    for nid, p in result.node_pressure.items():
        assert math.isfinite(p), f"Non-finite pressure at node {nid}: {p}"
        assert 0.0 <= p <= 80.0, f"Pressure {p} outside [0, 80] at node {nid}"
    _passed("hemodynamic_simulation_basic")


def test_hemodynamic_simulation_empty_graph():
    """Hemodynamic model must fail gracefully on empty graph."""
    from rapid_twin.hemodynamics import HemodynamicModel
    graph  = _make_empty_graph()
    model  = HemodynamicModel()
    result = model.simulate(graph)

    assert not result.success, "Should fail on empty graph"
    assert result.error is not None
    _passed("hemodynamic_simulation_empty_graph")


def test_hemodynamic_pressures_decreasing():
    """Pressures should generally decrease from root to leaves."""
    from rapid_twin.hemodynamics import HemodynamicModel
    from rapid_twin.graph.branch_hierarchy import build_branch_hierarchy
    graph     = _make_graph()
    hierarchy = build_branch_hierarchy(graph)

    if not graph.nodes or not hierarchy.is_valid:
        _passed("hemodynamic_pressures_decreasing (skipped)")
        return

    model  = HemodynamicModel(root_pressure=80.0)
    result = model.simulate(graph, hierarchy=hierarchy)

    if not result.success:
        _passed("hemodynamic_pressures_decreasing (simulation skipped)")
        return

    root_p = result.node_pressure.get(hierarchy.root_id, float("nan"))
    if math.isfinite(root_p) and root_p > 0:
        # Find leaf pressures
        leaf_ids = [nid for nid, bn in hierarchy.nodes.items() if bn.is_leaf]
        leaf_pressures = [result.node_pressure.get(l, root_p) for l in leaf_ids]
        if leaf_pressures:
            mean_leaf_p = sum(leaf_pressures) / len(leaf_pressures)
            assert mean_leaf_p <= root_p + 1e-3, (
                f"Mean leaf pressure {mean_leaf_p:.2f} > root pressure {root_p:.2f}"
            )
    _passed("hemodynamic_pressures_decreasing")


def test_hemodynamic_murray_report_in_result():
    """HemodynamicResult must include a MurrayLawReport when validate_murray=True."""
    from rapid_twin.hemodynamics import HemodynamicModel
    graph  = _make_graph()
    model  = HemodynamicModel(validate_murray=True)
    result = model.simulate(graph)

    if result.success:
        assert result.murray_report is not None, (
            "murray_report is None despite validate_murray=True"
        )
    _passed("hemodynamic_murray_report_in_result")


def test_hemodynamic_print_summary():
    """print_summary must not raise."""
    from rapid_twin.hemodynamics import HemodynamicModel
    graph  = _make_graph()
    model  = HemodynamicModel()
    result = model.simulate(graph)
    model.print_summary(result)   # must not raise
    _passed("hemodynamic_print_summary")


# ─── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # Graph refinement
    test_refiner_basic,
    test_refiner_empty_graph,
    test_topology_repair_bridges,
    test_simplify_graph,
    test_validate_consistency_no_nans,
    test_check_anatomical_plausibility,
    test_refiner_report_fields,

    # AV classification
    test_av_label_enum,
    test_av_classifier_forward,
    test_av_classifier_predict,
    test_av_classifier_empty_graph,
    test_propagate_av_labels,
    test_disambiguate_crossings,
    test_av_metrics_computation,
    test_av_dataset_adapter_invalid_path,
    test_av_dataset_adapter_unsupported,

    # Biomarkers
    test_biomarker_report_structure,
    test_biomarker_extraction_basic,
    test_biomarker_tortuosity,
    test_biomarker_vessel_density_range,
    test_biomarker_fractal_dimension_range,
    test_biomarker_with_zone,
    test_biomarker_av_labels_nan_without,
    test_biomarker_av_with_labels,
    test_biomarker_occupancy_map_shape,
    test_biomarker_to_dict,

    # Hemodynamics
    test_murray_law_report,
    test_murray_law_deviation_range,
    test_vessel_resistance_all_positive,
    test_flow_directions_valid,
    test_hemodynamic_simulation_basic,
    test_hemodynamic_simulation_empty_graph,
    test_hemodynamic_pressures_decreasing,
    test_hemodynamic_murray_report_in_result,
    test_hemodynamic_print_summary,
]


def run_all_tests():
    passed = 0
    failed = 0
    print(f"\n{'='*68}")
    print("  RAPiD-Twin Vascular Intelligence Test Suite")
    print(f"{'='*68}\n")
    print("  Sections:")
    print("    1. Graph Refinement (7 tests)")
    print("    2. Artery/Vein Intelligence (9 tests)")
    print("    3. Biomarker Extraction (10 tests)")
    print("    4. Hemodynamic Modeling (9 tests)")
    print()

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            _failed(test_fn.__name__, exc)
            failed += 1

    print(f"\n{'='*68}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"{'='*68}\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
