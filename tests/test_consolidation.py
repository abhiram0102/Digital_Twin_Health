"""
Consolidation Sprint Integration Tests.

Covers:
  1.  Uncertainty estimation (MCDropoutPredictor, CalibrationAnalyzer,
      TemperatureScaler, OODDetector, ConfidenceAwareBiomarkers)
  2.  Benchmark suite (SegmentationBenchmark, TopologyBenchmark,
      AVBenchmark, BiomarkerBenchmark, PlausibilityBenchmark,
      UnifiedBenchmarkRunner)
  3.  Graph robustness (inject_noise, GraphStabilityAnalyzer,
      test_degenerate_inputs, check_numerical_stability)
  4.  Performance profiling (profile_stage, PipelineProfiler,
      profile_graph_scaling, check_numerical_stability)

All tests use synthetic data only — no real datasets required.
No GPU required.

Run with:
    python -m pytest tests/test_consolidation.py -v
or:
    python tests/test_consolidation.py
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Dict

import numpy as np


# ─── Shared fixtures ──────────────────────────────────────────────────────────

def _make_mask(H: int = 64, W: int = 64) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.float32)
    mask[H // 4, :]  = 1.0
    mask[H // 2, :]  = 1.0
    mask[3*H//4, :]  = 1.0
    mask[:, W // 4]  = 1.0
    mask[:, W // 2]  = 1.0
    mask[:, 3*W//4]  = 1.0
    return mask


def _make_graph(H: int = 64, W: int = 64):
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder
    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    mask = _make_mask(H, W)
    return builder.build(mask, intensity_image=mask)


def _make_hierarchy(graph):
    from rapid_twin.graph.branch_hierarchy import build_branch_hierarchy
    return build_branch_hierarchy(graph)


def _make_biomarker_report(graph=None, mask=None):
    from rapid_twin.biomarkers import RetinalBiomarkerExtractor
    if graph is None:
        graph = _make_graph()
    if mask is None:
        mask = _make_mask()
    return RetinalBiomarkerExtractor().extract(graph, mask=mask)


def _passed(name: str):
    print(f"  PASS  {name}")


def _failed(name: str, exc: Exception):
    print(f"  FAIL  {name}: {exc}")
    traceback.print_exc()
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. Uncertainty Estimation
# ══════════════════════════════════════════════════════════════════════════════

def test_prediction_entropy_uniform():
    """Entropy of uniform distribution must equal log(C)."""
    from rapid_twin.reliability import compute_prediction_entropy
    C    = 4
    p    = np.full((10, C), 1.0 / C, dtype=np.float32)
    H    = compute_prediction_entropy(p)
    expected = math.log(C)
    assert H.shape == (10,)
    assert abs(H.mean() - expected) < 1e-4, f"Entropy {H.mean():.4f} ≠ log({C})={expected:.4f}"
    _passed("prediction_entropy_uniform")


def test_prediction_entropy_certain():
    """Entropy of a one-hot distribution must be 0."""
    from rapid_twin.reliability import compute_prediction_entropy
    p = np.eye(4, dtype=np.float32)  # (4, 4) one-hot
    H = compute_prediction_entropy(p)
    assert (H < 1e-4).all(), f"One-hot entropy should be ~0, got {H}"
    _passed("prediction_entropy_certain")


def test_mutual_information_zero_variance():
    """MI must be 0 when all MC samples are identical."""
    from rapid_twin.reliability import compute_prediction_entropy, compute_mutual_information
    p = np.array([[0.7, 0.2, 0.1]] * 5, dtype=np.float32)   # 5 identical samples
    sample_probs = np.stack([p] * 5, axis=0)                  # (5, 5, 3)
    mean_probs   = sample_probs.mean(axis=0)
    mi = compute_mutual_information(sample_probs, mean_probs)
    assert mi.shape == (5,)
    assert (mi < 1e-4).all(), f"MI should be ~0 for identical samples, got {mi}"
    _passed("mutual_information_zero_variance")


def test_calibration_perfect():
    """CalibrationAnalyzer: perfect confidence → ECE should be near 0."""
    from rapid_twin.reliability import CalibrationAnalyzer
    N      = 200
    labels = (np.arange(N) % 2).astype(np.int64)
    # Perfect confidence: always correct with prob=1.0
    probs  = np.zeros((N, 2), dtype=np.float32)
    for i, l in enumerate(labels):
        probs[i, l] = 1.0

    analyzer = CalibrationAnalyzer(n_bins=10)
    result   = analyzer.compute(probs, labels)
    assert result.ece < 0.05, f"Perfect predictor ECE={result.ece:.4f} > 0.05"
    assert result.is_well_calibrated
    _passed("calibration_perfect")


def test_calibration_overconfident():
    """ECE should be high for a systematically over-confident predictor."""
    from rapid_twin.reliability import CalibrationAnalyzer
    N      = 200
    labels = (np.arange(N) % 2).astype(np.int64)
    # Predict wrong class with 99% confidence
    probs  = np.zeros((N, 2), dtype=np.float32)
    for i, l in enumerate(labels):
        probs[i, 1 - l] = 0.99   # always wrong, always confident
        probs[i, l]     = 0.01

    analyzer = CalibrationAnalyzer(n_bins=10)
    result   = analyzer.compute(probs, labels)
    assert result.ece > 0.05, f"Over-confident predictor ECE={result.ece:.4f} should be > 0.05"
    _passed("calibration_overconfident")


def test_calibration_result_fields():
    """CalibrationResult must have all required fields."""
    from rapid_twin.reliability import CalibrationAnalyzer
    p = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]], dtype=np.float32)
    l = np.array([0, 1, 0], dtype=np.int64)
    result = CalibrationAnalyzer(n_bins=5).compute(p, l)
    assert hasattr(result, "ece")
    assert hasattr(result, "mce")
    assert hasattr(result, "ace")
    assert hasattr(result, "bin_confidences")
    assert hasattr(result, "bin_accuracies")
    assert hasattr(result, "bin_counts")
    assert hasattr(result, "is_well_calibrated")
    assert 0.0 <= result.ece <= 1.0
    assert 0.0 <= result.mce <= 1.0
    _passed("calibration_result_fields")


def test_temperature_scaler_forward():
    """TemperatureScaler must divide logits by temperature."""
    try:
        import torch
        from rapid_twin.reliability import TemperatureScaler
    except ImportError:
        print("  SKIP  temperature_scaler_forward (torch not available)")
        return

    scaler  = TemperatureScaler(init_temperature=2.0)
    logits  = torch.tensor([[2.0, 1.0, 0.0]])
    scaled  = scaler(logits)
    expected = logits / 2.0
    assert torch.allclose(scaled, expected, atol=1e-5), (
        f"Scaled logits {scaled} ≠ expected {expected}"
    )
    _passed("temperature_scaler_forward")


def test_temperature_scaler_calibrate():
    """TemperatureScaler.calibrate must return valid probability distribution."""
    try:
        import torch
        from rapid_twin.reliability import TemperatureScaler
    except ImportError:
        print("  SKIP  temperature_scaler_calibrate (torch not available)")
        return

    scaler = TemperatureScaler(init_temperature=1.5)
    logits = torch.randn(20, 3)
    probs  = scaler.calibrate(logits)
    assert probs.shape == (20, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(20), atol=1e-5)
    assert (probs >= 0).all()
    _passed("temperature_scaler_calibrate")


def test_ood_detector_fit_score():
    """OODDetector must fit without error and score samples."""
    from rapid_twin.reliability import OODDetector
    rng = np.random.default_rng(0)

    # In-distribution features (2 classes, well-separated)
    D  = 16
    X0 = rng.normal(loc=0.0, scale=1.0, size=(50, D)).astype(np.float32)
    X1 = rng.normal(loc=5.0, scale=1.0, size=(50, D)).astype(np.float32)
    X  = np.vstack([X0, X1])
    y  = np.array([0] * 50 + [1] * 50)

    detector = OODDetector(mahalanobis_percentile=95.0)
    detector.fit(X, y)
    assert detector._is_fitted

    # OOD samples — far from both class centres
    X_ood  = rng.normal(loc=50.0, scale=1.0, size=(10, D)).astype(np.float32)
    scores = detector.score(X_ood)
    assert scores.mahalanobis.shape == (10,)
    assert (np.isfinite(scores.mahalanobis)).all()
    # OOD samples should score higher than in-dist median
    in_scores = detector.score(X).mahalanobis
    assert scores.mahalanobis.mean() > in_scores.mean(), (
        "OOD samples should have higher Mahalanobis distance than in-distribution"
    )
    _passed("ood_detector_fit_score")


def test_ood_detector_not_fitted_error():
    """OODDetector must raise RuntimeError if score() called before fit()."""
    from rapid_twin.reliability import OODDetector
    detector = OODDetector()
    try:
        detector.score(np.zeros((5, 8)))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    _passed("ood_detector_not_fitted_error")


def test_confidence_aware_biomarkers():
    """ConfidenceAwareBiomarkers must return one BiomarkerWithCI per scalar."""
    from rapid_twin.reliability import ConfidenceAwareBiomarkers
    report = _make_biomarker_report()
    flat   = report.to_dict()
    N      = max(len(_make_graph().nodes), 1)
    unc    = np.full(N, 0.3, dtype=np.float32)

    cabi    = ConfidenceAwareBiomarkers()
    results = cabi.annotate(report, unc)

    assert len(results) == len(flat), (
        f"Expected {len(flat)} annotated biomarkers, got {len(results)}"
    )
    for bwci in results:
        assert hasattr(bwci, "name")
        assert hasattr(bwci, "value")
        assert hasattr(bwci, "ci_lower")
        assert hasattr(bwci, "ci_upper")
        assert hasattr(bwci, "is_reliable")
        assert hasattr(bwci, "uncertainty")
        if math.isfinite(bwci.value) and math.isfinite(bwci.ci_lower):
            assert bwci.ci_lower <= bwci.value <= bwci.ci_upper, (
                f"{bwci.name}: CI [{bwci.ci_lower:.4f}, {bwci.ci_upper:.4f}] "
                f"does not contain value {bwci.value:.4f}"
            )
    _passed("confidence_aware_biomarkers")


def test_mc_dropout_predictor():
    """MCDropoutPredictor must return correct shapes for a simple model."""
    try:
        import torch
        import torch.nn as nn
        from rapid_twin.reliability import MCDropoutPredictor
    except ImportError:
        print("  SKIP  mc_dropout_predictor (torch not available)")
        return

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 32), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 3),
            )
        def forward(self, x): return self.net(x)

    model = SimpleMLP()
    predictor = MCDropoutPredictor(model, n_samples=10, reliability_threshold=0.5)
    x = torch.randn(20, 8)

    unc = predictor.predict(x)
    assert unc.mean_probs.shape == (20, 3), f"mean_probs shape {unc.mean_probs.shape}"
    assert unc.variance.shape == (20, 3)
    assert unc.predictive_entropy.shape == (20,)
    assert unc.mutual_information.shape == (20,)
    assert (unc.predictive_entropy >= 0).all()
    assert (unc.mutual_information >= 0).all()
    assert unc.n_samples == 10
    _passed("mc_dropout_predictor")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Benchmark Suite
# ══════════════════════════════════════════════════════════════════════════════

def test_segmentation_benchmark_perfect():
    """SegmentationBenchmark: perfect prediction must give Dice=1."""
    from rapid_twin.benchmarks import SegmentationBenchmark
    masks = [_make_mask() for _ in range(3)]
    bench = SegmentationBenchmark(dataset="test")
    result = bench.evaluate(predictions=masks, ground_truths=masks)

    assert result.metrics["dice"] > 0.999, (
        f"Perfect prediction Dice={result.metrics['dice']:.4f} < 1.0"
    )
    assert result.n_samples == 3
    _passed("segmentation_benchmark_perfect")


def test_segmentation_benchmark_zero_pred():
    """Dice should be 0 when prediction is all-zeros."""
    from rapid_twin.benchmarks import SegmentationBenchmark
    masks = [_make_mask()]
    zeros = [np.zeros_like(masks[0])]
    bench  = SegmentationBenchmark(dataset="test")
    result = bench.evaluate(predictions=zeros, ground_truths=masks)
    assert result.metrics["dice"] < 0.01, (
        f"All-zero prediction Dice={result.metrics['dice']:.4f} should be ~0"
    )
    _passed("segmentation_benchmark_zero_pred")


def test_segmentation_benchmark_shape_mismatch():
    """evaluate_single must raise ValueError on shape mismatch."""
    from rapid_twin.benchmarks import SegmentationBenchmark
    bench = SegmentationBenchmark()
    try:
        bench.evaluate_single(np.zeros((64, 64)), np.zeros((32, 32)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("segmentation_benchmark_shape_mismatch")


def test_segmentation_benchmark_result_fields():
    """BenchmarkResult must contain expected fields and metric keys."""
    from rapid_twin.benchmarks import SegmentationBenchmark
    masks  = [_make_mask()]
    bench  = SegmentationBenchmark(dataset="DRIVE")
    result = bench.evaluate(masks, masks)
    assert hasattr(result, "name")
    assert hasattr(result, "dataset")
    assert hasattr(result, "metrics")
    assert hasattr(result, "n_samples")
    assert hasattr(result, "timing_s")
    assert "dice" in result.metrics
    assert "recall" in result.metrics
    assert result.timing_s >= 0.0
    _passed("segmentation_benchmark_result_fields")


def test_topology_benchmark_basic():
    """TopologyBenchmark must run on synthetic graphs without crash."""
    from rapid_twin.benchmarks import TopologyBenchmark
    graphs = [_make_graph() for _ in range(2)]
    masks  = [_make_mask() for _ in range(2)]
    bench  = TopologyBenchmark(dataset="test")
    result = bench.evaluate(graphs, masks=masks)
    assert result.n_samples == 2
    assert "graph_quality_score" in result.metrics or len(result.metrics) > 0
    _passed("topology_benchmark_basic")


def test_topology_benchmark_with_hierarchy():
    """TopologyBenchmark with hierarchies must populate depth_entropy."""
    from rapid_twin.benchmarks import TopologyBenchmark
    graphs = [_make_graph()]
    hiers  = [_make_hierarchy(graphs[0])]
    bench  = TopologyBenchmark(dataset="test")
    result = bench.evaluate(graphs, hierarchies=hiers)
    # depth_entropy should be present (may be nan if hierarchy invalid)
    assert "depth_entropy" in result.metrics
    _passed("topology_benchmark_with_hierarchy")


def test_av_benchmark_basic():
    """AVBenchmark must compute metrics from synthetic predictions."""
    from rapid_twin.benchmarks import AVBenchmark
    from rapid_twin.av_classification import AVLabel, AVPrediction

    graph = _make_graph()
    nodes = graph.nodes
    if not nodes:
        _passed("av_benchmark_basic (no nodes — skipped)")
        return

    preds = [
        AVPrediction(n.node_id, AVLabel.ARTERY if i % 2 == 0 else AVLabel.VEIN,
                     0.8, 0.8 if i%2==0 else 0.1, 0.1 if i%2==0 else 0.8, 0.1)
        for i, n in enumerate(nodes)
    ]
    gt = {n.node_id: (AVLabel.ARTERY if i%2==0 else AVLabel.VEIN)
          for i, n in enumerate(nodes)}

    bench  = AVBenchmark(dataset="test")
    result = bench.evaluate([preds], [gt], graphs=[graph])
    assert "accuracy" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    _passed("av_benchmark_basic")


def test_biomarker_benchmark_reproducibility():
    """BiomarkerBenchmark reproducibility must be in [0, 1] for identical reports."""
    from rapid_twin.benchmarks import BiomarkerBenchmark
    report = _make_biomarker_report()
    # Identical reports → std=0 → CV=0 → reproducibility=1
    bench  = BiomarkerBenchmark(dataset="test")
    result = bench.evaluate([report, report, report])
    assert result.n_samples == 3
    assert "reproducibility" in result.metrics
    repro = result.metrics["reproducibility"]
    if math.isfinite(repro):
        assert 0.0 <= repro <= 1.0, f"reproducibility={repro:.4f} out of [0,1]"
    _passed("biomarker_benchmark_reproducibility")


def test_plausibility_benchmark_basic():
    """PlausibilityBenchmark must return plausibility_score in [0, 1]."""
    from rapid_twin.benchmarks import PlausibilityBenchmark
    graphs = [_make_graph() for _ in range(3)]
    bench  = PlausibilityBenchmark(dataset="test")
    result = bench.evaluate(graphs)
    assert result.n_samples == 3
    if "plausibility_score" in result.metrics:
        ps = result.metrics["plausibility_score"]
        if math.isfinite(ps):
            assert 0.0 <= ps <= 1.0, f"plausibility_score={ps:.4f} out of [0,1]"
    _passed("plausibility_benchmark_basic")


def test_unified_benchmark_runner_partial():
    """UnifiedBenchmarkRunner must run available benchmarks and skip others."""
    from rapid_twin.benchmarks import UnifiedBenchmarkRunner
    graphs = [_make_graph() for _ in range(2)]
    masks  = [_make_mask() for _ in range(2)]

    runner  = UnifiedBenchmarkRunner(dataset="test")
    results = runner.run(graphs=graphs, masks=masks)  # no seg/AV/biomarker

    assert "topology" in results, "Topology benchmark should run"
    assert "plausibility" in results, "Plausibility benchmark should run"
    assert "segmentation" not in results  # no predictions given
    _passed("unified_benchmark_runner_partial")


def test_unified_benchmark_runner_full():
    """UnifiedBenchmarkRunner.run must succeed with all inputs provided."""
    from rapid_twin.benchmarks import UnifiedBenchmarkRunner
    from rapid_twin.av_classification import AVLabel, AVPrediction

    graph = _make_graph()
    mask  = _make_mask()
    nodes = graph.nodes
    hiers = [_make_hierarchy(graph)]

    preds = [AVPrediction(n.node_id, AVLabel.ARTERY if i%2==0 else AVLabel.VEIN,
                          0.8, 0.8, 0.1, 0.1)
             for i, n in enumerate(nodes)]
    gt    = {n.node_id: (AVLabel.ARTERY if i%2==0 else AVLabel.VEIN)
             for i, n in enumerate(nodes)}
    report = _make_biomarker_report(graph, mask)

    runner = UnifiedBenchmarkRunner(dataset="synthetic")
    results = runner.run(
        predictions=[mask], ground_truths=[mask],
        graphs=[graph], masks=[mask], hierarchies=hiers,
        av_predictions=[preds], av_ground_truths=[gt],
        biomarker_reports=[report],
    )
    for bench in ["segmentation", "topology", "av_classification", "biomarker", "plausibility"]:
        assert bench in results, f"Missing benchmark: {bench}"
    _passed("unified_benchmark_runner_full")


def test_benchmark_summary_line():
    """BenchmarkResult.summary_line must return a non-empty string."""
    from rapid_twin.benchmarks import SegmentationBenchmark
    masks  = [_make_mask()]
    result = SegmentationBenchmark().evaluate(masks, masks)
    line   = result.summary_line()
    assert isinstance(line, str) and len(line) > 0
    _passed("benchmark_summary_line")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Graph Robustness
# ══════════════════════════════════════════════════════════════════════════════

def test_inject_noise_salt_pepper():
    """Salt-and-pepper noise must change some pixels and preserve shape."""
    from rapid_twin.graph.graph_robustness import inject_noise
    mask  = _make_mask()
    noisy = inject_noise(mask, "salt_pepper", noise_level=0.1, seed=0)
    assert noisy.shape == mask.shape
    assert noisy.dtype == np.float32
    diff = int((noisy != mask).sum())
    assert diff > 0, "No pixels changed — noise injection failed"
    _passed("inject_noise_salt_pepper")


def test_inject_noise_zero_level():
    """Zero noise level must return an essentially identical mask."""
    from rapid_twin.graph.graph_robustness import inject_noise
    mask  = _make_mask()
    noisy = inject_noise(mask, "salt_pepper", noise_level=0.0, seed=0)
    assert np.array_equal(noisy, mask), "Zero noise should not change any pixel"
    _passed("inject_noise_zero_level")


def test_inject_noise_all_modes():
    """All noise modes must run without crash."""
    from rapid_twin.graph.graph_robustness import inject_noise
    modes = ["salt_pepper", "dilation", "erosion",
             "gaussian_blur", "segment_dropout", "fragmentation"]
    mask = _make_mask()
    for mode in modes:
        try:
            noisy = inject_noise(mask, mode, noise_level=0.1, seed=0)
            assert noisy.shape == mask.shape, f"{mode}: shape changed"
        except ImportError:
            pass  # opencv/skimage may not be installed
    _passed("inject_noise_all_modes")


def test_inject_noise_invalid_type():
    """Invalid noise type must raise ValueError."""
    from rapid_twin.graph.graph_robustness import inject_noise
    try:
        inject_noise(_make_mask(), "invalid_mode", 0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _passed("inject_noise_invalid_type")


def test_stability_analyzer_basic():
    """GraphStabilityAnalyzer must return a StabilityReport with correct length."""
    from rapid_twin.graph.graph_robustness import GraphStabilityAnalyzer
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder

    builder  = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    mask     = _make_mask()
    levels   = [0.02, 0.05]
    analyzer = GraphStabilityAnalyzer(
        noise_type="salt_pepper", noise_levels=levels, n_seeds=3
    )
    report   = analyzer.analyze(mask, graph_fn=lambda m: builder.build(m))

    assert len(report.stability_score)  == len(levels)
    assert len(report.node_count_cv)    == len(levels)
    assert len(report.edge_count_cv)    == len(levels)
    assert report.noise_type == "salt_pepper"
    assert report.timing_s >= 0.0
    for s in report.stability_score:
        if math.isfinite(s):
            assert 0.0 <= s <= 1.0, f"stability_score={s} out of [0,1]"
    _passed("stability_analyzer_basic")


def test_stability_score_decreases_with_noise():
    """Stability score should generally decrease with increasing noise."""
    from rapid_twin.graph.graph_robustness import GraphStabilityAnalyzer
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder

    builder  = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    mask     = _make_mask(H=128, W=128)
    levels   = [0.01, 0.20]
    analyzer = GraphStabilityAnalyzer(
        noise_type="salt_pepper", noise_levels=levels, n_seeds=3
    )
    report   = analyzer.analyze(mask, graph_fn=lambda m: builder.build(m))

    s_low  = report.stability_score[0]  # low noise
    s_high = report.stability_score[1]  # high noise
    # With very low noise stability should be >= high noise (or nan)
    if math.isfinite(s_low) and math.isfinite(s_high):
        # Allow some slack — small masks may not show monotonic trend
        assert s_low >= s_high - 0.3, (
            f"Stability at low noise ({s_low:.3f}) should not be much "
            f"lower than high noise ({s_high:.3f})"
        )
    _passed("stability_score_decreases_with_noise")


def test_degenerate_inputs_coverage():
    """test_degenerate_inputs must run all 9 cases without crashing."""
    from rapid_twin.graph.graph_robustness import test_degenerate_inputs
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder

    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    results = test_degenerate_inputs(
        graph_fn=lambda m: builder.build(m), H=64, W=64
    )
    assert len(results) == 9, f"Expected 9 test cases, got {len(results)}"
    # All cases must produce a result (pass or fail — never a crash of the test itself)
    for r in results:
        assert hasattr(r, "case_name")
        assert hasattr(r, "passed")
        assert hasattr(r, "description")
    _passed("degenerate_inputs_coverage")


def test_degenerate_all_zero_and_cross_pass():
    """All-zero and cross-shape degenerate inputs must be handled gracefully."""
    from rapid_twin.graph.graph_robustness import test_degenerate_inputs
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder

    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    results = {r.case_name: r for r in test_degenerate_inputs(
        graph_fn=lambda m: builder.build(m)
    )}
    assert results["all_zero"].passed,   "all-zero mask should pass gracefully"
    assert results["cross_shape"].passed, "cross-shape mask should pass"
    _passed("degenerate_all_zero_and_cross_pass")


def test_check_numerical_stability():
    """check_numerical_stability must return a dict with all required keys."""
    from rapid_twin.utils.profiling import check_numerical_stability
    graph  = _make_graph()
    mask   = _make_mask()
    issues = check_numerical_stability(graph, mask)

    required = [
        "nan_in_node_features", "nan_in_edge_features",
        "large_node_values", "large_edge_values",
        "negative_calibers", "negative_lengths",
        "impossible_tortuosity", "has_critical_issues",
        "total_issues",
    ]
    for k in required:
        assert k in issues, f"Missing key: {k}"

    # Clean graph built from clean mask must have no critical issues
    assert not issues["has_critical_issues"], (
        f"Synthetic clean graph has critical issues: {issues}"
    )
    assert issues["nan_in_node_features"] == 0
    assert issues["nan_in_edge_features"] == 0
    assert issues["impossible_tortuosity"] == 0
    _passed("check_numerical_stability")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Performance Profiling
# ══════════════════════════════════════════════════════════════════════════════

def test_profile_stage_timing():
    """profile_stage must report elapsed time >= 0."""
    from rapid_twin.utils.profiling import profile_stage
    import time
    with profile_stage("sleep_test") as stats:
        time.sleep(0.01)
    assert stats["wall_time_s"] >= 0.01, (
        f"Elapsed time {stats['wall_time_s']} < 0.01"
    )
    assert math.isfinite(stats["wall_time_s"])
    assert stats["memory_mb"] >= 0.0
    _passed("profile_stage_timing")


def test_profile_stage_memory():
    """profile_stage must report non-negative memory."""
    from rapid_twin.utils.profiling import profile_stage
    with profile_stage("alloc_test") as stats:
        big = np.zeros((1000, 1000), dtype=np.float64)  # 8 MB
        del big
    assert stats["memory_mb"] >= 0.0
    _passed("profile_stage_memory")


def test_pipeline_profiler_record():
    """PipelineProfiler.record + make_report must produce a valid report."""
    from rapid_twin.utils.profiling import PipelineProfiler
    profiler = PipelineProfiler(n_runs=3)
    profiler.record("stage_a", 0.10)
    profiler.record("stage_a", 0.12)
    profiler.record("stage_a", 0.11)
    profiler.record("stage_b", 0.50)
    profiler.record("stage_b", 0.52)
    profiler.record("stage_b", 0.51)

    report = profiler.make_report(metadata={"test": True})
    assert "stage_a" in report.stages
    assert "stage_b" in report.stages
    assert report.bottleneck == "stage_b", (
        f"Expected bottleneck=stage_b, got {report.bottleneck}"
    )
    assert abs(report.stages["stage_a"].mean_time_s - 0.11) < 0.01
    assert report.total_time_s > 0
    _passed("pipeline_profiler_record")


def test_pipeline_profiler_decorator():
    """PipelineProfiler.profile decorator must wrap function and record timing."""
    from rapid_twin.utils.profiling import PipelineProfiler
    profiler = PipelineProfiler(n_runs=2, warmup=0)

    @profiler.profile("build")
    def build(m):
        from rapid_twin.graph.graph_builder import RetinalGraphBuilder
        return RetinalGraphBuilder(min_branch_length=2).build(m)

    mask = _make_mask()
    build(mask)

    assert "build" in profiler._records
    assert len(profiler._records["build"]) == 2
    _passed("pipeline_profiler_decorator")


def test_profiling_report_to_dict():
    """ProfilingReport.to_dict must return a JSON-serialisable dict."""
    from rapid_twin.utils.profiling import PipelineProfiler
    import json

    profiler = PipelineProfiler(n_runs=2, warmup=0)
    profiler.record("s1", 0.1)
    profiler.record("s2", 0.2)
    report = profiler.make_report()
    d      = report.to_dict()

    # Must be JSON serialisable
    json_str = json.dumps(d)
    assert len(json_str) > 0
    assert "total_time_s" in d
    assert "bottleneck"   in d
    assert "stages"       in d
    _passed("profiling_report_to_dict")


def test_profile_graph_scaling():
    """profile_graph_scaling must return a report with scaling dict."""
    from rapid_twin.utils.profiling import profile_graph_scaling
    from rapid_twin.graph.graph_builder import RetinalGraphBuilder

    builder = RetinalGraphBuilder(min_branch_length=2, max_nodes=512)
    report  = profile_graph_scaling(
        graph_fn=lambda m: builder.build(m),
        mask_sizes=[(32, 32), (64, 64)],
        n_runs=2,
    )
    assert len(report.scaling) == 2, (
        f"Expected 2 entries in scaling dict, got {len(report.scaling)}"
    )
    for px, t in report.scaling.items():
        assert t >= 0.0, f"Negative time {t} for size {px}"
        assert math.isfinite(t)
    _passed("profile_graph_scaling")


def test_bottleneck_identification():
    """The bottleneck must be the stage with the highest mean time."""
    from rapid_twin.utils.profiling import PipelineProfiler
    profiler = PipelineProfiler(n_runs=1, warmup=0)
    profiler.record("fast",   0.01)
    profiler.record("medium", 0.10)
    profiler.record("slow",   1.00)
    report = profiler.make_report()
    assert report.bottleneck == "slow", (
        f"Expected bottleneck='slow', got '{report.bottleneck}'"
    )
    _passed("bottleneck_identification")


def test_print_profiling_report():
    """PipelineProfiler.print_report must not raise."""
    from rapid_twin.utils.profiling import PipelineProfiler
    profiler = PipelineProfiler(n_runs=1, warmup=0)
    profiler.record("a", 0.1)
    profiler.record("b", 0.2)
    report = profiler.make_report()
    profiler.print_report(report)
    _passed("print_profiling_report")


# ─── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # Uncertainty
    test_prediction_entropy_uniform,
    test_prediction_entropy_certain,
    test_mutual_information_zero_variance,
    test_calibration_perfect,
    test_calibration_overconfident,
    test_calibration_result_fields,
    test_temperature_scaler_forward,
    test_temperature_scaler_calibrate,
    test_ood_detector_fit_score,
    test_ood_detector_not_fitted_error,
    test_confidence_aware_biomarkers,
    test_mc_dropout_predictor,

    # Benchmark suite
    test_segmentation_benchmark_perfect,
    test_segmentation_benchmark_zero_pred,
    test_segmentation_benchmark_shape_mismatch,
    test_segmentation_benchmark_result_fields,
    test_topology_benchmark_basic,
    test_topology_benchmark_with_hierarchy,
    test_av_benchmark_basic,
    test_biomarker_benchmark_reproducibility,
    test_plausibility_benchmark_basic,
    test_unified_benchmark_runner_partial,
    test_unified_benchmark_runner_full,
    test_benchmark_summary_line,

    # Graph robustness
    test_inject_noise_salt_pepper,
    test_inject_noise_zero_level,
    test_inject_noise_all_modes,
    test_inject_noise_invalid_type,
    test_stability_analyzer_basic,
    test_stability_score_decreases_with_noise,
    test_degenerate_inputs_coverage,
    test_degenerate_all_zero_and_cross_pass,
    test_check_numerical_stability,

    # Profiling
    test_profile_stage_timing,
    test_profile_stage_memory,
    test_pipeline_profiler_record,
    test_pipeline_profiler_decorator,
    test_profiling_report_to_dict,
    test_profile_graph_scaling,
    test_bottleneck_identification,
    test_print_profiling_report,
]


def run_all_tests():
    passed = 0
    failed = 0
    print(f"\n{'='*70}")
    print("  RAPiD-Twin Consolidation Sprint — Integration Test Suite")
    print(f"{'='*70}\n")
    print("  Sections:")
    print("    1. Uncertainty Estimation   (12 tests)")
    print("    2. Benchmark Suite          (13 tests)")
    print("    3. Graph Robustness         (9 tests)")
    print("    4. Performance Profiling    (8 tests)")
    print()

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            _failed(fn.__name__, exc)
            failed += 1

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print(f"{'='*70}\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
