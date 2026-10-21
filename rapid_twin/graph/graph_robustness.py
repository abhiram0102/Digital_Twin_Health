"""
Graph Robustness Analysis — Stress-Testing Under Noisy Segmentations.

Real clinical segmentation models produce imperfect masks.  The graph pipeline
must degrade gracefully under:
  - Salt-and-pepper noise  (random vessel pixel toggling)
  - Dilation / erosion noise (vessel width perturbation)
  - Gaussian blur then threshold (probability map smoothing artefacts)
  - Missing vessel segments (random connected-component dropout)
  - Vessel fragmentation (break long segments into short stubs)

Provides:
  - inject_noise()            — six noise modes on binary masks
  - GraphStabilityAnalyzer    — measures graph metric stability across noise levels
  - StabilityReport           — structured results per noise level
  - test_degenerate_inputs()  — systematic edge-case coverage

Design contracts:
  • All noise injection is reproducible given a random seed.
  • Stability scores are in [0, 1]; 1 = perfectly stable.
  • Degenerate input test returns explicit True/False per case with reasons.
  • Never crashes on any input — returns explicit failure records instead.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─── Noise injection ──────────────────────────────────────────────────────────

def inject_noise(
    mask:       np.ndarray,
    noise_type: str  = "salt_pepper",
    noise_level: float = 0.05,
    seed:       int  = 42,
) -> np.ndarray:
    """
    Inject synthetic noise into a binary vessel mask.

    Args:
        mask:        (H, W) binary float32 mask (vessels = 1.0).
        noise_type:  one of:
                     "salt_pepper"    — randomly flip pixels
                     "dilation"       — morphologically dilate vessels
                     "erosion"        — morphologically erode vessels
                     "gaussian_blur"  — blur then re-threshold
                     "segment_dropout" — remove random connected components
                     "fragmentation"  — break long vessels into short stubs
        noise_level: intensity of noise in [0, 1].  Semantics vary by type.
        seed:        random seed for reproducibility.

    Returns:
        (H, W) float32 binary mask with noise applied.
    """
    VALID = {"salt_pepper", "dilation", "erosion", "gaussian_blur",
             "segment_dropout", "fragmentation"}
    if noise_type not in VALID:
        raise ValueError(f"noise_type must be one of {VALID}, got '{noise_type}'")
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")

    rng    = np.random.default_rng(seed)
    binary = (mask > 0.5).astype(np.float32)

    if noise_type == "salt_pepper":
        flip   = rng.random(binary.shape) < noise_level
        noisy  = np.where(flip, 1.0 - binary, binary)
        return noisy.astype(np.float32)

    if noise_type == "dilation":
        try:
            import cv2
            radius = max(1, int(noise_level * 5))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
            return cv2.dilate(binary.astype(np.uint8), kernel).astype(np.float32)
        except ImportError:
            warnings.warn("opencv required for 'dilation' noise. Returning original.")
            return binary

    if noise_type == "erosion":
        try:
            import cv2
            radius = max(1, int(noise_level * 5))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
            return cv2.erode(binary.astype(np.uint8), kernel).astype(np.float32)
        except ImportError:
            warnings.warn("opencv required for 'erosion' noise. Returning original.")
            return binary

    if noise_type == "gaussian_blur":
        try:
            import cv2
            sigma   = noise_level * 5.0
            blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
            thresh  = 0.5 - (noise_level * 0.2)  # lower threshold = more vessel
            return (blurred > thresh).astype(np.float32)
        except ImportError:
            warnings.warn("opencv required for 'gaussian_blur' noise. Returning original.")
            return binary

    if noise_type == "segment_dropout":
        try:
            from skimage.measure import label as sklabel
            labeled, n = sklabel(binary.astype(np.uint8), return_num=True)
            to_drop = set(rng.choice(n, max(1, int(n * noise_level)), replace=False) + 1)
            noisy = binary.copy()
            for comp_id in to_drop:
                noisy[labeled == comp_id] = 0.0
            return noisy
        except ImportError:
            warnings.warn("scikit-image required for 'segment_dropout' noise. Returning original.")
            return binary

    if noise_type == "fragmentation":
        # Random horizontal/vertical cuts through vessel pixels
        noisy = binary.copy()
        H, W  = binary.shape
        n_cuts = max(1, int(noise_level * min(H, W) // 4))
        for _ in range(n_cuts):
            if rng.random() < 0.5:
                row = int(rng.integers(0, H))
                noisy[row, :] *= (rng.random(W) > noise_level).astype(np.float32)
            else:
                col = int(rng.integers(0, W))
                noisy[:, col] *= (rng.random(H) > noise_level).astype(np.float32)
        return noisy

    return binary  # unreachable


# ─── Stability report ─────────────────────────────────────────────────────────

@dataclass
class StabilityReport:
    """
    Graph stability metrics across noise levels.

    Per noise_level (indexed by position):
      node_count_cv:         coefficient of variation of node count across seeds
      edge_count_cv:         CV of edge count
      quality_score_cv:      CV of graph_quality_score
      fragmentation_cv:      CV of fragmentation_index
      mean_tortuosity_cv:    CV of mean_tortuosity
      node_drift:            mean Euclidean shift of matched nodes (pixels)
      stability_score:       composite stability in [0, 1]; 1 = perfectly stable
    """
    noise_type:         str
    noise_levels:       List[float]
    n_seeds:            int
    node_count_cv:      List[float] = field(default_factory=list)
    edge_count_cv:      List[float] = field(default_factory=list)
    quality_score_cv:   List[float] = field(default_factory=list)
    fragmentation_cv:   List[float] = field(default_factory=list)
    mean_tortuosity_cv: List[float] = field(default_factory=list)
    stability_score:    List[float] = field(default_factory=list)
    timing_s:           float = 0.0


# ─── Graph Stability Analyzer ─────────────────────────────────────────────────

class GraphStabilityAnalyzer:
    """
    Measures vascular graph stability under synthetic segmentation noise.

    For each noise level, runs `n_seeds` independent perturbations of the
    input mask, builds a graph for each, and measures the coefficient of
    variation of key graph metrics across the seed ensemble.

    Low CV = stable graph (insensitive to segmentation noise) — desirable.
    High CV = fragile graph (metric values fluctuate wildly) — problematic.

    Usage:
        from rapid_twin.graph.graph_builder import RetinalGraphBuilder

        builder  = RetinalGraphBuilder(min_branch_length=5, max_nodes=512)
        analyzer = GraphStabilityAnalyzer(noise_type="salt_pepper", n_seeds=5)
        report   = analyzer.analyze(mask, graph_fn=lambda m: builder.build(m))
        analyzer.print_report(report)
    """

    def __init__(
        self,
        noise_type:   str          = "salt_pepper",
        noise_levels: List[float]  = None,
        n_seeds:      int          = 5,
        verbose:      bool         = False,
    ):
        self.noise_type   = noise_type
        self.noise_levels = noise_levels if noise_levels is not None else [0.02, 0.05, 0.10, 0.20]
        self.n_seeds      = n_seeds
        self.verbose      = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GraphStability] {msg}")

    def analyze(
        self,
        mask:     np.ndarray,
        graph_fn: Callable[[np.ndarray], object],
    ) -> StabilityReport:
        """
        Analyze graph metric stability across noise levels.

        Args:
            mask:     (H, W) float32 binary vessel mask.
            graph_fn: callable that takes a (H, W) binary mask and returns a
                      VascularGraph.  Must be a pure function (no global state).

        Returns:
            StabilityReport.
        """
        from rapid_twin.graph.graph_metrics import compute_all_graph_metrics

        t0     = time.perf_counter()
        report = StabilityReport(
            noise_type=self.noise_type,
            noise_levels=self.noise_levels,
            n_seeds=self.n_seeds,
        )

        for level in self.noise_levels:
            self._log(f"Noise level {level:.2f}")
            metrics_per_seed: List[Dict[str, float]] = []

            for seed in range(self.n_seeds):
                try:
                    noisy  = inject_noise(mask, self.noise_type, level, seed)
                    graph  = graph_fn(noisy)
                    m      = compute_all_graph_metrics(
                        graph, noisy,
                        compute_efficiency=False,
                        compute_fractal=False,
                    )
                    m["n_nodes"] = float(len(graph.nodes))
                    m["n_edges"] = float(len(graph.edges))
                    metrics_per_seed.append(m)
                except Exception as exc:
                    self._log(f"  seed={seed} level={level} failed: {exc}")

            if not metrics_per_seed:
                report.node_count_cv.append(float("nan"))
                report.edge_count_cv.append(float("nan"))
                report.quality_score_cv.append(float("nan"))
                report.fragmentation_cv.append(float("nan"))
                report.mean_tortuosity_cv.append(float("nan"))
                report.stability_score.append(float("nan"))
                continue

            def _cv(key: str) -> float:
                vals = [d[key] for d in metrics_per_seed if np.isfinite(d.get(key, float("nan")))]
                if not vals or abs(np.mean(vals)) < 1e-8:
                    return float("nan")
                return float(np.std(vals) / (abs(np.mean(vals)) + 1e-8))

            nc_cv = _cv("n_nodes")
            ec_cv = _cv("n_edges")
            qs_cv = _cv("graph_quality_score")
            fi_cv = _cv("fragmentation_index")
            mt_cv = _cv("mean_tortuosity")

            report.node_count_cv.append(nc_cv)
            report.edge_count_cv.append(ec_cv)
            report.quality_score_cv.append(qs_cv)
            report.fragmentation_cv.append(fi_cv)
            report.mean_tortuosity_cv.append(mt_cv)

            # Composite stability: 1 - mean of finite CVs (clipped to [0,1])
            cv_vals = [v for v in [nc_cv, ec_cv, qs_cv, fi_cv, mt_cv] if np.isfinite(v)]
            if cv_vals:
                stability = float(np.clip(1.0 - np.mean(cv_vals), 0.0, 1.0))
            else:
                stability = float("nan")
            report.stability_score.append(stability)

            self._log(
                f"  Stability={stability:.3f}  "
                f"node_cv={nc_cv:.3f}  edge_cv={ec_cv:.3f}  qs_cv={qs_cv:.3f}"
            )

        report.timing_s = time.perf_counter() - t0
        return report

    @staticmethod
    def print_report(report: StabilityReport):
        print(f"\n{'─'*65}")
        print(f"  Graph Robustness Report — {report.noise_type}")
        print(f"  n_seeds={report.n_seeds}  t={report.timing_s:.2f}s")
        print(f"{'─'*65}")
        print(f"  {'Noise':>8}  {'Stability':>10}  {'Node CV':>10}  "
              f"{'Edge CV':>10}  {'QScore CV':>10}")
        for i, level in enumerate(report.noise_levels):
            def _fmt(lst, i):
                v = lst[i] if i < len(lst) else float("nan")
                return f"{v:.4f}" if np.isfinite(v) else "  nan"
            print(
                f"  {level:>8.3f}  "
                f"{_fmt(report.stability_score, i):>10}  "
                f"{_fmt(report.node_count_cv, i):>10}  "
                f"{_fmt(report.edge_count_cv, i):>10}  "
                f"{_fmt(report.quality_score_cv, i):>10}"
            )
        print()


# ─── Degenerate input testing ─────────────────────────────────────────────────

@dataclass
class DegenerateTestResult:
    """Result for a single degenerate input test case."""
    case_name:   str
    passed:      bool     # True = handled gracefully (no crash, correct failure)
    description: str
    error:       Optional[str] = None


def test_degenerate_inputs(
    graph_fn: Callable[[np.ndarray], object],
    H: int = 64,
    W: int = 64,
) -> List[DegenerateTestResult]:
    """
    Systematically test the graph pipeline against known degenerate inputs.

    Test cases:
      1. All-zero mask        — empty segmentation
      2. All-one mask         — fully filled image
      3. Single-pixel mask    — degenerate 1-vessel segmentation
      4. Single horizontal line — minimal graph topology
      5. Checkerboard noise   — highly fragmented / impossible graph
      6. Single dot           — isolated endpoint, no edges
      7. Near-zero float mask — below threshold, should produce empty graph
      8. Very thin (1px wide) vessel
      9. Only-endpoints mask  — disconnected pixels, no vessel paths

    Args:
        graph_fn: callable (H, W) binary mask → VascularGraph.
        H, W:     image dimensions.

    Returns:
        List of DegenerateTestResult.
    """
    from rapid_twin.graph.graph_metrics import compute_all_graph_metrics

    results: List[DegenerateTestResult] = []

    def _run(case_name: str, mask: np.ndarray, expect_empty: bool = False) -> DegenerateTestResult:
        try:
            graph   = graph_fn(mask)
            metrics = compute_all_graph_metrics(graph, mask, compute_efficiency=False, compute_fractal=False)
            n_nodes = len(graph.nodes)
            if expect_empty and n_nodes > 0:
                return DegenerateTestResult(
                    case_name=case_name, passed=False,
                    description=f"Expected empty graph, got {n_nodes} nodes",
                )
            q = metrics.get("graph_quality_score", 0.0)
            assert 0.0 <= q <= 1.0, f"quality_score={q} out of [0,1]"
            return DegenerateTestResult(
                case_name=case_name, passed=True,
                description=f"OK: {n_nodes} nodes, quality={q:.3f}",
            )
        except Exception as exc:
            return DegenerateTestResult(
                case_name=case_name, passed=False,
                description=f"CRASH: {type(exc).__name__}",
                error=str(exc),
            )

    # Case 1: all-zero
    results.append(_run("all_zero", np.zeros((H, W), np.float32), expect_empty=True))

    # Case 2: all-one
    results.append(_run("all_one", np.ones((H, W), np.float32)))

    # Case 3: single pixel
    m = np.zeros((H, W), np.float32); m[H//2, W//2] = 1.0
    results.append(_run("single_pixel", m, expect_empty=True))

    # Case 4: single horizontal line
    m = np.zeros((H, W), np.float32); m[H//2, :] = 1.0
    results.append(_run("single_line", m))

    # Case 5: checkerboard (very fragmented)
    rr, cc = np.mgrid[:H, :W]
    m = ((rr + cc) % 2).astype(np.float32)
    results.append(_run("checkerboard", m))

    # Case 6: four isolated dots
    m = np.zeros((H, W), np.float32)
    for r, c in [(H//4, W//4), (H//4, 3*W//4), (3*H//4, W//4), (3*H//4, 3*W//4)]:
        m[r, c] = 1.0
    results.append(_run("isolated_dots", m, expect_empty=True))

    # Case 7: near-zero (all below threshold)
    m = np.full((H, W), 0.1, dtype=np.float32)
    results.append(_run("near_zero", m, expect_empty=True))

    # Case 8: 1-pixel-wide diagonal vessel
    m = np.zeros((H, W), np.float32)
    for i in range(min(H, W)):
        m[i, i] = 1.0
    results.append(_run("diagonal_vessel", m))

    # Case 9: cross shape (simple bifurcation)
    m = np.zeros((H, W), np.float32)
    m[H//2, :] = 1.0
    m[:, W//2] = 1.0
    results.append(_run("cross_shape", m))

    return results


def print_degenerate_report(results: List[DegenerateTestResult]):
    """Print formatted degenerate input test results."""
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'─'*60}")
    print(f"  Degenerate Input Coverage — {passed}/{len(results)} passed")
    print(f"{'─'*60}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}]  {r.case_name:<25}  {r.description}")
    print()
