"""
Pipeline Performance Profiling — Memory and Compute Tracking.

Provides lightweight, zero-dependency profiling for all pipeline stages:

  profile_stage()       — context manager: wall-time + RSS memory delta
  PipelineProfiler      — run a callable N times and collect timing stats
  ProfilingReport       — structured results with bottleneck identification
  profile_graph_scaling — measure time vs graph size (nodes/edges)

Design contracts:
  • No heavy profiling dependencies required (cProfile, py-spy, etc.).
    Uses only `time`, `resource`, and `tracemalloc` from stdlib.
  • Memory is measured as RSS delta (resident set size) using `resource`.
    Falls back to tracemalloc if resource is not available (Windows).
  • All timing uses `time.perf_counter()` for maximum precision.
  • ProfilingReport is JSON-serialisable (no numpy arrays, no torch tensors).
  • profile_stage is thread-safe for independent stage profiling.
  • No stdout output unless verbose=True or explicitly called.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np


# ─── Detect memory measurement method ─────────────────────────────────────────

try:
    import resource as _resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False   # Windows


def _get_rss_mb() -> float:
    """Resident set size in MiB.  Returns 0.0 if not available."""
    if _HAS_RESOURCE:
        # Linux: KB; macOS: bytes
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0  # KB → MB
    return 0.0


# ─── Stage result ─────────────────────────────────────────────────────────────

@dataclass
class StageProfile:
    """Profiling result for a single pipeline stage."""
    name:         str
    wall_time_s:  float        # elapsed wall-clock seconds
    memory_mb:    float        # peak memory delta (MiB); 0 if not available
    n_calls:      int = 1      # for aggregated multi-run profiles

    # Aggregated stats (filled when n_calls > 1)
    mean_time_s:  float = 0.0
    std_time_s:   float = 0.0
    min_time_s:   float = 0.0
    max_time_s:   float = 0.0

    @property
    def throughput_per_s(self) -> float:
        """Calls per second based on mean time."""
        return 1.0 / (self.mean_time_s + 1e-12)


@dataclass
class ProfilingReport:
    """
    Complete profiling report for a pipeline run.

    Attributes:
        stages:         dict of stage_name → StageProfile.
        total_time_s:   sum of all stage times.
        bottleneck:     name of the slowest stage.
        scaling:        dict of graph_size → time_s (from profile_graph_scaling).
        n_nodes_tested: list of node counts tested in scaling analysis.
        metadata:       arbitrary key-value pairs (e.g., device, image size).
    """
    stages:         Dict[str, StageProfile]
    total_time_s:   float
    bottleneck:     str
    scaling:        Dict[int, float] = field(default_factory=dict)
    n_nodes_tested: List[int]        = field(default_factory=list)
    metadata:       Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict."""
        out = {
            "total_time_s": self.total_time_s,
            "bottleneck":   self.bottleneck,
            "metadata":     self.metadata,
            "stages":       {},
        }
        for name, sp in self.stages.items():
            out["stages"][name] = {
                "wall_time_s": sp.wall_time_s,
                "memory_mb":   sp.memory_mb,
                "mean_time_s": sp.mean_time_s,
                "std_time_s":  sp.std_time_s,
            }
        return out


# ─── Context manager ──────────────────────────────────────────────────────────

@contextmanager
def profile_stage(name: str) -> Generator[Dict[str, float], None, None]:
    """
    Context manager for profiling a single code block.

    Usage:
        with profile_stage("postprocess") as stats:
            result = postprocess(mask)
        print(stats["wall_time_s"], stats["memory_mb"])

    Yields:
        dict that is populated AFTER the block exits:
          - wall_time_s: elapsed seconds
          - memory_mb:   peak tracemalloc delta (MiB)
          - rss_delta_mb: RSS delta if resource is available
    """
    stats: Dict[str, float] = {"wall_time_s": 0.0, "memory_mb": 0.0, "rss_delta_mb": 0.0}

    tracemalloc.start()
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _get_rss_mb()

        stats["wall_time_s"]  = float(elapsed)
        stats["memory_mb"]    = float(peak / (1024 * 1024))     # bytes → MB
        stats["rss_delta_mb"] = float(max(0.0, rss_after - rss_before))


# ─── Pipeline Profiler ────────────────────────────────────────────────────────

class PipelineProfiler:
    """
    Profile a callable across N repeated runs to collect stable timing stats.

    Usage:
        profiler = PipelineProfiler(n_runs=5, warmup=1)

        @profiler.profile("graph_build")
        def build(mask):
            return builder.build(mask)

        build(mask)
        report = profiler.make_report()
        profiler.print_report(report)
    """

    def __init__(
        self,
        n_runs:  int  = 5,
        warmup:  int  = 1,
        verbose: bool = False,
    ):
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        self.n_runs  = n_runs
        self.warmup  = warmup
        self.verbose = verbose
        self._records: Dict[str, List[float]] = {}      # stage → list of times
        self._memory:  Dict[str, List[float]] = {}      # stage → list of peak MBs

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Profiler] {msg}")

    def profile(self, stage_name: str):
        """Decorator that wraps a function to collect per-call timing."""
        def decorator(fn: Callable):
            def wrapper(*args, **kwargs):
                # Warmup
                for _ in range(self.warmup):
                    _ = fn(*args, **kwargs)
                    gc.collect()

                # Timed runs
                times, mems = [], []
                for r in range(self.n_runs):
                    with profile_stage(stage_name) as stats:
                        result = fn(*args, **kwargs)
                    times.append(stats["wall_time_s"])
                    mems.append(stats["memory_mb"])
                    gc.collect()
                    self._log(f"{stage_name} run {r}: {stats['wall_time_s']:.4f}s")

                self._records.setdefault(stage_name, []).extend(times)
                self._memory.setdefault(stage_name, []).extend(mems)
                return result
            return wrapper
        return decorator

    def record(self, stage_name: str, time_s: float, memory_mb: float = 0.0):
        """Manually record a stage profiling result."""
        self._records.setdefault(stage_name, []).append(time_s)
        self._memory.setdefault(stage_name, []).append(memory_mb)

    def make_report(self, metadata: Optional[Dict[str, Any]] = None) -> ProfilingReport:
        """Aggregate all recorded timings into a ProfilingReport."""
        stages: Dict[str, StageProfile] = {}

        for name, times in self._records.items():
            times_arr = np.array(times)
            mems_arr  = np.array(self._memory.get(name, [0.0]))
            sp = StageProfile(
                name=name,
                wall_time_s=float(times_arr.sum()),
                memory_mb=float(mems_arr.max()),
                n_calls=len(times),
                mean_time_s=float(times_arr.mean()),
                std_time_s=float(times_arr.std()),
                min_time_s=float(times_arr.min()),
                max_time_s=float(times_arr.max()),
            )
            stages[name] = sp

        total = float(sum(sp.mean_time_s for sp in stages.values()))
        bottleneck = max(stages, key=lambda k: stages[k].mean_time_s) if stages else "none"

        return ProfilingReport(
            stages=stages,
            total_time_s=total,
            bottleneck=bottleneck,
            metadata=metadata or {},
        )

    def print_report(self, report: Optional[ProfilingReport] = None):
        """Print a formatted profiling table."""
        if report is None:
            report = self.make_report()

        total = report.total_time_s
        print(f"\n{'─'*70}")
        print(f"  Pipeline Profiling Report")
        if report.metadata:
            for k, v in report.metadata.items():
                print(f"  {k}: {v}")
        print(f"{'─'*70}")
        print(f"  {'Stage':<30} {'Mean(s)':>10} {'Std(s)':>10} {'Mem(MB)':>10} {'%Total':>8}")

        for name, sp in sorted(report.stages.items(), key=lambda x: -x[1].mean_time_s):
            pct = 100.0 * sp.mean_time_s / (total + 1e-12)
            flag = " ◀ BOTTLENECK" if name == report.bottleneck else ""
            print(
                f"  {name:<30} "
                f"{sp.mean_time_s:>10.4f} "
                f"{sp.std_time_s:>10.4f} "
                f"{sp.memory_mb:>10.2f} "
                f"{pct:>7.1f}%"
                f"{flag}"
            )

        print(f"{'─'*70}")
        print(f"  {'TOTAL':<30} {total:>10.4f}")
        print(f"  Bottleneck: {report.bottleneck}")
        print()


# ─── Graph-size scaling analysis ──────────────────────────────────────────────

def profile_graph_scaling(
    graph_fn:   Callable[[np.ndarray], object],
    mask_sizes: Optional[List[Tuple[int, int]]] = None,
    metric_fn:  Optional[Callable] = None,
    n_runs:     int = 3,
) -> ProfilingReport:
    """
    Measure pipeline time as a function of graph/image size.

    Generates synthetic masks of increasing size and measures the time to:
      (a) build the graph (graph_fn)
      (b) optionally compute metrics (metric_fn)

    Args:
        graph_fn:   callable (H, W) mask → VascularGraph.
        mask_sizes: list of (H, W) pairs to test.  Default: [(64,64) ... (512,512)].
        metric_fn:  optional callable VascularGraph → dict (for metric timing).
        n_runs:     runs per size for stability.

    Returns:
        ProfilingReport with scaling dict {n_pixels: mean_time_s}.
    """
    if mask_sizes is None:
        mask_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    scaling: Dict[int, float] = {}
    n_nodes_tested: List[int] = []
    profiler = PipelineProfiler(n_runs=n_runs, warmup=0)

    for H, W in mask_sizes:
        # Generate a synthetic retinal-like mask
        mask = _synthetic_mask(H, W)
        n_px = H * W

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            graph = graph_fn(mask)
            if metric_fn is not None:
                metric_fn(graph)
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        scaling[n_px] = mean_t
        n_nodes_tested.append(n_px)

        # Also record for the profiler
        stage_name = f"size_{H}x{W}"
        for t in times:
            profiler.record(stage_name, t)

    report = profiler.make_report(metadata={"analysis": "graph_scaling"})
    report.scaling        = scaling
    report.n_nodes_tested = n_nodes_tested
    return report


def _synthetic_mask(H: int, W: int) -> np.ndarray:
    """Generate a simple synthetic vessel-like mask of given size."""
    mask = np.zeros((H, W), dtype=np.float32)
    # Horizontal and vertical vessels at regular intervals
    step = max(8, min(H, W) // 8)
    for r in range(0, H, step):
        mask[r, :] = 1.0
    for c in range(0, W, step):
        mask[:, c] = 1.0
    return mask


# ─── Numerical stability checker ─────────────────────────────────────────────

def check_numerical_stability(
    graph,
    mask: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Audit all feature matrices and computed metrics for numerical issues.

    Checks:
      - NaN / Inf in node_features, edge_features
      - Very large values (|x| > 1e6) suggesting un-normalised features
      - Tortuosity < 1.0 (physically impossible)
      - Negative calibers or lengths
      - Quality score outside [0, 1]

    Args:
        graph:     VascularGraph.
        mask:      optional binary mask for skeleton-based checks.
        tolerance: numerical tolerance for near-zero checks.

    Returns:
        dict with issue descriptions and counts.
    """
    from rapid_twin.graph.graph_metrics import compute_all_graph_metrics

    issues: Dict[str, Any] = {
        "nan_in_node_features":  int((~np.isfinite(graph.node_features)).sum()),
        "nan_in_edge_features":  int((~np.isfinite(graph.edge_features)).sum()),
        "large_node_values":     int((np.abs(graph.node_features) > 1e6).sum()),
        "large_edge_values":     int((np.abs(graph.edge_features) > 1e6).sum()),
        "negative_calibers":     sum(1 for e in graph.edges if e.mean_diameter < 0),
        "negative_lengths":      sum(1 for e in graph.edges if e.length < 0),
        "impossible_tortuosity": sum(1 for e in graph.edges if e.tortuosity < 1.0 - tolerance),
    }

    # Metrics check
    try:
        m = compute_all_graph_metrics(graph, mask, compute_efficiency=False, compute_fractal=False)
        q = m.get("graph_quality_score", float("nan"))
        issues["quality_score_valid"]    = 0.0 <= q <= 1.0 if np.isfinite(q) else False
        issues["nan_metric_count"]       = sum(1 for v in m.values() if not np.isfinite(v))
        issues["total_metric_count"]     = len(m)
        issues["finite_metric_fraction"] = 1.0 - issues["nan_metric_count"] / max(len(m), 1)
    except Exception as exc:
        issues["metrics_error"] = str(exc)

    # Summary
    critical = (
        issues["nan_in_node_features"]  > 0
        or issues["nan_in_edge_features"] > 0
        or issues["impossible_tortuosity"] > 0
    )
    issues["has_critical_issues"] = critical
    issues["total_issues"]        = sum(
        v for k, v in issues.items()
        if isinstance(v, int) and k not in ("total_metric_count", "total_issues")
    )

    return issues
