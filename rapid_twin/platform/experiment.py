"""
Experiment Tracking — Stage 4, Research-Grade Platform.

Lightweight, zero-dependency experiment tracker for reproducible research.
All state is stored in-memory (and optionally written to JSON files).

Compatible with the existing benchmark suite and profiling infrastructure.
MLflow / W&B integration can be layered on top by implementing the same
interface, but is not required.

Provides:
  ExperimentRun     — one model run with metrics, config, and artifact paths
  ExperimentTracker — manages multiple runs, comparison, and reporting

Usage:
    tracker = ExperimentTracker(project="rapid_twin_ablation")

    run = tracker.start_run(
        name="full_pipeline_drive",
        config={"dataset": "DRIVE", "min_branch_length": 5},
        tags=["segmentation", "baseline"],
    )

    tracker.log_metric(run.run_id, "dice",              0.812)
    tracker.log_metric(run.run_id, "graph_quality",     0.743)
    tracker.log_benchmark(run.run_id, benchmark_result)

    tracker.finish_run(run.run_id)
    tracker.print_leaderboard(metric="dice")
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ─── Run ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentRun:
    """
    One experiment run.

    Attributes:
        run_id:      unique identifier (uuid4 hex).
        name:        human-readable run name.
        project:     project name this run belongs to.
        config:      hyperparameters and settings dict.
        tags:        list of category tags.
        metrics:     dict of metric_name → list of (step, value) tuples.
        benchmarks:  dict of benchmark_name → BenchmarkResult dict.
        artifacts:   list of artifact file paths.
        start_time:  Unix timestamp of run start.
        end_time:    Unix timestamp of run end (0 if still running).
        status:      "running" / "finished" / "failed".
    """
    run_id:     str
    name:       str
    project:    str
    config:     Dict[str, Any]                      = field(default_factory=dict)
    tags:       List[str]                           = field(default_factory=list)
    metrics:    Dict[str, List[tuple]]              = field(default_factory=dict)
    benchmarks: Dict[str, Dict[str, Any]]           = field(default_factory=dict)
    artifacts:  List[str]                           = field(default_factory=list)
    start_time: float                               = field(default_factory=time.time)
    end_time:   float                               = 0.0
    status:     str                                 = "running"

    @property
    def duration_s(self) -> float:
        """Wall-clock duration in seconds. 0 if still running."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def latest_metric(self, name: str) -> float:
        """Return the most recently logged value for a metric."""
        series = self.metrics.get(name, [])
        if not series:
            return float("nan")
        return float(series[-1][1])

    def best_metric(self, name: str, mode: str = "max") -> float:
        """Return the best (max or min) logged value for a metric."""
        series = self.metrics.get(name, [])
        if not series:
            return float("nan")
        vals = [v for _, v in series if math.isfinite(v)]
        if not vals:
            return float("nan")
        return float(max(vals) if mode == "max" else min(vals))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":     self.run_id,
            "name":       self.name,
            "project":    self.project,
            "config":     self.config,
            "tags":       self.tags,
            "metrics":    {k: [[s, v] for s, v in vs] for k, vs in self.metrics.items()},
            "benchmarks": self.benchmarks,
            "artifacts":  self.artifacts,
            "start_time": self.start_time,
            "end_time":   self.end_time,
            "status":     self.status,
            "duration_s": self.duration_s,
        }


# ─── Tracker ──────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Manages multiple experiment runs with comparison and reporting.

    State is stored in-memory.  Optionally saves to a JSON directory
    when save_dir is provided.

    Usage:
        tracker = ExperimentTracker(project="rapid_twin", save_dir="./runs")
        run = tracker.start_run(name="v1", config={"dataset": "DRIVE"})
        tracker.log_metric(run.run_id, "dice", 0.81, step=1)
        tracker.finish_run(run.run_id)
        tracker.print_leaderboard("dice")
    """

    def __init__(self, project: str = "rapid_twin", save_dir: Optional[str] = None):
        self.project  = project
        self.save_dir = save_dir
        self._runs:   Dict[str, ExperimentRun] = {}

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def start_run(
        self,
        name:   str,
        config: Dict[str, Any] = None,
        tags:   List[str]      = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            name:   human-readable run name.
            config: hyperparameter dict.
            tags:   category tags.

        Returns:
            ExperimentRun (status="running").
        """
        run_id = uuid.uuid4().hex[:12]
        run    = ExperimentRun(
            run_id=run_id,
            name=name,
            project=self.project,
            config=config or {},
            tags=tags or [],
        )
        self._runs[run_id] = run
        return run

    def finish_run(
        self,
        run_id: str,
        status: str = "finished",
    ) -> ExperimentRun:
        """
        Mark a run as complete and persist to disk if save_dir is set.

        Args:
            run_id: run identifier.
            status: "finished" or "failed".

        Returns:
            Finished ExperimentRun.
        """
        run = self._get(run_id)
        run.end_time = time.time()
        run.status   = status
        self._save(run)
        return run

    def fail_run(self, run_id: str) -> ExperimentRun:
        """Mark a run as failed."""
        return self.finish_run(run_id, status="failed")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_metric(
        self,
        run_id: str,
        name:   str,
        value:  float,
        step:   int = None,
    ) -> None:
        """
        Log a scalar metric value.

        Args:
            run_id: run identifier.
            name:   metric name (e.g., "dice", "ece").
            value:  scalar float.
            step:   optional step/epoch index. Defaults to count of existing entries.
        """
        run = self._get(run_id)
        if name not in run.metrics:
            run.metrics[name] = []
        s = step if step is not None else len(run.metrics[name])
        run.metrics[name].append((s, float(value)))

    def log_metrics(
        self,
        run_id:  str,
        metrics: Dict[str, float],
        step:    int = None,
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            if math.isfinite(value):
                self.log_metric(run_id, name, value, step)

    def log_benchmark(
        self,
        run_id: str,
        result,                 # BenchmarkResult
    ) -> None:
        """
        Log a BenchmarkResult.

        Args:
            run_id: run identifier.
            result: BenchmarkResult from any benchmark module.
        """
        run = self._get(run_id)
        run.benchmarks[result.name] = {
            "dataset":   result.dataset,
            "n_samples": result.n_samples,
            "timing_s":  result.timing_s,
            "metrics":   {k: v for k, v in result.metrics.items()
                          if math.isfinite(v)},
        }
        # Also log key metrics as individual metric series
        for k, v in result.metrics.items():
            if math.isfinite(v) and not k.endswith("_std"):
                self.log_metric(run_id, f"{result.name}/{k}", v)

    def log_artifact(self, run_id: str, path: str) -> None:
        """Register an artifact file path with a run."""
        self._get(run_id).artifacts.append(str(path))

    def set_tags(self, run_id: str, tags: List[str]) -> None:
        """Add tags to a run."""
        self._get(run_id).tags.extend(tags)

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_run(self, run_id: str) -> ExperimentRun:
        return self._get(run_id)

    def list_runs(
        self,
        tag:    Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ExperimentRun]:
        """Return runs filtered by tag and/or status."""
        runs = list(self._runs.values())
        if tag:
            runs = [r for r in runs if tag in r.tags]
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.start_time)

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare specific metrics across multiple runs.

        Returns:
            dict of run_id → dict of metric_name → latest value.
        """
        result = {}
        for run_id in run_ids:
            run = self._get(run_id)
            result[run_id] = {
                m: run.latest_metric(m) for m in metrics
            }
        return result

    def best_run(self, metric: str, mode: str = "max") -> Optional[ExperimentRun]:
        """Return the run with the best value for a given metric."""
        finished = [r for r in self._runs.values() if r.status == "finished"]
        if not finished:
            return None
        fn = max if mode == "max" else min
        return fn(finished, key=lambda r: r.latest_metric(metric)
                  if math.isfinite(r.latest_metric(metric)) else -float("inf"))

    # ── Reporting ──────────────────────────────────────────────────────────────

    def print_leaderboard(
        self,
        metric:    str       = "dice",
        mode:      str       = "max",
        top_n:     int       = 10,
        extra:     List[str] = None,
    ) -> None:
        """
        Print a leaderboard sorted by a primary metric.

        Args:
            metric: primary ranking metric.
            mode:   "max" or "min".
            top_n:  show top N runs only.
            extra:  additional metric columns to show.
        """
        runs     = [r for r in self._runs.values() if r.status == "finished"]
        fn       = (lambda r: r.latest_metric(metric)) if mode == "max" else (lambda r: -r.latest_metric(metric))
        runs     = sorted(runs, key=fn, reverse=(mode == "max"))[:top_n]
        extra    = extra or []
        all_cols = [metric] + extra

        print(f"\n{'='*65}")
        print(f"  Leaderboard: {self.project}  (by {metric}, {mode})")
        print(f"{'='*65}")
        header = f"  {'Rank':>4}  {'Name':<28}  " + "".join(f"{c[:12]:>13}" for c in all_cols)
        print(header)

        for i, run in enumerate(runs, 1):
            vals = "".join(
                f"{run.latest_metric(c):>13.4f}" if math.isfinite(run.latest_metric(c))
                else f"{'nan':>13}"
                for c in all_cols
            )
            print(f"  {i:>4}  {run.name:<28}  {vals}  [{run.run_id}]")
        print()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, run: ExperimentRun) -> None:
        if self.save_dir is None:
            return
        path = os.path.join(self.save_dir, f"{run.run_id}.json")
        with open(path, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def save_all(self) -> None:
        """Persist all runs to the save directory."""
        for run in self._runs.values():
            self._save(run)

    def load_run(self, path: str) -> ExperimentRun:
        """Load a previously saved run from a JSON file."""
        with open(path) as f:
            d = json.load(f)
        run = ExperimentRun(
            run_id=d["run_id"], name=d["name"], project=d["project"],
            config=d["config"], tags=d["tags"],
            metrics={k: [tuple(x) for x in v] for k, v in d["metrics"].items()},
            benchmarks=d["benchmarks"], artifacts=d["artifacts"],
            start_time=d["start_time"], end_time=d["end_time"], status=d["status"],
        )
        self._runs[run.run_id] = run
        return run

    def _get(self, run_id: str) -> ExperimentRun:
        if run_id not in self._runs:
            raise KeyError(f"Run '{run_id}' not found in tracker '{self.project}'.")
        return self._runs[run_id]
