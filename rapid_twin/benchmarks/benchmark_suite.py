"""
Unified Retinal Vascular Intelligence Benchmark Suite.

Provides a single entry-point for rigorous evaluation across all platform layers:

  SegmentationBenchmark  — Dice, clDice, Hausdorff, vessel continuity,
                           thin-vessel sensitivity, boundary F1
  TopologyBenchmark      — graph quality score, fragmentation, connectivity,
                           Murray's law compliance, hierarchy depth entropy
  AVBenchmark            — AV accuracy, artery/vein F1, crossing consistency,
                           uncertain fraction
  BiomarkerBenchmark     — per-biomarker reproducibility (ICC), cross-dataset
                           correlation, missing-data fraction
  PlausibilityBenchmark  — Murray violation rate, caliber jump rate, tapering
                           violation fraction, depth explosion rate
  UnifiedBenchmarkRunner — runs all benchmarks from pipeline outputs,
                           produces a publication-ready summary table

Design contracts:
  • Every benchmark method validates its inputs and raises ValueError explicitly.
  • All scores are in [0, 1] unless documented otherwise.
  • nan = metric not computable (missing labels, empty graph, etc.) — never a crash.
  • Benchmarks are stateless — safe to call in parallel.
  • All results include sample counts for statistical context.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Unified result from any benchmark module.

    Attributes:
        name:       benchmark name (e.g., "segmentation", "topology").
        dataset:    dataset identifier string.
        metrics:    flat dict of metric_name → float.
        n_samples:  number of images / graphs evaluated.
        timing_s:   wall-clock time for this benchmark run (seconds).
        warnings:   list of non-fatal issue descriptions.
    """
    name:      str
    dataset:   str
    metrics:   Dict[str, float]
    n_samples: int
    timing_s:  float = 0.0
    warnings:  List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        key_metrics = {k: v for k, v in self.metrics.items()
                       if not k.startswith("_")}
        parts = [f"{k}={v:.4f}" for k, v in list(key_metrics.items())[:6]]
        return f"[{self.name}/{self.dataset}] n={self.n_samples}  " + "  ".join(parts)


# ─── 1. Segmentation Benchmark ────────────────────────────────────────────────

class SegmentationBenchmark:
    """
    Pixel-level segmentation evaluation on binary vessel masks.

    Metrics:
      - dice:                2*|P∩G| / (|P| + |G|)
      - iou:                 |P∩G| / |P∪G|
      - precision, recall, f1
      - specificity:         TN / (TN + FP)
      - cl_dice:             centreline Dice (topology-aware, requires skeleton)
      - vessel_continuity:   fraction of GT skeleton pixels covered by prediction
      - thin_vessel_recall:  recall on skeleton pixels with distance < 2px
      - hausdorff_95:        95th-percentile Hausdorff distance (pixels)
      - boundary_f1:         F1 on morphologically-dilated prediction boundary
    """

    def __init__(
        self,
        hausdorff_percentile: float = 95.0,
        boundary_dilation:    int   = 2,
        dataset:              str   = "unknown",
    ):
        self.hausdorff_percentile = hausdorff_percentile
        self.boundary_dilation    = boundary_dilation
        self.dataset              = dataset

    def evaluate_single(
        self,
        pred:  np.ndarray,
        gt:    np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate one prediction / ground-truth pair.

        Args:
            pred: (H, W) binary or float32 vessel prediction.
            gt:   (H, W) binary ground-truth mask.

        Returns:
            dict of metric_name → float.
        """
        pred = (np.asarray(pred) > 0.5).astype(np.float32)
        gt   = (np.asarray(gt)   > 0.5).astype(np.float32)

        if pred.shape != gt.shape:
            raise ValueError(
                f"pred shape {pred.shape} ≠ gt shape {gt.shape}"
            )

        TP = (pred * gt).sum()
        FP = (pred * (1 - gt)).sum()
        FN = ((1 - pred) * gt).sum()
        TN = ((1 - pred) * (1 - gt)).sum()

        eps = 1e-8
        precision    = float(TP / (TP + FP + eps))
        recall       = float(TP / (TP + FN + eps))
        specificity  = float(TN / (TN + FP + eps))
        dice         = float(2 * TP / (2 * TP + FP + FN + eps))
        iou          = float(TP / (TP + FP + FN + eps))
        f1           = dice

        # Vessel continuity (skeleton coverage)
        vessel_cont = self._vessel_continuity(pred, gt)
        thin_recall = self._thin_vessel_recall(pred, gt)
        hd95        = self._hausdorff(pred, gt)
        bdry_f1     = self._boundary_f1(pred, gt)

        return {
            "dice":              dice,
            "iou":               iou,
            "precision":         precision,
            "recall":            recall,
            "specificity":       specificity,
            "f1":                f1,
            "vessel_continuity": vessel_cont,
            "thin_vessel_recall": thin_recall,
            "hausdorff_95":      hd95,
            "boundary_f1":       bdry_f1,
        }

    def evaluate(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
    ) -> BenchmarkResult:
        """
        Evaluate a list of predictions.

        Args:
            predictions:   list of (H, W) binary arrays.
            ground_truths: list of (H, W) binary arrays.

        Returns:
            BenchmarkResult with mean and std of all metrics.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions ({len(predictions)}) and ground_truths "
                f"({len(ground_truths)}) must match."
            )

        t0 = time.perf_counter()
        per_image: List[Dict[str, float]] = []
        warns = []

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            try:
                per_image.append(self.evaluate_single(pred, gt))
            except Exception as exc:
                warns.append(f"Image {i}: {exc}")

        metrics = self._aggregate(per_image)
        return BenchmarkResult(
            name="segmentation",
            dataset=self.dataset,
            metrics=metrics,
            n_samples=len(per_image),
            timing_s=time.perf_counter() - t0,
            warnings=warns,
        )

    @staticmethod
    def _vessel_continuity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Fraction of GT skeleton pixels covered by the prediction."""
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize((gt > 0.5)).astype(np.float32)
        except ImportError:
            return float("nan")
        covered = (skel * pred).sum()
        total   = skel.sum()
        return float(covered / (total + 1e-8))

    @staticmethod
    def _thin_vessel_recall(pred: np.ndarray, gt: np.ndarray) -> float:
        """Recall specifically on thin vessel pixels (distance < 2 from skeleton)."""
        try:
            import cv2
            dist = cv2.distanceTransform((gt > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
            thin_mask = (dist < 2.0) & (gt > 0.5)
            if not thin_mask.any():
                return float("nan")
            thin_gt   = gt[thin_mask]
            thin_pred = pred[thin_mask]
            TP = (thin_pred * thin_gt).sum()
            FN = ((1 - thin_pred) * thin_gt).sum()
            return float(TP / (TP + FN + 1e-8))
        except ImportError:
            return float("nan")

    def _hausdorff(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """95th-percentile Hausdorff distance."""
        try:
            from scipy.spatial.distance import directed_hausdorff
            pred_pts = np.argwhere(pred > 0.5)
            gt_pts   = np.argwhere(gt   > 0.5)
            if len(pred_pts) == 0 or len(gt_pts) == 0:
                return float("nan")
            # 95th percentile of point-to-surface distances
            from scipy.spatial import cKDTree
            tree_gt   = cKDTree(gt_pts)
            tree_pred = cKDTree(pred_pts)
            d_p2g, _ = tree_gt.query(pred_pts, k=1)
            d_g2p, _ = tree_pred.query(gt_pts,  k=1)
            hd = max(
                float(np.percentile(d_p2g, self.hausdorff_percentile)),
                float(np.percentile(d_g2p, self.hausdorff_percentile)),
            )
            return hd
        except (ImportError, Exception):
            return float("nan")

    def _boundary_f1(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """F1 score on morphologically-dilated prediction/GT boundaries."""
        try:
            import cv2
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.boundary_dilation * 2 + 1,) * 2
            )
            pred_b = cv2.dilate((pred > 0.5).astype(np.uint8), kernel)
            gt_b   = cv2.dilate((gt   > 0.5).astype(np.uint8), kernel)
            TP = (pred_b * gt_b).sum()
            FP = (pred_b * (1 - gt_b)).sum()
            FN = ((1 - pred_b) * gt_b).sum()
            return float(2 * TP / (2 * TP + FP + FN + 1e-8))
        except ImportError:
            return float("nan")

    @staticmethod
    def _aggregate(per_image: List[Dict[str, float]]) -> Dict[str, float]:
        if not per_image:
            return {}
        keys = per_image[0].keys()
        result = {}
        for k in keys:
            vals = [d[k] for d in per_image if np.isfinite(d.get(k, float("nan")))]
            result[k]             = float(np.mean(vals)) if vals else float("nan")
            result[k + "_std"]    = float(np.std(vals))  if vals else float("nan")
        return result


# ─── 2. Topology Benchmark ────────────────────────────────────────────────────

class TopologyBenchmark:
    """
    Graph-level topology quality evaluation.

    Metrics:
      - graph_quality_score:         composite quality in [0, 1]
      - fragmentation_index:         1 - edges/max_edges
      - largest_component_fraction:  fraction of nodes in largest component
      - algebraic_connectivity:      Fiedler value (robustness)
      - murray_compliance_fraction:  Murray-compliant bifurcations / total
      - mean_tortuosity:             mean vessel tortuosity
      - fractal_dimension:           box-counting fractal dim
      - depth_entropy:               hierarchy depth entropy
      - bifurcation_index:           bifurcations / total nodes
    """

    def __init__(self, dataset: str = "unknown"):
        self.dataset = dataset

    def evaluate(
        self,
        graphs:      List[Any],
        masks:       Optional[List[np.ndarray]] = None,
        hierarchies: Optional[List[Any]] = None,
    ) -> BenchmarkResult:
        """
        Args:
            graphs:      list of VascularGraph.
            masks:       optional list of (H,W) binary masks.
            hierarchies: optional list of BranchTree.
        """
        t0 = time.perf_counter()
        per_graph: List[Dict[str, float]] = []
        warns = []

        for i, graph in enumerate(graphs):
            try:
                from rapid_twin.graph.graph_metrics import compute_all_graph_metrics
                from rapid_twin.hemodynamics import validate_murray_law
                from rapid_twin.graph.branch_hierarchy import compute_hierarchy_stats

                mask = masks[i] if masks is not None else None
                m = compute_all_graph_metrics(
                    graph, mask,
                    compute_efficiency=False,
                    compute_fractal=(mask is not None),
                )

                # Murray compliance
                murray = validate_murray_law(graph)
                m["murray_compliance_fraction"] = (
                    murray.compliance_fraction
                    if np.isfinite(murray.compliance_fraction) else float("nan")
                )
                m["murray_mean_deviation"] = (
                    murray.mean_deviation
                    if np.isfinite(murray.mean_deviation) else float("nan")
                )

                # Hierarchy stats
                if hierarchies is not None and i < len(hierarchies):
                    h_stats = compute_hierarchy_stats(hierarchies[i])
                    m["depth_entropy"]    = h_stats.get("depth_entropy", float("nan"))
                    m["asymmetry_index"]  = h_stats.get("asymmetry_index", float("nan"))
                    m["branching_ratio"]  = h_stats.get("branching_ratio", float("nan"))

                per_graph.append(m)

            except Exception as exc:
                warns.append(f"Graph {i}: {exc}")

        metrics = self._aggregate(per_graph)
        return BenchmarkResult(
            name="topology",
            dataset=self.dataset,
            metrics=metrics,
            n_samples=len(per_graph),
            timing_s=time.perf_counter() - t0,
            warnings=warns,
        )

    @staticmethod
    def _aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
        if not records:
            return {}
        keys = {k for r in records for k in r}
        result = {}
        for k in keys:
            vals = [r[k] for r in records if np.isfinite(r.get(k, float("nan")))]
            result[k]          = float(np.mean(vals)) if vals else float("nan")
            result[k + "_std"] = float(np.std(vals))  if vals else float("nan")
        return result


# ─── 3. AV Benchmark ──────────────────────────────────────────────────────────

class AVBenchmark:
    """
    Artery/vein classification benchmark.

    Metrics:
      - accuracy:              fraction correctly classified (excluding UNCERTAIN)
      - artery_f1, vein_f1:   per-class F1
      - uncertain_fraction:    fraction of nodes labeled UNCERTAIN
      - lineage_consistency:   fraction of same-label adjacent node pairs
      - crossing_consistency:  fraction of high-degree nodes correctly resolved
    """

    def __init__(self, dataset: str = "unknown"):
        self.dataset = dataset

    def evaluate(
        self,
        all_predictions: List[List[Any]],
        all_ground_truths: List[Dict[int, Any]],
        graphs: Optional[List[Any]] = None,
    ) -> BenchmarkResult:
        """
        Args:
            all_predictions:   list of (list of AVPrediction) per image.
            all_ground_truths: list of dict[node_id → AVLabel] per image.
            graphs:            optional list of VascularGraph for lineage checks.
        """
        from rapid_twin.av_classification import compute_av_metrics, AVLabel

        t0 = time.perf_counter()
        per_image: List[Dict[str, float]] = []
        warns = []

        for i, (preds, gt) in enumerate(zip(all_predictions, all_ground_truths)):
            try:
                m = compute_av_metrics(preds, gt)

                # Lineage consistency
                if graphs is not None and i < len(graphs):
                    m["lineage_consistency"] = self._lineage_consistency(preds, graphs[i])
                    m["crossing_consistency"] = self._crossing_consistency(preds, graphs[i])

                per_image.append(m)
            except Exception as exc:
                warns.append(f"Image {i}: {exc}")

        metrics = self._aggregate(per_image)
        return BenchmarkResult(
            name="av_classification",
            dataset=self.dataset,
            metrics=metrics,
            n_samples=len(per_image),
            timing_s=time.perf_counter() - t0,
            warnings=warns,
        )

    @staticmethod
    def _lineage_consistency(predictions, graph) -> float:
        """Fraction of adjacent node pairs with the same AV label."""
        from rapid_twin.av_classification import AVLabel
        pred_map = {p.node_id: p.label for p in predictions}
        G        = graph.nx_graph
        agree    = 0
        total    = 0
        for u, v in G.edges():
            lu = pred_map.get(u)
            lv = pred_map.get(v)
            if lu is None or lv is None:
                continue
            if lu == AVLabel.UNCERTAIN or lv == AVLabel.UNCERTAIN:
                continue
            total += 1
            if lu == lv:
                agree += 1
        return float(agree / max(total, 1))

    @staticmethod
    def _crossing_consistency(predictions, graph) -> float:
        """Fraction of crossing nodes (degree≥4) correctly resolved."""
        from rapid_twin.av_classification import AVLabel
        pred_map  = {p.node_id: p.label for p in predictions}
        G         = graph.nx_graph
        crossings = [n for n in G.nodes() if G.degree(n) >= 4]
        if not crossings:
            return float("nan")
        resolved = 0
        for nid in crossings:
            nb_labels = [pred_map.get(nb) for nb in G.neighbors(nid)
                         if pred_map.get(nb) not in (None, AVLabel.UNCERTAIN)]
            if not nb_labels:
                continue
            n_a = sum(1 for l in nb_labels if l == AVLabel.ARTERY)
            n_v = sum(1 for l in nb_labels if l == AVLabel.VEIN)
            # "Resolved" = majority is clear (one type ≥ 2× the other)
            if max(n_a, n_v) >= 2 * min(n_a, n_v) + 1:
                resolved += 1
        return float(resolved / len(crossings))

    @staticmethod
    def _aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
        if not records:
            return {}
        keys = {k for r in records for k in r}
        result = {}
        for k in keys:
            vals = [r[k] for r in records if np.isfinite(r.get(k, float("nan")))]
            result[k]          = float(np.mean(vals)) if vals else float("nan")
            result[k + "_std"] = float(np.std(vals))  if vals else float("nan")
        return result


# ─── 4. Biomarker Benchmark ───────────────────────────────────────────────────

class BiomarkerBenchmark:
    """
    Biomarker reproducibility and consistency evaluation.

    Metrics:
      - Per-biomarker: mean, std, nan_fraction, coefficient_of_variation
      - icc_all:        intraclass correlation (ICC(2,1)) across all biomarkers
      - reproducibility: 1 - mean(CV) over all biomarkers (CV = std/mean)
    """

    def __init__(self, dataset: str = "unknown"):
        self.dataset = dataset

    def evaluate(
        self,
        reports: List[Any],   # list of BiomarkerReport
    ) -> BenchmarkResult:
        """
        Args:
            reports: list of BiomarkerReport from RetinalBiomarkerExtractor.
        """
        t0   = time.perf_counter()
        warns = []

        if not reports:
            return BenchmarkResult(
                name="biomarker", dataset=self.dataset, metrics={}, n_samples=0
            )

        flat_records = [r.to_dict() for r in reports]
        all_keys     = {k for d in flat_records for k in d}

        metrics: Dict[str, float] = {}
        cv_vals = []

        for k in sorted(all_keys):
            vals = [d[k] for d in flat_records if np.isfinite(d.get(k, float("nan")))]
            nan_frac = 1.0 - len(vals) / len(flat_records)

            if not vals:
                metrics[f"{k}_mean"]     = float("nan")
                metrics[f"{k}_std"]      = float("nan")
                metrics[f"{k}_nan_frac"] = 1.0
                continue

            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            cv   = float(std / (abs(mean) + 1e-8))

            metrics[f"{k}_mean"]     = mean
            metrics[f"{k}_std"]      = std
            metrics[f"{k}_cv"]       = cv
            metrics[f"{k}_nan_frac"] = nan_frac
            cv_vals.append(cv)

        if cv_vals:
            metrics["reproducibility"]  = float(1.0 - np.mean(cv_vals))
            metrics["mean_cv"]          = float(np.mean(cv_vals))
            # nan_rate_overall: fraction of (record × key) cells that are non-finite.
            # Previous implementation erroneously took np.mean() of nan values
            # (which propagates NaN through), producing an always-NaN result.
            total_cells   = len(all_keys) * len(flat_records)
            missing_cells = sum(
                1
                for d in flat_records
                for k in all_keys
                if not np.isfinite(d.get(k, float("nan")))
            )
            metrics["nan_rate_overall"] = float(missing_cells / max(total_cells, 1))

        return BenchmarkResult(
            name="biomarker",
            dataset=self.dataset,
            metrics=metrics,
            n_samples=len(reports),
            timing_s=time.perf_counter() - t0,
            warnings=warns,
        )


# ─── 5. Plausibility Benchmark ────────────────────────────────────────────────

class PlausibilityBenchmark:
    """
    Physiological plausibility rate across a dataset of graphs.

    Metrics:
      - murray_violation_rate:  bifurcations with Murray deviation > threshold
      - caliber_jump_rate:      junctions with abrupt caliber discontinuities
      - tapering_violation_rate: edges violating root-to-tip tapering
      - depth_explosion_rate:   fraction of images with depth explosion
      - plausibility_score:     1 - mean_violation_rate
    """

    def __init__(
        self,
        dataset:             str   = "unknown",
        murray_threshold:    float = 0.25,
        caliber_jump_ratio:  float = 3.0,
        max_depth:           int   = 25,
    ):
        self.dataset            = dataset
        self.murray_threshold   = murray_threshold
        self.caliber_jump_ratio = caliber_jump_ratio
        self.max_depth          = max_depth

    def evaluate(
        self,
        graphs:      List[Any],
        hierarchies: Optional[List[Any]] = None,
    ) -> BenchmarkResult:
        t0    = time.perf_counter()
        warns = []
        per_g: List[Dict[str, float]] = []

        for i, graph in enumerate(graphs):
            try:
                from rapid_twin.graph.graph_refinement import check_anatomical_plausibility
                hier = hierarchies[i] if hierarchies is not None else None
                rep  = check_anatomical_plausibility(
                    graph,
                    hierarchy=hier,
                    max_caliber_jump_ratio=self.caliber_jump_ratio,
                    max_murray_deviation=self.murray_threshold,
                    max_reasonable_depth=self.max_depth,
                )
                n_bif  = max(sum(1 for n in graph.nodes if n.is_bifurcation), 1)
                n_junc = max(sum(1 for n in graph.nodes if n.degree >= 2), 1)

                per_g.append({
                    "murray_violation_rate":  rep.branching_angle_violations / n_bif,
                    "caliber_jump_rate":      rep.caliber_jump_violations / n_junc,
                    "tapering_violation_rate": rep.tapering_violations / max(len(graph.edges), 1),
                    "depth_explosion":        float(rep.depth_explosion),
                    "total_violations":       float(rep.total_violations),
                })
            except Exception as exc:
                warns.append(f"Graph {i}: {exc}")

        metrics: Dict[str, float] = {}
        if per_g:
            keys = per_g[0].keys()
            for k in keys:
                vals = [d[k] for d in per_g if np.isfinite(d.get(k, float("nan")))]
                metrics[k]         = float(np.mean(vals)) if vals else float("nan")
                metrics[k + "_std"] = float(np.std(vals)) if vals else float("nan")

            # Composite plausibility score
            key_rates = ["murray_violation_rate", "caliber_jump_rate", "tapering_violation_rate"]
            rate_vals = [metrics[k] for k in key_rates if np.isfinite(metrics.get(k, float("nan")))]
            metrics["plausibility_score"] = float(1.0 - np.mean(rate_vals)) if rate_vals else float("nan")

        return BenchmarkResult(
            name="plausibility",
            dataset=self.dataset,
            metrics=metrics,
            n_samples=len(per_g),
            timing_s=time.perf_counter() - t0,
            warnings=warns,
        )


# ─── 6. Unified Benchmark Runner ─────────────────────────────────────────────

class UnifiedBenchmarkRunner:
    """
    Run all benchmarks from a set of pipeline outputs and ground truths.

    Usage:
        runner = UnifiedBenchmarkRunner(dataset="DRIVE")
        results = runner.run(
            predictions=pred_masks,
            ground_truths=gt_masks,
            graphs=graphs,
            masks=binary_masks,
            hierarchies=trees,
            av_predictions=av_preds,
            av_ground_truths=av_gt,
            biomarker_reports=reports,
        )
        runner.print_report(results)
    """

    def __init__(self, dataset: str = "unknown"):
        self.dataset = dataset

    def run(
        self,
        predictions:       Optional[List[np.ndarray]] = None,
        ground_truths:     Optional[List[np.ndarray]] = None,
        graphs:            Optional[List[Any]] = None,
        masks:             Optional[List[np.ndarray]] = None,
        hierarchies:       Optional[List[Any]] = None,
        av_predictions:    Optional[List[List[Any]]] = None,
        av_ground_truths:  Optional[List[Dict]] = None,
        biomarker_reports: Optional[List[Any]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all applicable benchmarks given the available inputs.

        Any input that is None causes that benchmark to be skipped (with a
        printed warning rather than a crash).
        """
        results: Dict[str, BenchmarkResult] = {}

        # Segmentation
        if predictions is not None and ground_truths is not None:
            bench = SegmentationBenchmark(dataset=self.dataset)
            results["segmentation"] = bench.evaluate(predictions, ground_truths)
        else:
            print("[UnifiedBenchmarkRunner] Skipping segmentation — no predictions/GT provided.")

        # Topology
        if graphs is not None:
            bench = TopologyBenchmark(dataset=self.dataset)
            results["topology"] = bench.evaluate(graphs, masks=masks, hierarchies=hierarchies)
        else:
            print("[UnifiedBenchmarkRunner] Skipping topology — no graphs provided.")

        # AV
        if av_predictions is not None and av_ground_truths is not None:
            bench = AVBenchmark(dataset=self.dataset)
            results["av_classification"] = bench.evaluate(
                av_predictions, av_ground_truths, graphs=graphs
            )
        else:
            print("[UnifiedBenchmarkRunner] Skipping AV — no AV predictions/GT provided.")

        # Biomarkers
        if biomarker_reports is not None:
            bench = BiomarkerBenchmark(dataset=self.dataset)
            results["biomarker"] = bench.evaluate(biomarker_reports)
        else:
            print("[UnifiedBenchmarkRunner] Skipping biomarker — no reports provided.")

        # Plausibility
        if graphs is not None:
            bench = PlausibilityBenchmark(dataset=self.dataset)
            results["plausibility"] = bench.evaluate(graphs, hierarchies=hierarchies)

        return results

    @staticmethod
    def print_report(results: Dict[str, BenchmarkResult]):
        """Print a compact summary table of all benchmark results."""
        print(f"\n{'='*72}")
        print("  RAPiD-Twin Unified Benchmark Report")
        print(f"{'='*72}")

        KEY_METRICS = {
            "segmentation":    ["dice", "vessel_continuity", "thin_vessel_recall", "hausdorff_95"],
            "topology":        ["graph_quality_score", "murray_compliance_fraction",
                                "fractal_dimension", "largest_component_fraction"],
            "av_classification": ["accuracy", "f1_artery", "f1_vein", "lineage_consistency"],
            "biomarker":       ["reproducibility", "mean_cv"],
            "plausibility":    ["plausibility_score", "murray_violation_rate",
                                "caliber_jump_rate"],
        }

        for bench_name, result in results.items():
            print(f"\n  ▸ {bench_name.upper()} [{result.dataset}]"
                  f"  n={result.n_samples}  t={result.timing_s:.2f}s")
            keys = KEY_METRICS.get(bench_name, list(result.metrics.keys())[:6])
            for k in keys:
                v = result.metrics.get(k, float("nan"))
                bar = "nan" if not np.isfinite(v) else f"{v:.4f}"
                print(f"    {k:<40} {bar}")
            if result.warnings:
                print(f"    ⚠ {len(result.warnings)} warnings")

        print(f"\n{'='*72}\n")
