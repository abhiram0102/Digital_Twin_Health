"""
Topology-aware retinal vessel metrics.

Provides metrics beyond pixel-level Dice that measure vascular graph quality,
continuity, and thin-vessel recovery — the properties that matter most for
downstream graph extraction and digital twin integration.

All functions accept binary numpy masks (H, W) of dtype bool or float32.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Core topology metrics ────────────────────────────────────────────────────

def skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning via iterative morphological erosion.
    Returns binary skeleton (same shape, dtype uint8, values 0/255).
    """
    binary = (mask > 0.5).astype(np.uint8) * 255
    skel   = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded  = cv2.erode(binary, kernel)
        opened  = cv2.dilate(eroded, kernel)
        temp    = cv2.subtract(binary, opened)
        skel    = cv2.bitwise_or(skel, temp)
        binary  = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return skel


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Returns (num_components, label_image) for a binary prediction mask.
    """
    binary = (mask > 0.5).astype(np.uint8)
    n_components, labels = cv2.connectedComponents(binary, connectivity=8)
    return n_components - 1, labels


def clDice_numpy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Centreline Dice (clDice) — topology preservation metric.

    Measures how well the predicted skeleton overlaps the GT vessel mask
    and vice versa.  Higher = better vessel continuity.
    """
    skel_pred = skeletonize(pred) / 255.0
    skel_gt   = skeletonize(gt)   / 255.0
    pred_f = (pred > 0.5).astype(np.float32)
    gt_f   = (gt   > 0.5).astype(np.float32)

    prec_num = (skel_pred * gt_f).sum()
    prec_den = skel_pred.sum()
    sens_num = (skel_gt * pred_f).sum()
    sens_den = skel_gt.sum()

    if prec_den < 1 or sens_den < 1:
        return 0.0

    tprec = prec_num / prec_den
    tsens = sens_num / sens_den
    denom = tprec + tsens
    return (2.0 * tprec * tsens / denom) if denom > 0 else 0.0


def vessel_continuity_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Vessel continuity: fraction of GT connected components that have at least
    one corresponding connected region in the prediction.

    Score in [0, 1].  Low score → vessel fragmentation in prediction.
    """
    gt_bin   = (gt   > 0.5).astype(np.uint8)
    pred_bin = (pred > 0.5).astype(np.uint8)

    n_gt, gt_labels = cv2.connectedComponents(gt_bin, connectivity=8)
    if n_gt <= 1:
        return 1.0

    covered = 0
    for comp_id in range(1, n_gt):
        component_mask = (gt_labels == comp_id).astype(np.uint8)
        if (component_mask * pred_bin).sum() > 0:
            covered += 1

    return covered / (n_gt - 1)


def branch_preservation_score(pred: np.ndarray, gt: np.ndarray,
                               min_branch_px: int = 10) -> float:
    """
    What fraction of GT vessel branches (components larger than min_branch_px)
    are represented in the prediction.  Robust to noise suppression.
    """
    gt_bin   = (gt   > 0.5).astype(np.uint8)
    pred_bin = (pred > 0.5).astype(np.uint8)

    n_gt, gt_labels = cv2.connectedComponents(gt_bin, connectivity=8)
    large_branches  = 0
    covered         = 0

    for comp_id in range(1, n_gt):
        mask = (gt_labels == comp_id).astype(np.uint8)
        if mask.sum() < min_branch_px:
            continue
        large_branches += 1
        if (mask * pred_bin).sum() > 0:
            covered += 1

    return covered / large_branches if large_branches > 0 else 1.0


def thin_vessel_sensitivity(pred: np.ndarray, gt: np.ndarray,
                             thin_radius_px: int = 3) -> float:
    """
    Recall specifically on thin vessels (vessels narrower than thin_radius_px).

    Thin vessels are identified by morphological erosion: pixels that disappear
    after erosion with a disk of radius thin_radius_px are classified as thin.
    """
    gt_bin   = (gt > 0.5).astype(np.uint8)
    kernel   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thin_radius_px * 2 + 1, thin_radius_px * 2 + 1))
    thick    = cv2.erode(gt_bin, kernel)
    thin_mask = ((gt_bin - thick) > 0).astype(np.float32)

    if thin_mask.sum() < 1:
        return 1.0

    pred_f = (pred > 0.5).astype(np.float32)
    return float((thin_mask * pred_f).sum() / thin_mask.sum())


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance between predicted and GT vessel boundaries.

    Lower = better.  Returns pixel distance.
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin   = (gt   > 0.5).astype(np.uint8)

    pred_edge = cv2.Canny(pred_bin * 255, 50, 150)
    gt_edge   = cv2.Canny(gt_bin   * 255, 50, 150)

    pred_pts = np.argwhere(pred_edge > 0).astype(np.float32)
    gt_pts   = np.argwhere(gt_edge   > 0).astype(np.float32)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    from scipy.spatial import cKDTree  # optional import
    try:
        tree_gt   = cKDTree(gt_pts)
        tree_pred = cKDTree(pred_pts)
        d_pred_gt = tree_gt.query(pred_pts)[0]
        d_gt_pred = tree_pred.query(gt_pts)[0]
        return float(max(np.percentile(d_pred_gt, 95), np.percentile(d_gt_pred, 95)))
    except Exception:
        return float("nan")


def vessel_length_preservation(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Ratio of predicted skeleton length to GT skeleton length.

    Values close to 1.0 indicate good vessel-length preservation.
    Values < 1.0 indicate fragmentation / missed vessels.
    Values > 1.0 indicate spurious vessels.
    """
    len_pred = (skeletonize(pred) > 0).sum()
    len_gt   = (skeletonize(gt)   > 0).sum()
    if len_gt == 0:
        return 1.0
    return float(len_pred / len_gt)


# ─── Standard segmentation metrics ───────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = (pred > 0.5).astype(np.float32)
    gt_f   = (gt   > 0.5).astype(np.float32)
    inter  = (pred_f * gt_f).sum()
    union  = pred_f.sum() + gt_f.sum()
    return float(2 * inter / (union + 1e-8))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = (pred > 0.5).astype(np.float32)
    gt_f   = (gt   > 0.5).astype(np.float32)
    inter  = (pred_f * gt_f).sum()
    union  = pred_f.sum() + gt_f.sum() - inter
    return float(inter / (union + 1e-8))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = (pred > 0.5).astype(np.float32)
    gt_f   = (gt   > 0.5).astype(np.float32)
    tp = (pred_f * gt_f).sum()
    fn = ((1 - pred_f) * gt_f).sum()
    return float(tp / (tp + fn + 1e-8))


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = (pred > 0.5).astype(np.float32)
    gt_f   = (gt   > 0.5).astype(np.float32)
    tn = ((1 - pred_f) * (1 - gt_f)).sum()
    fp = (pred_f * (1 - gt_f)).sum()
    return float(tn / (tn + fp + 1e-8))


# ─── Full metric bundle ───────────────────────────────────────────────────────

def compute_all_metrics(
    pred: np.ndarray,
    gt:   np.ndarray,
    compute_hausdorff: bool = False,
    thin_radius_px:    int  = 3,
    min_branch_px:     int  = 10,
) -> Dict[str, float]:
    """
    Compute the full suite of retinal vessel metrics for one image.

    Args:
        pred:              (H, W) predicted probability map or binary mask
        gt:                (H, W) ground-truth binary mask
        compute_hausdorff: include 95th-pct Hausdorff (slow; requires scipy)
        thin_radius_px:    radius threshold for thin-vessel classification
        min_branch_px:     min branch size for branch_preservation_score

    Returns:
        dict of metric_name → float
    """
    metrics = {
        "dice":                   dice_coefficient(pred, gt),
        "iou":                    iou(pred, gt),
        "sensitivity":            sensitivity(pred, gt),
        "specificity":            specificity(pred, gt),
        "cldice":                 clDice_numpy(pred, gt),
        "vessel_continuity":      vessel_continuity_score(pred, gt),
        "branch_preservation":    branch_preservation_score(pred, gt, min_branch_px),
        "thin_vessel_sensitivity": thin_vessel_sensitivity(pred, gt, thin_radius_px),
        "vessel_length_ratio":    vessel_length_preservation(pred, gt),
    }

    n_pred, _ = connected_components(pred)
    n_gt,   _ = connected_components(gt)
    metrics["n_components_pred"] = float(n_pred)
    metrics["n_components_gt"]   = float(n_gt)
    metrics["fragmentation_ratio"] = float(n_pred / max(n_gt, 1))

    if compute_hausdorff:
        metrics["hausdorff_95"] = hausdorff_distance_95(pred, gt)

    return metrics


# ─── Per-dataset benchmarking ─────────────────────────────────────────────────

class SegmentationBenchmark:
    """
    Accumulates per-image metrics and produces per-dataset + overall summaries.

    Usage:
        bench = SegmentationBenchmark()
        for pred, gt, dataset_name in eval_generator():
            bench.update(pred, gt, dataset=dataset_name)
        summary = bench.summary()
    """

    def __init__(self, compute_hausdorff: bool = False):
        self.compute_hausdorff = compute_hausdorff
        self._records: List[Dict] = []

    def update(self, pred: np.ndarray, gt: np.ndarray, dataset: str = "unknown"):
        m = compute_all_metrics(pred, gt,
                                compute_hausdorff=self.compute_hausdorff)
        m["dataset"] = dataset
        self._records.append(m)

    def summary(self) -> Dict[str, Dict[str, float]]:
        if not self._records:
            return {}

        METRIC_KEYS = [k for k in self._records[0] if k != "dataset"]
        all_ds = sorted(set(r["dataset"] for r in self._records))
        result: Dict[str, Dict[str, float]] = {}

        for ds in all_ds + ["overall"]:
            rows = ([r for r in self._records if r["dataset"] == ds]
                    if ds != "overall" else self._records)
            if not rows:
                continue
            result[ds] = {
                k: float(np.mean([r[k] for r in rows]))
                for k in METRIC_KEYS
            }

        return result

    def print_report(self):
        summary = self.summary()
        print(f"\n{'='*70}")
        print("  Retinal Vessel Segmentation Benchmark")
        print(f"{'='*70}")

        key_metrics = ["dice", "iou", "sensitivity", "cldice",
                       "vessel_continuity", "thin_vessel_sensitivity",
                       "vessel_length_ratio"]

        header = f"{'Dataset':>12}"
        for k in key_metrics:
            header += f"  {k:>10}"
        print(header)
        print("─" * len(header))

        for ds_name, metrics in summary.items():
            row = f"{ds_name:>12}"
            for k in key_metrics:
                v = metrics.get(k, float("nan"))
                row += f"  {v:>10.4f}"
            print(row)
        print()
