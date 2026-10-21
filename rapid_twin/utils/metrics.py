"""
Clinical & Research Metrics Suite for RAPiD-Twin.

Covers:
  Segmentation:   Dice, IoU, Hausdorff Distance (95th pct), Sensitivity, Specificity
  Classification: Accuracy, Precision, Recall, F1 (macro/weighted), AUC-ROC,
                  Sensitivity, Specificity, per-class breakdown
  Calibration:    Expected Calibration Error (ECE), Brier Score
  Utility:        set_seed, count_parameters, AverageMeter
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        confusion_matrix, average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─── Basic utility ────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running mean of a scalar metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def count_parameters(model) -> int:
    if not TORCH_AVAILABLE:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Segmentation metrics ─────────────────────────────────────────────────────

def dice_coefficient(
    pred:      Union[np.ndarray, "torch.Tensor"],
    target:    Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    smooth:    float = 1.0,
) -> float:
    """Dice / F1 score for binary segmentation maps."""
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        pred_bin   = (pred > threshold).float()
        inter      = (pred_bin * target).sum()
        union      = pred_bin.sum() + target.sum()
        return float((2 * inter + smooth) / (union + smooth))
    pred_bin = (np.asarray(pred) > threshold).astype(float)
    tgt      = np.asarray(target).astype(float)
    inter    = (pred_bin * tgt).sum()
    return float((2 * inter + smooth) / (pred_bin.sum() + tgt.sum() + smooth))


def iou_score(
    pred:      Union[np.ndarray, "torch.Tensor"],
    target:    Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
) -> float:
    """Intersection-over-Union (Jaccard index)."""
    pred_bin = (np.asarray(pred) > threshold).astype(float)
    tgt      = np.asarray(target).astype(float)
    inter    = (pred_bin * tgt).sum()
    union    = pred_bin.sum() + tgt.sum() - inter
    return float(inter / (union + 1e-8))


def hausdorff_distance_95(
    pred:      Union[np.ndarray, "torch.Tensor"],
    target:    Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
) -> float:
    """
    95th-percentile Hausdorff Distance (HD95) in pixels.

    Computed as: max(perc_95(d_pred→gt), perc_95(d_gt→pred))
    where distances are computed over binary boundary pixels.

    Falls back to 0.0 if either mask is empty or scipy is unavailable.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0

    pred_bin = (np.asarray(pred) > threshold).astype(bool)
    tgt_bin  = (np.asarray(target) > 0.5).astype(bool)

    # Handle batch dim
    if pred_bin.ndim == 4:
        pred_bin = pred_bin[:, 0]
        tgt_bin  = tgt_bin[:, 0]

    if pred_bin.ndim == 3:
        hd_vals = []
        for p, t in zip(pred_bin, tgt_bin):
            hd_vals.append(_hd95_2d(p, t))
        return float(np.mean(hd_vals))
    return _hd95_2d(pred_bin, tgt_bin)


def _hd95_2d(pred: np.ndarray, tgt: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt
    if not pred.any() or not tgt.any():
        return 0.0
    # Distance from pred boundary to nearest tgt pixel
    dist_gt  = distance_transform_edt(~tgt)
    dist_pred = distance_transform_edt(~pred)
    d1 = dist_gt[pred]
    d2 = dist_pred[tgt]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def sensitivity_specificity(
    pred:      Union[np.ndarray, "torch.Tensor"],
    target:    Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Sensitivity, specificity, precision, F1 from binary maps."""
    pred_bin = (np.asarray(pred) > threshold).astype(float)
    tgt      = np.asarray(target).astype(float)
    tp = (pred_bin * tgt).sum()
    fp = (pred_bin * (1 - tgt)).sum()
    fn = ((1 - pred_bin) * tgt).sum()
    tn = ((1 - pred_bin) * (1 - tgt)).sum()
    prec = tp / (tp + fp + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1   = 2 * prec * sens / (prec + sens + 1e-8)
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision":   float(prec),
        "f1":          float(f1),
    }


def segmentation_metrics_full(
    pred:      Union[np.ndarray, "torch.Tensor"],
    target:    Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    compute_hd: bool = True,
) -> Dict[str, float]:
    """
    Full segmentation metric bundle.

    Returns:
        dice, iou, sensitivity, specificity, precision, f1,
        hausdorff_95 (if compute_hd=True)
    """
    ss  = sensitivity_specificity(pred, target, threshold)
    out = {
        "dice":        dice_coefficient(pred, target, threshold),
        "iou":         iou_score(pred, target, threshold),
        **ss,
    }
    if compute_hd:
        out["hausdorff_95"] = hausdorff_distance_95(pred, target, threshold)
    return out


# ─── Classification metrics ───────────────────────────────────────────────────

def classification_metrics_full(
    y_true:        np.ndarray,              # (N,)  int labels
    y_pred_proba:  np.ndarray,              # (N, C) or (N,) probabilities
    y_pred_label:  Optional[np.ndarray] = None,   # (N,) if None computed from proba
    num_classes:   Optional[int] = None,
) -> Dict[str, float]:
    """
    Full classification metric bundle.

    Returns:
        accuracy, auc_roc, f1_macro, f1_weighted,
        precision_macro, recall_macro,
        sensitivity (binary: any positive vs. negative),
        specificity (binary),
        average_precision,
        per_class_f1 (dict)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_pred_proba)

    if y_pred_label is None:
        if y_proba.ndim == 2:
            y_pred_label = y_proba.argmax(axis=1)
        else:
            y_pred_label = (y_proba > 0.5).astype(int)
    y_pred_label = np.asarray(y_pred_label, dtype=int)

    n   = len(y_true)
    acc = float((y_pred_label == y_true).mean())

    metrics: Dict[str, float] = {"accuracy": acc, "num_samples": float(n)}

    # AUC-ROC
    if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
        try:
            if y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba,
                                        multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(y_true, y_proba)
            metrics["auc_roc"] = float(auc)
        except Exception:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    # F1 / Precision / Recall
    if SKLEARN_AVAILABLE:
        try:
            metrics["f1_macro"]        = float(f1_score(y_true, y_pred_label, average="macro",    zero_division=0))
            metrics["f1_weighted"]     = float(f1_score(y_true, y_pred_label, average="weighted", zero_division=0))
            metrics["precision_macro"] = float(precision_score(y_true, y_pred_label, average="macro",    zero_division=0))
            metrics["recall_macro"]    = float(recall_score(y_true, y_pred_label, average="macro",       zero_division=0))
        except Exception:
            for k in ("f1_macro", "f1_weighted", "precision_macro", "recall_macro"):
                metrics[k] = 0.0

        # Per-class F1
        try:
            per_f1 = f1_score(y_true, y_pred_label, average=None, zero_division=0)
            for i, v in enumerate(per_f1):
                metrics[f"f1_class_{i}"] = float(v)
        except Exception:
            pass

        # Average Precision (mean of per-class AP)
        try:
            if y_proba.ndim == 2 and len(np.unique(y_true)) > 1:
                ap = average_precision_score(
                    np.eye(y_proba.shape[1])[y_true], y_proba,
                    average="macro"
                )
                metrics["average_precision"] = float(ap)
        except Exception:
            pass
    else:
        for k in ("f1_macro", "f1_weighted", "precision_macro", "recall_macro"):
            metrics[k] = 0.0

    # Binary sensitivity / specificity (positive = any non-zero class)
    y_bin_true = (y_true > 0).astype(int)
    y_bin_pred = (y_pred_label > 0).astype(int)
    tp = int(((y_bin_pred == 1) & (y_bin_true == 1)).sum())
    tn = int(((y_bin_pred == 0) & (y_bin_true == 0)).sum())
    fp = int(((y_bin_pred == 1) & (y_bin_true == 0)).sum())
    fn = int(((y_bin_pred == 0) & (y_bin_true == 1)).sum())
    metrics["sensitivity"] = tp / (tp + fn + 1e-8)
    metrics["specificity"] = tn / (tn + fp + 1e-8)

    return metrics


# ─── Calibration metrics ──────────────────────────────────────────────────────

def expected_calibration_error(
    y_true:  np.ndarray,    # (N,) int
    y_proba: np.ndarray,    # (N, C) softmax probabilities
    n_bins:  int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).
    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true    = np.asarray(y_true, dtype=int)
    y_proba   = np.asarray(y_proba)
    conf      = y_proba.max(axis=1)
    pred      = y_proba.argmax(axis=1)
    correct   = (pred == y_true).astype(float)
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    N         = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece   += (mask.sum() / N) * abs(acc_b - conf_b)
    return float(ece)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Multiclass Brier Score = mean squared distance between
    one-hot labels and predicted probabilities.
    """
    y_true  = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba)
    C       = y_proba.shape[1]
    one_hot = np.eye(C)[y_true]
    return float(((y_proba - one_hot) ** 2).sum(axis=1).mean())
