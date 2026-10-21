"""
Per-Task Calibration: Temperature Scaling and Conformal Prediction.

Provides post-training calibration for the multi-task disease prediction heads.

Classes:
    PerTaskTemperatureScaler  — one temperature parameter per task, LBFGS fitting.
    SplitConformalPredictor   — split conformal prediction with guaranteed coverage.
    CalibrationReport         — serialisable calibration metrics per task.

Mathematical guarantees:
    Temperature scaling:
        Minimises NLL on the calibration set; typically reduces ECE from ~0.15 to
        < 0.05 for over-confident models (Guo et al., 2017).

    Split conformal:
        P(true_class ∈ prediction_set) ≥ 1 − α  (finite-sample, distribution-free).
        Holds for any model and distribution as long as calibration and test sets
        are i.i.d. (Angelopoulos & Bates, 2022, arXiv:2107.07511).

Usage:
    # Temperature scaling
    scaler = PerTaskTemperatureScaler(task_names, n_classes_per_task)
    scaler.calibrate(val_logits_dict, val_labels_dict)
    print(scaler.ece("diabetic_retinopathy", probs, labels))
    scaler.save("calibration.json")

    # Conformal prediction
    cp = SplitConformalPredictor("diabetic_retinopathy", n_classes=5, alpha=0.05)
    cp.calibrate(cal_logits, cal_labels)
    pred_sets = cp.predict_set(test_logits)   # (B, 5) bool — True = class included
    coverage  = cp.coverage_check(test_logits, test_labels)

Requirements:
    numpy  — always required (ECE computation, conformal calibration).
    torch  — required only for LBFGS temperature fitting; numpy fallback provided.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn(
        "PyTorch not available — PerTaskTemperatureScaler.calibrate() is disabled. "
        "ECE computation and conformal prediction work without PyTorch.",
        stacklevel=2,
    )


# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """
    Serialisable calibration metrics for a single disease task.

    Attributes:
        task:            disease task name.
        temperature:     fitted temperature scalar T > 0.
        ece_before:      Expected Calibration Error before temperature scaling.
        ece_after:       Expected Calibration Error after temperature scaling.
        n_bins:          number of reliability-diagram bins used.
        n_samples:       number of calibration samples.
        is_calibrated:   True when ece_after < 0.05.
        bin_confidences: mean max-confidence per bin (reliability diagram x-axis).
        bin_accuracies:  mean accuracy per bin (reliability diagram y-axis).
        bin_counts:      sample count per bin.
    """
    task:            str
    temperature:     float
    ece_before:      float
    ece_after:       float
    n_bins:          int
    n_samples:       int
    is_calibrated:   bool
    bin_confidences: List[float] = field(default_factory=list)
    bin_accuracies:  List[float] = field(default_factory=list)
    bin_counts:      List[int]   = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task":            self.task,
            "temperature":     self.temperature,
            "ece_before":      round(self.ece_before, 6),
            "ece_after":       round(self.ece_after, 6),
            "n_bins":          self.n_bins,
            "n_samples":       self.n_samples,
            "is_calibrated":   self.is_calibrated,
            "bin_confidences": [round(c, 4) for c in self.bin_confidences],
            "bin_accuracies":  [round(a, 4) for a in self.bin_accuracies],
            "bin_counts":      self.bin_counts,
        }


# ─── ECE (pure numpy — no PyTorch dependency) ─────────────────────────────────

def _compute_ece(
    probs:  np.ndarray,   # (N, C) probability array, rows sum to 1
    labels: np.ndarray,   # (N,)   integer ground-truth labels in [0, C)
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error using equal-width confidence bins.

    Returns:
        ece:             scalar ECE in [0, 1].
        bin_confidences: (n_bins,) mean max-confidence per non-empty bin.
        bin_accuracies:  (n_bins,) mean accuracy per non-empty bin.
        bin_counts:      (n_bins,) sample count per bin (int64).
    """
    probs  = np.asarray(probs,  dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N      = len(labels)
    if N == 0:
        return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=np.int64)

    max_conf = probs.max(axis=1)         # (N,) highest predicted probability
    preds    = probs.argmax(axis=1)      # (N,) predicted class
    correct  = (preds == labels).astype(np.float64)

    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf   = np.zeros(n_bins)
    bin_acc    = np.zeros(n_bins)
    bin_cnt    = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bin_bounds[b], bin_bounds[b + 1]
        # Include upper bound in last bin to capture confidence = 1.0
        mask = (max_conf >= lo) & (max_conf < hi) if b < n_bins - 1 \
               else (max_conf >= lo) & (max_conf <= hi)
        n = mask.sum()
        if n > 0:
            bin_conf[b] = max_conf[mask].mean()
            bin_acc[b]  = correct[mask].mean()
            bin_cnt[b]  = n

    ece = float(np.sum(bin_cnt * np.abs(bin_conf - bin_acc)) / max(N, 1))
    return ece, bin_conf, bin_acc, bin_cnt


# ─── Per-Task Temperature Scaler ──────────────────────────────────────────────

class PerTaskTemperatureScaler:
    """
    Post-hoc temperature scaling calibration — one temperature T per disease task.

    Temperature scaling divides raw logits by T > 0 before softmax, which
    preserves the predicted class (argmax) while adjusting confidence levels.

        p̃_c = softmax(z / T)_c

    Fitting (LBFGS):
        Each T is optimised independently to minimise NLL on a validation set.
        A good T > 1 reduces over-confidence; T < 1 increases it.

    Requirements:
        PyTorch ≥ 2.0 required for calibrate(). ECE computation and scaling
        work with numpy only.
    """

    DEFAULT_TEMPERATURE: float = 1.5   # mild over-confidence is the common case

    def __init__(
        self,
        task_names:         List[str],
        n_classes_per_task: Dict[str, int],
        n_bins:             int = 15,
    ):
        """
        Args:
            task_names:         ordered list of disease task names.
            n_classes_per_task: {task: n_classes} mapping.
            n_bins:             number of reliability-diagram bins for ECE.
        """
        self.task_names = list(task_names)
        self.n_classes  = dict(n_classes_per_task)
        self.n_bins     = n_bins

        # One temperature per task — updated by calibrate()
        self.temperatures: Dict[str, float] = {
            t: self.DEFAULT_TEMPERATURE for t in task_names
        }
        self._reports: Dict[str, CalibrationReport] = {}

    # ── Core scaling ──────────────────────────────────────────────────────────

    def scale_logits(self, task: str, logits: "torch.Tensor") -> "torch.Tensor":
        """
        Divide raw logits by the fitted temperature for `task`.

        Args:
            task:   disease task name.
            logits: (B, C) raw logit tensor.

        Returns:
            (B, C) temperature-scaled logit tensor on the same device.
        """
        if not _TORCH:
            raise RuntimeError("PyTorch is required for scale_logits().")
        T = max(self.temperatures.get(task, 1.0), 1e-6)
        return logits / T

    def scale_logits_np(self, task: str, logits: np.ndarray) -> np.ndarray:
        """Numpy version of scale_logits — no PyTorch dependency."""
        T = max(self.temperatures.get(task, 1.0), 1e-6)
        return logits / T

    # ── LBFGS calibration ─────────────────────────────────────────────────────

    def calibrate(
        self,
        val_logits: Dict[str, "torch.Tensor"],   # {task: (N, C) logits}
        val_labels: Dict[str, "torch.Tensor"],   # {task: (N,) int labels}
        max_iter:   int   = 50,
        lr:         float = 0.01,
        tol:        float = 1e-6,
    ) -> Dict[str, CalibrationReport]:
        """
        Fit one temperature per task on the calibration (validation) set.

        Each task's temperature is fitted independently via L-BFGS minimising
        the negative log-likelihood. Pre- and post-calibration ECE are reported.

        Args:
            val_logits: {task: logits (N, C)}.
            val_labels: {task: labels (N,)} — integer class indices.
            max_iter:   L-BFGS max iterations per task.
            lr:         L-BFGS learning rate.
            tol:        L-BFGS convergence tolerance.

        Returns:
            {task: CalibrationReport} for all successfully calibrated tasks.

        Raises:
            RuntimeError: if PyTorch is not available.
        """
        if not _TORCH:
            raise RuntimeError(
                "PyTorch is required for temperature calibration. "
                "Install torch >= 2.0 and retry."
            )

        reports: Dict[str, CalibrationReport] = {}

        for task in self.task_names:
            if task not in val_logits or task not in val_labels:
                continue

            logits = val_logits[task].float().detach()    # (N, C)
            labels = val_labels[task].long().detach()     # (N,)
            N      = logits.size(0)
            if N == 0:
                continue

            # Pre-calibration ECE (raw logits → softmax)
            probs_raw     = F.softmax(logits, dim=-1).cpu().numpy()
            labels_np     = labels.cpu().numpy()
            ece_before, _, _, _ = _compute_ece(probs_raw, labels_np, self.n_bins)

            # Fit temperature via L-BFGS
            T_param   = torch.nn.Parameter(
                torch.tensor([self.DEFAULT_TEMPERATURE], dtype=torch.float32)
            )
            optimizer = torch.optim.LBFGS(
                [T_param],
                lr              = lr,
                max_iter        = max_iter,
                tolerance_change = tol,
                tolerance_grad  = tol,
            )

            def closure() -> "torch.Tensor":
                optimizer.zero_grad()
                loss = F.cross_entropy(logits / T_param.clamp(min=1e-4), labels)
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except Exception as exc:
                warnings.warn(
                    f"Temperature LBFGS failed for task '{task}': {exc}. "
                    "Keeping default temperature.",
                    stacklevel=2,
                )

            T_fitted = float(T_param.clamp(min=1e-4).detach().item())
            self.temperatures[task] = T_fitted

            # Post-calibration ECE
            probs_cal = F.softmax(logits / T_fitted, dim=-1).detach().cpu().numpy()
            ece_after, bin_conf, bin_acc, bin_cnt = _compute_ece(
                probs_cal, labels_np, self.n_bins
            )

            report = CalibrationReport(
                task            = task,
                temperature     = T_fitted,
                ece_before      = ece_before,
                ece_after       = ece_after,
                n_bins          = self.n_bins,
                n_samples       = N,
                is_calibrated   = ece_after < 0.05,
                bin_confidences = bin_conf.tolist(),
                bin_accuracies  = bin_acc.tolist(),
                bin_counts      = [int(c) for c in bin_cnt],
            )
            reports[task]        = report
            self._reports[task]  = report

        return reports

    # ── ECE helper ────────────────────────────────────────────────────────────

    def ece(
        self,
        task:   str,
        probs:  np.ndarray,           # (N, C) probability array
        labels: np.ndarray,           # (N,) integer labels
        n_bins: Optional[int] = None,
    ) -> float:
        """
        Compute ECE for calibrated probabilities.

        Args:
            task:   task name (used for logging only).
            probs:  (N, C) softmax probability array.
            labels: (N,) integer ground-truth labels.
            n_bins: override bin count (defaults to self.n_bins).

        Returns:
            ECE scalar in [0, 1].
        """
        ece, _, _, _ = _compute_ece(probs, labels, n_bins or self.n_bins)
        return ece

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save temperatures and reports to a JSON file."""
        data = {
            "temperatures": self.temperatures,
            "task_names":   self.task_names,
            "n_classes":    self.n_classes,
            "n_bins":       self.n_bins,
            "reports":      {t: r.to_dict() for t, r in self._reports.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PerTaskTemperatureScaler":
        """Load a previously saved scaler from JSON."""
        with open(path) as f:
            data = json.load(f)
        scaler = cls(
            task_names         = data["task_names"],
            n_classes_per_task = data.get("n_classes", {}),
            n_bins             = data.get("n_bins", 15),
        )
        scaler.temperatures = data["temperatures"]
        return scaler

    def summary(self) -> str:
        lines = ["PerTaskTemperatureScaler calibration summary:"]
        for task in self.task_names:
            T = self.temperatures.get(task, "not fitted")
            r = self._reports.get(task)
            if r:
                lines.append(
                    f"  {task:<30}: T={T:.4f}  "
                    f"ECE {r.ece_before:.4f} → {r.ece_after:.4f}  "
                    f"{'OK ✓' if r.is_calibrated else 'NEEDS ATTENTION'}"
                )
            else:
                lines.append(f"  {task:<30}: T={T} (not calibrated)")
        return "\n".join(lines)


# ─── Split Conformal Predictor ────────────────────────────────────────────────

class SplitConformalPredictor:
    """
    Split conformal prediction for multi-class disease classification.

    Provides a finite-sample, distribution-free marginal coverage guarantee:
        P(true_class ∈ prediction_set(X)) ≥ 1 − α

    The guarantee requires only that the calibration and test sets are i.i.d.
    (or exchangeable) — it is independent of the model architecture or training.

    Interpretation:
        prediction_set[b, c] = True  → class c cannot be ruled out for sample b.
        |prediction_set| == 1        → unambiguous, model is confident.
        |prediction_set| > 1         → ambiguous, flag for human review.

    Non-conformity score:  s_i = 1 − P(y_i | x_i)
    Threshold:             q = quantile at level ⌈(N+1)(1-α)⌉ / N of {s_i}.
    Prediction set:        {c : 1 − P(c|x) ≤ q}  ↔  {c : P(c|x) ≥ 1 − q}

    Reference:
        Angelopoulos & Bates (2022). "A Gentle Introduction to Conformal
        Prediction and Distribution-Free Uncertainty Quantification."
        arXiv:2107.07511.
    """

    def __init__(
        self,
        task:      str,
        n_classes: int,
        alpha:     float = 0.05,
    ):
        """
        Args:
            task:      disease task name (used for repr / serialisation).
            n_classes: number of classes for this task.
            alpha:     desired miscoverage rate; coverage guaranteed ≥ 1 − alpha.

        Raises:
            ValueError: if alpha is outside (0, 1).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.task      = task
        self.n_classes = n_classes
        self.alpha     = alpha
        self._threshold: Optional[float] = None   # non-conformity quantile
        self._n_cal:     Optional[int]   = None   # calibration set size

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        cal_logits: np.ndarray,   # (N, C) raw logits
        cal_labels: np.ndarray,   # (N,) integer labels in [0, C)
    ) -> float:
        """
        Compute the non-conformity score threshold from a calibration set.

        Args:
            cal_logits: (N, C) raw logits — softmax applied internally.
            cal_labels: (N,) integer true class labels.

        Returns:
            threshold: the non-conformity quantile stored for later prediction.

        Raises:
            ValueError: if the calibration set is empty or shapes are inconsistent.
        """
        cal_logits = np.asarray(cal_logits, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.int64)
        N          = len(cal_labels)

        if N == 0:
            raise ValueError("Calibration set is empty — provide at least 1 sample.")
        if cal_logits.shape[0] != N:
            raise ValueError(
                f"Shape mismatch: cal_logits has {cal_logits.shape[0]} rows "
                f"but cal_labels has {N} elements."
            )
        if cal_labels.min() < 0 or cal_labels.max() >= cal_logits.shape[1]:
            raise ValueError(
                f"cal_labels values must be in [0, {cal_logits.shape[1]}). "
                f"Got range [{cal_labels.min()}, {cal_labels.max()}]."
            )

        # Numerically stable softmax
        shifted = cal_logits - cal_logits.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=1, keepdims=True)   # (N, C)

        # Non-conformity scores: 1 − P(true class)
        true_probs = probs[np.arange(N), cal_labels]          # (N,)
        scores     = 1.0 - true_probs                         # (N,)

        # Finite-sample corrected quantile level (Theorem 1 in reference)
        q_level = math.ceil((N + 1) * (1.0 - self.alpha)) / N
        q_level = min(q_level, 1.0)   # clamp for very small N

        self._threshold = float(np.quantile(scores, q_level))
        self._n_cal     = N
        return self._threshold

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_set(self, logits: np.ndarray) -> np.ndarray:
        """
        Return a conformal prediction set for each sample.

        A class c is included when its non-conformity score ≤ threshold, i.e.
        when P(c | x) ≥ 1 − threshold.

        Args:
            logits: (B, C) raw logits or (C,) for a single sample.

        Returns:
            (B, C) bool array — True means the class is in the prediction set.

        Raises:
            RuntimeError: if calibrate() has not been called.
        """
        if self._threshold is None:
            raise RuntimeError(
                "SplitConformalPredictor is not calibrated. Call .calibrate() first."
            )
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=1, keepdims=True)      # (B, C)

        # Include class c iff P(c|x) ≥ 1 − threshold
        return probs >= (1.0 - self._threshold)                  # (B, C) bool

    # ── Empirical coverage ────────────────────────────────────────────────────

    def coverage_check(
        self,
        test_logits: np.ndarray,   # (N, C)
        test_labels: np.ndarray,   # (N,) integer labels
    ) -> float:
        """
        Compute empirical marginal coverage on a test set.

        Should be ≥ 1 − alpha when calibration and test sets are i.i.d.

        Returns:
            Empirical coverage in [0, 1].
        """
        pred_sets   = self.predict_set(test_logits)       # (N, C) bool
        test_labels = np.asarray(test_labels, dtype=np.int64)
        N           = len(test_labels)
        if N == 0:
            return 0.0
        covered = pred_sets[np.arange(N), test_labels]
        return float(covered.mean())

    def average_set_size(self, logits: np.ndarray) -> float:
        """
        Mean prediction set cardinality over a batch.

        Smaller is better (more decisive). Coverage guarantee may require
        larger sets when the model is uncertain.
        """
        pred_sets = self.predict_set(logits)
        return float(pred_sets.sum(axis=1).mean())

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "task":      self.task,
            "n_classes": self.n_classes,
            "alpha":     self.alpha,
            "threshold": self._threshold,
            "n_cal":     self._n_cal,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SplitConformalPredictor":
        obj = cls(
            task      = data["task"],
            n_classes = data["n_classes"],
            alpha     = data["alpha"],
        )
        obj._threshold = data.get("threshold")
        obj._n_cal     = data.get("n_cal")
        return obj

    def __repr__(self) -> str:
        status = (
            f"threshold={self._threshold:.4f}, n_cal={self._n_cal}"
            if self._threshold is not None
            else "not calibrated"
        )
        return (
            f"SplitConformalPredictor(task={self.task!r}, "
            f"n_classes={self.n_classes}, alpha={self.alpha}, {status})"
        )


# ─── Factory helpers ──────────────────────────────────────────────────────────

def build_scaler_for_disease_heads() -> PerTaskTemperatureScaler:
    """
    Build a PerTaskTemperatureScaler pre-configured for the standard 7-task
    MultiTaskDiseaseHead:
        DR (5 classes), glaucoma (2), AMD (4), RVO (2),
        diabetes (3), hypertension (3), cardiovascular (4).
    """
    task_names = [
        "diabetic_retinopathy",
        "glaucoma",
        "macular_degeneration",
        "retinal_vein_occlusion",
        "diabetes",
        "hypertension",
        "cardiovascular",
    ]
    n_classes = {
        "diabetic_retinopathy":   5,
        "glaucoma":               2,
        "macular_degeneration":   4,
        "retinal_vein_occlusion": 2,
        "diabetes":               3,
        "hypertension":           3,
        "cardiovascular":         4,
    }
    return PerTaskTemperatureScaler(task_names, n_classes)


def build_conformal_predictors(
    alpha: float = 0.05,
) -> Dict[str, SplitConformalPredictor]:
    """
    Build one SplitConformalPredictor per standard disease task.

    Returns:
        {task_name: SplitConformalPredictor} for the 7 standard disease tasks.
    """
    task_n_classes = {
        "diabetic_retinopathy":   5,
        "glaucoma":               2,
        "macular_degeneration":   4,
        "retinal_vein_occlusion": 2,
        "diabetes":               3,
        "hypertension":           3,
        "cardiovascular":         4,
    }
    return {
        task: SplitConformalPredictor(task, n_cls, alpha)
        for task, n_cls in task_n_classes.items()
    }
