"""
Clinical Reliability Layer — Uncertainty Estimation, Calibration, OOD Detection.

Provides the infrastructure needed for a medically trustworthy AI system:

  1. MCDropoutPredictor     — Monte Carlo Dropout inference (mean, variance, entropy)
  2. CalibrationAnalyzer    — Expected Calibration Error (ECE), reliability diagrams
  3. TemperatureScaler      — post-hoc logit calibration via temperature scaling
  4. OODDetector            — Mahalanobis-distance and energy-based OOD scoring
  5. ConfidenceAwareBiomarkers — propagate model uncertainty into biomarker CIs

Design contracts:
  • MCDropoutPredictor requires model with Dropout layers.  Dropout is forced
    to train mode during inference; all other BatchNorm layers stay in eval mode.
  • CalibrationAnalyzer never requires retraining — operates on saved logits/probs.
  • TemperatureScaler is fit on a calibration set (val split) and then applied
    at inference time.
  • OODDetector stores training-set class statistics; never touches model weights.
  • All outputs carry an explicit `is_reliable` flag when uncertainty is too high.
  • No silent NaN propagation — every edge case is explicitly handled.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("PyTorch not available — MCDropoutPredictor and TemperatureScaler disabled.")


# ─── Uncertainty output types ─────────────────────────────────────────────────

@dataclass
class PredictionUncertainty:
    """
    Aggregated uncertainty over MC samples.

    Attributes:
        mean_probs:        (N, C) mean class probabilities over MC samples.
        variance:          (N, C) variance of class probabilities.
        predictive_entropy: (N,) total uncertainty.
        mutual_information: (N,) epistemic uncertainty (model disagreement).
        aleatoric:         (N,) aleatoric uncertainty = mean_entropy - MI.
        is_reliable:       (N,) bool — True when max_prob > reliability_threshold.
        n_samples:         number of MC samples used.
    """
    mean_probs:          np.ndarray
    variance:            np.ndarray
    predictive_entropy:  np.ndarray
    mutual_information:  np.ndarray
    aleatoric:           np.ndarray
    is_reliable:         np.ndarray
    n_samples:           int


@dataclass
class OODScore:
    """Per-sample OOD detection result."""
    mahalanobis:    np.ndarray   # higher = more OOD
    energy:         np.ndarray   # higher = more OOD (negative log-partition)
    is_ood:         np.ndarray   # bool, threshold-based
    threshold_maha: float
    threshold_energy: float


@dataclass
class CalibrationResult:
    """Output of calibration analysis."""
    ece:             float           # Expected Calibration Error in [0, 1]
    mce:             float           # Maximum Calibration Error
    ace:             float           # Average Calibration Error
    bin_confidences: np.ndarray      # (n_bins,) mean confidence per bin
    bin_accuracies:  np.ndarray      # (n_bins,) mean accuracy per bin
    bin_counts:      np.ndarray      # (n_bins,) samples per bin
    n_bins:          int
    is_well_calibrated: bool         # ECE < 0.05


# ─── Utility functions ────────────────────────────────────────────────────────

def compute_prediction_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Shannon entropy of a probability distribution.

    Args:
        probs: (N, C) probability array.  Must sum to 1 along axis=1.
        eps:   small constant to avoid log(0).

    Returns:
        (N,) entropy in nats.
    """
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1).astype(np.float32)


def compute_mutual_information(
    sample_probs: np.ndarray,
    mean_probs:   np.ndarray,
    eps:          float = 1e-8,
) -> np.ndarray:
    """
    Mutual information between model parameters and predictions (epistemic uncertainty).

    MI = H[E_θ[p(y|x,θ)]] - E_θ[H[p(y|x,θ)]]
       = H(mean_probs) - mean(H(sample_probs))

    Args:
        sample_probs: (S, N, C) probabilities per MC sample.
        mean_probs:   (N, C) mean over samples.

    Returns:
        (N,) mutual information in nats.
    """
    H_mean = compute_prediction_entropy(mean_probs, eps)           # (N,)
    S = sample_probs.shape[0]
    H_samples = np.stack(
        [compute_prediction_entropy(sample_probs[s], eps) for s in range(S)],
        axis=0,
    )                                                               # (S, N)
    mean_H = H_samples.mean(axis=0)                                # (N,)
    return np.clip(H_mean - mean_H, 0.0, None).astype(np.float32)


# ─── MC Dropout Predictor ─────────────────────────────────────────────────────

if _TORCH:
    class MCDropoutPredictor:
        """
        Monte Carlo Dropout uncertainty estimator.

        Forces all Dropout layers into train mode at inference time (all other
        layers remain in eval mode).  Runs the model `n_samples` times and
        aggregates statistics.

        Compatible with any nn.Module that has Dropout layers and outputs logits.

        Usage:
            predictor = MCDropoutPredictor(model, n_samples=20, device="cuda")
            uncertainty = predictor.predict(inputs)
            print(uncertainty.predictive_entropy)
        """

        def __init__(
            self,
            model:               "nn.Module",
            n_samples:           int   = 20,
            device:              str   = "cpu",
            reliability_threshold: float = 0.7,
        ):
            if n_samples < 2:
                raise ValueError(f"n_samples must be >= 2, got {n_samples}")
            self.model                 = model
            self.n_samples             = n_samples
            self.device                = device
            self.reliability_threshold = reliability_threshold

        def _enable_dropout(self):
            """Set Dropout layers to train mode, everything else to eval."""
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.train()

        def predict(
            self,
            *inputs,
            softmax_dim: int = -1,
        ) -> PredictionUncertainty:
            """
            Run MC Dropout inference.

            Args:
                *inputs:      model inputs (forwarded as-is).
                softmax_dim:  dimension along which to apply softmax to get probs.

            Returns:
                PredictionUncertainty with mean, variance, entropy, MI.
            """
            self._enable_dropout()
            sample_logits = []

            with torch.no_grad():
                for _ in range(self.n_samples):
                    out = self.model(*[
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    ])
                    # Support models that return (logits, ...) tuples
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    sample_logits.append(
                        F.softmax(out.float(), dim=softmax_dim).cpu().numpy()
                    )

            sample_probs = np.stack(sample_logits, axis=0)  # (S, N, C)
            mean_probs   = sample_probs.mean(axis=0)         # (N, C)
            variance     = sample_probs.var(axis=0)          # (N, C)

            entropy = compute_prediction_entropy(mean_probs)
            mi      = compute_mutual_information(sample_probs, mean_probs)
            aleat   = np.clip(entropy - mi, 0.0, None)

            max_prob     = mean_probs.max(axis=-1)
            is_reliable  = max_prob > self.reliability_threshold

            return PredictionUncertainty(
                mean_probs=mean_probs.astype(np.float32),
                variance=variance.astype(np.float32),
                predictive_entropy=entropy.astype(np.float32),
                mutual_information=mi.astype(np.float32),
                aleatoric=aleat.astype(np.float32),
                is_reliable=is_reliable,
                n_samples=self.n_samples,
            )

else:
    class MCDropoutPredictor:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required for MCDropoutPredictor.")


# ─── Calibration Analyzer ─────────────────────────────────────────────────────

class CalibrationAnalyzer:
    """
    Post-hoc calibration analysis.  Computes ECE, MCE, ACE and reliability
    diagram data without modifying the model.

    Usage:
        analyzer = CalibrationAnalyzer(n_bins=15)
        result = analyzer.compute(probs, labels)
        analyzer.print_report(result)
    """

    def __init__(self, n_bins: int = 15, ece_threshold: float = 0.05):
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_bins        = n_bins
        self.ece_threshold = ece_threshold

    def compute(
        self,
        probs:  np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationResult:
        """
        Compute calibration metrics from model outputs.

        Args:
            probs:  (N, C) softmax probabilities, or (N,) for binary.
            labels: (N,) integer class labels in [0, C-1].

        Returns:
            CalibrationResult.
        """
        probs  = np.asarray(probs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if probs.ndim == 1:
            # Binary: treat probs as P(class=1)
            max_prob = probs
            correct  = (probs > 0.5).astype(np.float32) == labels.astype(np.float32)
        elif probs.ndim == 2:
            max_prob = probs.max(axis=1)
            preds    = probs.argmax(axis=1)
            correct  = (preds == labels).astype(np.float32)
        else:
            raise ValueError(f"probs must be 1-D or 2-D, got shape {probs.shape}")

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_confidences = np.zeros(self.n_bins, dtype=np.float32)
        bin_accuracies  = np.zeros(self.n_bins, dtype=np.float32)
        bin_counts      = np.zeros(self.n_bins, dtype=np.int64)

        for b in range(self.n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask   = (max_prob >= lo) & (max_prob < hi)
            if b == self.n_bins - 1:
                mask = (max_prob >= lo) & (max_prob <= hi)
            n = mask.sum()
            bin_counts[b] = int(n)
            if n > 0:
                bin_confidences[b] = float(max_prob[mask].mean())
                bin_accuracies[b]  = float(correct[mask].mean())

        N   = len(labels)
        ece = float(np.sum(
            bin_counts * np.abs(bin_confidences - bin_accuracies)
        ) / max(N, 1))
        mce = float(np.max(np.abs(bin_confidences - bin_accuracies)))
        # ACE: unweighted mean over non-empty bins
        nonempty = bin_counts > 0
        ace = float(
            np.abs(bin_confidences[nonempty] - bin_accuracies[nonempty]).mean()
        ) if nonempty.any() else float("nan")

        return CalibrationResult(
            ece=ece,
            mce=mce,
            ace=ace,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
            is_well_calibrated=ece < self.ece_threshold,
        )

    def print_report(self, result: CalibrationResult):
        print(f"\n{'─'*55}")
        print("  Calibration Analysis")
        print(f"{'─'*55}")
        print(f"  ECE  (Expected Calibration Error): {result.ece:.4f}"
              f"  {'✓ WELL CALIBRATED' if result.is_well_calibrated else '✗ NEEDS CALIBRATION'}")
        print(f"  MCE  (Maximum  Calibration Error): {result.mce:.4f}")
        print(f"  ACE  (Average  Calibration Error): {result.ace:.4f}")
        print(f"\n  Reliability Diagram ({result.n_bins} bins):")
        print(f"  {'Bin Centre':>12}  {'Confidence':>12}  {'Accuracy':>12}  {'Count':>8}")
        for b in range(result.n_bins):
            if result.bin_counts[b] == 0:
                continue
            centre = (b + 0.5) / result.n_bins
            print(
                f"  {centre:>12.3f}  "
                f"{result.bin_confidences[b]:>12.4f}  "
                f"{result.bin_accuracies[b]:>12.4f}  "
                f"{result.bin_counts[b]:>8d}"
            )
        print(f"{'─'*55}\n")


# ─── Temperature Scaling ──────────────────────────────────────────────────────

if _TORCH:
    class TemperatureScaler(nn.Module):
        """
        Post-hoc calibration via temperature scaling (Guo et al. 2017).

        Single scalar T > 0 divides all logits before softmax.
        T > 1 softens predictions (reduces overconfidence).
        T < 1 sharpens predictions.

        Fit on a held-out calibration set; then use at inference time.

        Usage:
            scaler = TemperatureScaler()
            scaler.fit(val_logits, val_labels)
            calibrated_probs = scaler.calibrate(test_logits)
        """

        def __init__(self, init_temperature: float = 1.5):
            super().__init__()
            if init_temperature <= 0:
                raise ValueError("Temperature must be > 0.")
            self.temperature = nn.Parameter(
                torch.tensor([init_temperature], dtype=torch.float32)
            )

        def forward(self, logits: "torch.Tensor") -> "torch.Tensor":
            """Divide logits by temperature."""
            return logits / self.temperature.clamp(min=1e-4)

        def calibrate(self, logits: "torch.Tensor") -> "torch.Tensor":
            """Return softmax probabilities after temperature scaling."""
            return F.softmax(self.forward(logits), dim=-1)

        def fit(
            self,
            logits: "torch.Tensor",
            labels: "torch.Tensor",
            lr:     float = 0.01,
            max_iter: int = 200,
        ) -> float:
            """
            Fit temperature to minimise NLL on the calibration set.

            Args:
                logits: (N, C) raw logits on calibration set.
                labels: (N,) integer labels.
                lr:     learning rate for L-BFGS.
                max_iter: maximum L-BFGS iterations.

            Returns:
                Optimal temperature found.
            """
            logits = logits.detach()
            labels = labels.detach().long()

            optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
            nll = nn.CrossEntropyLoss()

            def _closure():
                optimizer.zero_grad()
                loss = nll(self.forward(logits), labels)
                loss.backward()
                return loss

            optimizer.step(_closure)

            T = float(self.temperature.clamp(min=1e-4).item())
            return T

        @property
        def T(self) -> float:
            """Current temperature value."""
            return float(self.temperature.clamp(min=1e-4).item())

else:
    class TemperatureScaler:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required for TemperatureScaler.")


# ─── OOD Detector ─────────────────────────────────────────────────────────────

class OODDetector:
    """
    Out-of-Distribution detection using Mahalanobis distance (Lee et al. 2018)
    and energy-based scoring (Liu et al. 2020).

    Fit on in-distribution feature representations.  At inference time, score
    new samples — high scores indicate OOD samples.

    Usage:
        detector = OODDetector()
        detector.fit(train_features, train_labels)
        scores = detector.score(test_features, test_logits)
    """

    def __init__(
        self,
        mahalanobis_percentile: float = 95.0,
        energy_percentile:      float = 95.0,
        energy_temperature:     float = 1.0,
    ):
        self.mahalanobis_percentile = mahalanobis_percentile
        self.energy_percentile      = energy_percentile
        self.energy_temperature     = energy_temperature

        self._class_means:     Optional[np.ndarray] = None
        self._precision:       Optional[np.ndarray] = None
        self._threshold_maha:  float = float("inf")
        self._threshold_energy: float = float("inf")
        self._is_fitted:       bool = False

    def fit(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
    ) -> "OODDetector":
        """
        Compute class-conditional Gaussian statistics from training features.

        Args:
            features: (N, D) feature representations.
            labels:   (N,) integer class labels.

        Returns:
            self (for chaining).
        """
        features = np.asarray(features, dtype=np.float64)
        labels   = np.asarray(labels,   dtype=np.int64)
        classes  = np.unique(labels)
        D        = features.shape[1]

        class_means = np.zeros((len(classes), D), dtype=np.float64)
        for i, cls in enumerate(classes):
            class_means[i] = features[labels == cls].mean(axis=0)

        # Shared covariance (tied Gaussian)
        centered = features - class_means[labels]
        cov = (centered.T @ centered) / max(len(features) - 1, 1)
        # Regularise
        cov += np.eye(D) * 1e-6

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            warnings.warn("OODDetector: covariance matrix singular — using identity precision.")
            precision = np.eye(D)

        self._class_means = class_means
        self._precision   = precision

        # Fit thresholds from training distribution
        train_scores = self._mahalanobis_scores(features)
        self._threshold_maha = float(np.percentile(
            train_scores, self.mahalanobis_percentile
        ))
        self._is_fitted = True
        return self

    def _mahalanobis_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance to any class mean."""
        if self._class_means is None or self._precision is None:
            raise RuntimeError("OODDetector not fitted. Call .fit() first.")

        features = np.asarray(features, dtype=np.float64)
        N = features.shape[0]
        K = self._class_means.shape[0]
        min_dists = np.full(N, np.inf)

        for k in range(K):
            diff = features - self._class_means[k]              # (N, D)
            dists = np.einsum("nd,dd,nd->n", diff, self._precision, diff)
            np.minimum(min_dists, dists, out=min_dists)

        return np.sqrt(np.clip(min_dists, 0.0, None)).astype(np.float32)

    @staticmethod
    def _energy_scores(
        logits:      np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Energy-based OOD score: -T * log Σ exp(logits / T).
        Higher energy = more OOD.
        """
        scaled   = logits / temperature
        log_part = np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1))
        log_part += scaled.max(axis=1)
        return (-temperature * log_part).astype(np.float32)

    def score(
        self,
        features: np.ndarray,
        logits:   Optional[np.ndarray] = None,
    ) -> OODScore:
        """
        Compute OOD scores for a batch of samples.

        Args:
            features: (N, D) feature representations.
            logits:   (N, C) raw logits (optional; required for energy score).

        Returns:
            OODScore with Mahalanobis and energy scores.
        """
        if not self._is_fitted:
            raise RuntimeError("OODDetector not fitted. Call .fit() first.")

        features = np.asarray(features, dtype=np.float64)
        maha     = self._mahalanobis_scores(features)

        if logits is not None:
            logits_np = np.asarray(logits, dtype=np.float64)
            energy    = self._energy_scores(logits_np, self.energy_temperature)
            threshold_e = self._threshold_energy
        else:
            energy      = np.full(len(features), float("nan"), dtype=np.float32)
            threshold_e = float("inf")

        is_ood = maha > self._threshold_maha

        return OODScore(
            mahalanobis=maha,
            energy=energy,
            is_ood=is_ood,
            threshold_maha=self._threshold_maha,
            threshold_energy=threshold_e,
        )


# ─── Confidence-Aware Biomarkers ──────────────────────────────────────────────

@dataclass
class BiomarkerWithCI:
    """A single biomarker value with uncertainty-derived confidence interval."""
    name:            str
    value:           float
    ci_lower:        float    # lower bound of 95% CI (nan if not available)
    ci_upper:        float
    is_reliable:     bool     # False if underlying model confidence is low
    uncertainty:     float    # propagated uncertainty measure


class ConfidenceAwareBiomarkers:
    """
    Propagate model uncertainty into biomarker confidence intervals.

    Given a BiomarkerReport and per-node model uncertainty (from MCDropoutPredictor),
    adjusts biomarker estimates with uncertainty-derived CIs.

    Uncertainty propagation strategy:
      - Biomarkers that aggregate over N nodes have CI width ∝ mean_uncertainty / √N.
      - Biomarkers that depend on AV classification use AV confidence directly.
      - A biomarker is flagged unreliable if mean uncertainty > unreliable_threshold.

    Usage:
        cabi = ConfidenceAwareBiomarkers(unreliable_threshold=0.5)
        results = cabi.annotate(biomarker_report, node_uncertainty)
    """

    def __init__(
        self,
        unreliable_threshold: float = 0.5,
        ci_coverage:          float = 0.95,
        z_score:              float = 1.96,   # z for 95% CI
    ):
        self.unreliable_threshold = unreliable_threshold
        self.ci_coverage          = ci_coverage
        self.z_score              = z_score

    def annotate(
        self,
        biomarker_report,
        node_uncertainty:   np.ndarray,      # (N,) predictive entropy per node
        av_confidence:      Optional[np.ndarray] = None,  # (N,) AV confidence
    ) -> List[BiomarkerWithCI]:
        """
        Annotate all scalar biomarkers with confidence intervals.

        Args:
            biomarker_report: BiomarkerReport from RetinalBiomarkerExtractor.
            node_uncertainty: (N,) predictive entropy per node (from MC Dropout).
            av_confidence:    (N,) AV classifier confidence (optional).

        Returns:
            List of BiomarkerWithCI — one per scalar biomarker.
        """
        node_uncertainty = np.asarray(node_uncertainty, dtype=np.float32)
        N                = max(len(node_uncertainty), 1)
        mean_unc         = float(node_uncertainty.mean()) if len(node_uncertainty) > 0 else 0.5
        sem              = mean_unc / (N ** 0.5 + 1e-8)  # standard error of mean

        is_reliable_global = mean_unc < self.unreliable_threshold

        results: List[BiomarkerWithCI] = []
        flat = biomarker_report.to_dict()

        # AV-dependent biomarkers use AV confidence as additional uncertainty
        av_unc_keys = {"crae", "crve", "avr", "av_nicking_index", "av_crossing_score"}
        if av_confidence is not None:
            mean_av_conf  = float(np.asarray(av_confidence).mean())
            av_uncertainty = 1.0 - mean_av_conf  # higher conf = lower uncertainty
        else:
            av_uncertainty = 0.5  # neutral

        for name, value in flat.items():
            import math
            if not math.isfinite(value):
                results.append(BiomarkerWithCI(
                    name=name, value=value,
                    ci_lower=float("nan"), ci_upper=float("nan"),
                    is_reliable=False, uncertainty=float("nan"),
                ))
                continue

            # Choose uncertainty source
            if name in av_unc_keys:
                unc = (sem + av_uncertainty) / 2.0
            else:
                unc = sem

            half_width = self.z_score * unc * abs(value + 1e-8)

            results.append(BiomarkerWithCI(
                name=name,
                value=value,
                ci_lower=value - half_width,
                ci_upper=value + half_width,
                is_reliable=is_reliable_global and (name not in av_unc_keys or av_uncertainty < 0.5),
                uncertainty=float(unc),
            ))

        return results
