"""
Comprehensive evaluation pipeline for RAPiD-Twin.

Metrics:
  Segmentation:   Dice, IoU, Hausdorff Distance 95th pct, Sensitivity, Specificity
  Classification: Accuracy, Precision, Recall, F1-macro/weighted, AUC-ROC (OVR),
                  Sensitivity, Specificity, per-class F1, Average Precision
  Calibration:    ECE, Brier Score
  Digital Twin:   state-space statistics, intra/inter-class distance
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                                  precision_score, recall_score,
                                  average_precision_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from rapid_twin.utils.metrics import (
    segmentation_metrics_full,
    classification_metrics_full,
    expected_calibration_error,
    brier_score,
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SegmentationMetrics:
    dice:           float = 0.0
    iou:            float = 0.0
    sensitivity:    float = 0.0
    specificity:    float = 0.0
    precision:      float = 0.0
    f1:             float = 0.0
    hausdorff_95:   float = 0.0


@dataclass
class ClassificationMetrics:
    auc_roc:          float = 0.0
    f1_macro:         float = 0.0
    f1_weighted:      float = 0.0
    precision_macro:  float = 0.0
    recall_macro:     float = 0.0
    sensitivity:      float = 0.0
    specificity:      float = 0.0
    accuracy:         float = 0.0
    average_precision: float = 0.0
    ece:              float = 0.0
    brier:            float = 0.0
    num_samples:      int   = 0
    per_class_f1:     Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    segmentation:  Optional[SegmentationMetrics]           = None
    disease_tasks: Dict[str, ClassificationMetrics]         = field(default_factory=dict)
    twin_metrics:  Dict[str, float]                         = field(default_factory=dict)
    summary:       Dict[str, float]                         = field(default_factory=dict)


# ─── Segmentation ─────────────────────────────────────────────────────────────

def compute_segmentation_metrics(
    pred:       torch.Tensor,   # (B, 1, H, W) float [0,1]
    target:     torch.Tensor,   # (B, 1, H, W) float {0,1}
    threshold:  float = 0.5,
    compute_hd: bool  = True,
) -> SegmentationMetrics:
    m = segmentation_metrics_full(
        pred.cpu().numpy(), target.cpu().numpy(),
        threshold=threshold, compute_hd=compute_hd,
    )
    return SegmentationMetrics(
        dice=m["dice"], iou=m["iou"],
        sensitivity=m["sensitivity"], specificity=m["specificity"],
        precision=m["precision"], f1=m["f1"],
        hausdorff_95=m.get("hausdorff_95", 0.0),
    )


# ─── Classification ───────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true:        np.ndarray,
    y_pred_proba:  np.ndarray,
    y_pred_label:  Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    m = classification_metrics_full(y_true, y_pred_proba, y_pred_label)

    # Calibration
    ece_val   = 0.0
    brier_val = 0.0
    if y_pred_proba.ndim == 2 and len(np.unique(y_true)) > 1:
        try:
            ece_val   = expected_calibration_error(y_true, y_pred_proba)
            brier_val = brier_score(y_true, y_pred_proba)
        except Exception:
            pass

    per_class = {k.replace("f1_class_", "class_"): v
                 for k, v in m.items() if k.startswith("f1_class_")}

    return ClassificationMetrics(
        auc_roc          = m.get("auc_roc", 0.0),
        f1_macro         = m.get("f1_macro", 0.0),
        f1_weighted      = m.get("f1_weighted", 0.0),
        precision_macro  = m.get("precision_macro", 0.0),
        recall_macro     = m.get("recall_macro", 0.0),
        sensitivity      = m.get("sensitivity", 0.0),
        specificity      = m.get("specificity", 0.0),
        accuracy         = m.get("accuracy", 0.0),
        average_precision = m.get("average_precision", 0.0),
        ece              = ece_val,
        brier            = brier_val,
        num_samples      = int(m.get("num_samples", len(y_true))),
        per_class_f1     = per_class,
    )


# ─── Evaluator ────────────────────────────────────────────────────────────────

class RAPiDTwinEvaluator:
    """
    Full evaluation pipeline for RAPiD-Twin.

    Evaluates:
      - Vessel segmentation (Dice, IoU, HD95, sensitivity, specificity)
      - Multi-task disease classification (AUC, F1, ECE, Brier)
      - Digital twin state consistency
    """

    ALL_TASKS = [
        "diabetic_retinopathy", "glaucoma", "macular_degeneration",
        "retinal_vein_occlusion", "diabetes", "hypertension", "cardiovascular",
    ]

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    # ── Segmentation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_segmentation(
        self,
        seg_model:   nn.Module,
        data_loader,
        compute_hd:  bool = True,
    ) -> SegmentationMetrics:
        seg_model.eval()
        all_pred, all_target = [], []

        for batch in data_loader:
            images = batch["image"].to(self.device)
            masks  = batch["mask"].to(self.device)
            out    = seg_model(images)
            all_pred.append(out["segmentation"].cpu())
            all_target.append(masks.cpu())

        pred   = torch.cat(all_pred,   dim=0)
        target = torch.cat(all_target, dim=0)
        m = compute_segmentation_metrics(pred, target, compute_hd=compute_hd)

        print(f"\n{'='*54}\nSegmentation Metrics")
        print(f"  Dice:          {m.dice:.4f}")
        print(f"  IoU:           {m.iou:.4f}")
        print(f"  Sensitivity:   {m.sensitivity:.4f}")
        print(f"  Specificity:   {m.specificity:.4f}")
        print(f"  Precision:     {m.precision:.4f}")
        if compute_hd:
            print(f"  HD95 (px):     {m.hausdorff_95:.2f}")
        return m

    # ── Disease classification ─────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_disease_prediction(
        self, data_loader
    ) -> Dict[str, ClassificationMetrics]:
        self.model.eval()
        collectors = defaultdict(lambda: {"proba": [], "pred": [], "true": []})

        for batch in data_loader:
            image  = batch["image"].to(self.device)
            physio = batch.get("physio_signals")
            if physio is not None:
                physio = physio.to(self.device)

            outputs = self.model(retinal_image=image, physio_signals=physio)
            preds   = outputs["disease_predictions"]
            labels  = batch.get("labels", {})

            for task in self.ALL_TASKS:
                if task not in preds or task not in labels:
                    continue
                logits     = preds[task]
                proba      = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_label = logits.argmax(dim=-1).cpu().numpy()
                true_label = labels[task].numpy() if isinstance(labels[task], torch.Tensor) \
                             else np.array(labels[task])

                collectors[task]["proba"].extend(proba.tolist())
                collectors[task]["pred"].extend(pred_label.tolist())
                collectors[task]["true"].extend(true_label.tolist())

        task_metrics: Dict[str, ClassificationMetrics] = {}
        print(f"\n{'='*74}\nDisease Prediction Metrics")
        hdr = f"{'Task':<30} {'AUC':>6} {'F1-W':>6} {'Sens':>6} {'Spec':>6} {'ECE':>6} {'N':>5}"
        print(hdr)
        print("-" * 74)

        for task in self.ALL_TASKS:
            if not collectors[task]["true"]:
                continue
            y_true  = np.array(collectors[task]["true"])
            y_proba = np.array(collectors[task]["proba"])
            y_pred  = np.array(collectors[task]["pred"])
            m = compute_classification_metrics(y_true, y_proba, y_pred)
            task_metrics[task] = m
            print(f"  {task:<28} {m.auc_roc:>6.4f} {m.f1_weighted:>6.4f} "
                  f"{m.sensitivity:>6.4f} {m.specificity:>6.4f} "
                  f"{m.ece:>6.4f} {m.num_samples:>5}")

        if task_metrics:
            mean_auc = np.mean([m.auc_roc for m in task_metrics.values()])
            mean_f1  = np.mean([m.f1_weighted for m in task_metrics.values()])
            mean_ece = np.mean([m.ece for m in task_metrics.values()])
            print("-" * 74)
            print(f"  {'MEAN':<28} {mean_auc:>6.4f} {mean_f1:>6.4f} "
                  f"{'':>6} {'':>6} {mean_ece:>6.4f}")

        return task_metrics

    # ── Twin consistency ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_twin_consistency(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        states: List[np.ndarray] = []
        labels_collected = defaultdict(list)

        for batch in data_loader:
            image   = batch["image"].to(self.device)
            outputs = self.model(retinal_image=image)
            state   = outputs["updated_state"].cpu().numpy()
            states.extend(state.tolist())
            for task, lbl in batch.get("labels", {}).items():
                if isinstance(lbl, torch.Tensor):
                    labels_collected[task].extend(lbl.numpy().tolist())

        states_arr = np.array(states)
        metrics    = {
            "state_mean_norm": float(np.linalg.norm(states_arr, axis=1).mean()),
            "state_std":       float(states_arr.std(axis=0).mean()),
            "num_evaluated":   len(states_arr),
        }

        if "diabetic_retinopathy" in labels_collected:
            dr_labels = np.array(labels_collected["diabetic_retinopathy"])
            if len(dr_labels) == len(states_arr):
                normal = states_arr[dr_labels == 0]
                dr_pos = states_arr[dr_labels > 0]
                if len(normal) > 1 and len(dr_pos) > 1:
                    metrics["intra_class_dist_normal"] = float(
                        np.linalg.norm(normal - normal.mean(0), axis=1).mean())
                    metrics["inter_class_dist_dr"] = float(
                        np.linalg.norm(normal.mean(0) - dr_pos.mean(0)))

        print(f"\n{'='*50}\nDigital Twin State Metrics")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return metrics

    # ── Full report ───────────────────────────────────────────────────────────

    def full_evaluation(
        self,
        data_loader,
        seg_model:   Optional[nn.Module] = None,
        seg_loader   = None,
        compute_hd:  bool = True,
    ) -> EvaluationReport:
        report = EvaluationReport()

        if seg_model is not None and seg_loader is not None:
            report.segmentation = self.evaluate_segmentation(
                seg_model, seg_loader, compute_hd=compute_hd)

        report.disease_tasks = self.evaluate_disease_prediction(data_loader)
        report.twin_metrics  = self.evaluate_twin_consistency(data_loader)

        if report.disease_tasks:
            report.summary["mean_auc"]   = float(np.mean(
                [m.auc_roc for m in report.disease_tasks.values()]))
            report.summary["mean_f1"]    = float(np.mean(
                [m.f1_weighted for m in report.disease_tasks.values()]))
            report.summary["mean_ece"]   = float(np.mean(
                [m.ece for m in report.disease_tasks.values()]))
            report.summary["mean_brier"] = float(np.mean(
                [m.brier for m in report.disease_tasks.values()]))
            for task, m in report.disease_tasks.items():
                report.summary[f"auc_{task}"] = m.auc_roc
                report.summary[f"f1_{task}"]  = m.f1_weighted

        if report.segmentation:
            report.summary["seg_dice"]         = report.segmentation.dice
            report.summary["seg_iou"]          = report.segmentation.iou
            report.summary["seg_hausdorff_95"] = report.segmentation.hausdorff_95

        return report
