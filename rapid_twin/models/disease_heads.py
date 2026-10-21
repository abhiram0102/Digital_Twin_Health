"""
Multi-Task Disease Prediction Heads.
Eye diseases: diabetic retinopathy (5-class), glaucoma (2), macular degeneration (4), RVO (2).
Systemic risks: diabetes (3), hypertension (3), cardiovascular (4).
Uses shared + task-specific layers for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class TaskHead(nn.Module):
    """Single disease prediction head."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskDiseaseHead(nn.Module):
    """
    Multi-task disease prediction module.
    Shared encoder → task-specific heads.
    Supports both eye diseases and systemic disease risk.
    """

    EYE_TASKS = ["diabetic_retinopathy", "glaucoma", "macular_degeneration", "retinal_vein_occlusion"]
    EYE_CLASSES = [5, 2, 4, 2]
    SYSTEMIC_TASKS = ["diabetes", "hypertension", "cardiovascular"]
    SYSTEMIC_CLASSES = [3, 3, 4]

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        # Eye disease heads
        self.eye_heads = nn.ModuleDict({
            name: TaskHead(hidden_dim, hidden_dim // 2, n_cls, dropout)
            for name, n_cls in zip(self.EYE_TASKS, self.EYE_CLASSES)
        })

        # Systemic risk heads
        self.systemic_heads = nn.ModuleDict({
            name: TaskHead(hidden_dim, hidden_dim // 2, n_cls, dropout)
            for name, n_cls in zip(self.SYSTEMIC_TASKS, self.SYSTEMIC_CLASSES)
        })

        # Severity / confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        embedding: (B, input_dim)
        Returns dict of {task_name: logits (B, num_classes)}
        """
        shared = self.shared(embedding)  # (B, hidden_dim)

        predictions = {}
        for name, head in self.eye_heads.items():
            predictions[name] = head(shared) / self.temperature

        for name, head in self.systemic_heads.items():
            predictions[name] = head(shared) / self.temperature

        return predictions

    def predict_proba(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns softmax probabilities for all tasks."""
        logits = self.forward(embedding)
        return {k: F.softmax(v, dim=-1) for k, v in logits.items()}


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in disease prediction."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), targets: (B,) long"""
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with uncertainty-weighted task balancing.
    Uses learned log-variance for automatic task weighting (Kendall et al.).
    """

    ALL_TASKS = (MultiTaskDiseaseHead.EYE_TASKS + MultiTaskDiseaseHead.SYSTEMIC_TASKS)

    def __init__(self, num_tasks: int = 7, gamma: float = 2.0,
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.use_uncertainty = use_uncertainty_weighting

        # Learnable log-variance per task for uncertainty weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        self.focal = FocalLoss(gamma=gamma)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        predictions: {task_name: logits (B, C)}
        targets: {task_name: labels (B,)}
        Returns dict with 'total', 'per_task', 'weights'
        """
        task_losses = []
        task_names = []

        for task in self.ALL_TASKS:
            if task in predictions and task in targets and targets[task] is not None:
                loss = self.focal(predictions[task], targets[task])
                task_losses.append(loss)
                task_names.append(task)

        if not task_losses:
            return {"total": torch.tensor(0.0), "per_task": {}, "weights": []}

        task_losses_t = torch.stack(task_losses)

        if self.use_uncertainty and len(task_losses) <= self.num_tasks:
            log_vars = self.log_vars[:len(task_losses)]
            precision = torch.exp(-log_vars)
            weighted = precision * task_losses_t + log_vars
            total = weighted.sum()
            weights = precision.detach().tolist()
        else:
            total = task_losses_t.mean()
            weights = [1.0] * len(task_losses)

        return {
            "total": total,
            "per_task": dict(zip(task_names, task_losses)),
            "weights": weights
        }
