"""Visualization utilities for retinal images, graphs, and health states."""

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Dict, List, Optional


def segmentation_overlay(image: np.ndarray, mask: np.ndarray,
                          pred_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay ground truth and predicted vessel masks on image."""
    import cv2
    overlay = image.copy()
    # GT in green, pred in red, overlap in yellow
    gt = (mask > 0.5)
    pr = (pred_mask > 0.5)
    overlay[gt & ~pr] = [0, 200, 0]      # True positive green
    overlay[pr & ~gt] = [200, 0, 0]      # False positive red
    overlay[gt & pr] = [200, 200, 0]     # Overlap yellow
    return (alpha * overlay + (1 - alpha) * image).astype(np.uint8)


def draw_vascular_graph(image: np.ndarray, graph, scale: float = 1.0) -> np.ndarray:
    """Draw vascular graph nodes and edges on image."""
    import cv2
    vis = image.copy()
    h, w = vis.shape[:2]

    for edge in graph.edges[:50]:  # limit to 50 edges
        src_node = graph.nodes[edge.src]
        dst_node = graph.nodes[edge.dst]
        pt1 = (int(src_node.position[1] * scale), int(src_node.position[0] * scale))
        pt2 = (int(dst_node.position[1] * scale), int(dst_node.position[0] * scale))
        pt1 = (max(0, min(pt1[0], w-1)), max(0, min(pt1[1], h-1)))
        pt2 = (max(0, min(pt2[0], w-1)), max(0, min(pt2[1], h-1)))
        cv2.line(vis, pt1, pt2, (0, 255, 255), 1)

    for node in graph.nodes[:100]:
        pt = (int(node.position[1] * scale), int(node.position[0] * scale))
        pt = (max(0, min(pt[0], w-1)), max(0, min(pt[1], h-1)))
        color = (255, 0, 0) if node.is_bifurcation else (0, 0, 255)
        cv2.circle(vis, pt, 3, color, -1)

    return vis


def health_state_radar(health_scores: Dict[str, float]) -> Dict:
    """Return data for radar/spider chart of health dimensions."""
    dims = ["vascular_health", "retinal_integrity", "metabolic_risk",
            "cardiovascular_risk", "lifestyle_health"]
    values = [health_scores.get(d, 0.5) for d in dims]
    return {"dimensions": dims, "values": values}


def format_prediction_report(outputs: Dict) -> str:
    """Format model outputs as a human-readable report."""
    lines = ["=" * 60, "RAPiD-Twin Health Report", "=" * 60]

    # Anomaly score
    if outputs.get("anomaly_score") is not None:
        score = outputs["anomaly_score"]
        if isinstance(score, torch.Tensor):
            score = score.item()
        risk = "HIGH" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
        lines.append(f"\nHealth Deviation Score: {score:.3f} [{risk}]")

    # Disease predictions
    if outputs.get("disease_predictions"):
        lines.append("\nDisease Risk Predictions:")
        task_labels = {
            "diabetic_retinopathy": "Diabetic Retinopathy",
            "glaucoma": "Glaucoma",
            "macular_degeneration": "Macular Degeneration",
            "retinal_vein_occlusion": "Retinal Vein Occlusion",
            "diabetes": "Diabetes Risk",
            "hypertension": "Hypertension Risk",
            "cardiovascular": "Cardiovascular Risk"
        }
        for task, logits in outputs["disease_predictions"].items():
            if isinstance(logits, torch.Tensor):
                proba = torch.softmax(logits, dim=-1)
                max_prob = proba.max().item()
                pred_cls = proba.argmax().item()
                display = task_labels.get(task, task)
                lines.append(f"  {display:<32}: Class {pred_cls} (p={max_prob:.3f})")

    # Health scores
    if outputs.get("health_scores"):
        lines.append("\nHealth State Dimensions:")
        for dim, score in outputs["health_scores"].items():
            if isinstance(score, torch.Tensor) and score.shape[-1] == 1:
                lines.append(f"  {dim:<28}: {score.item():.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
