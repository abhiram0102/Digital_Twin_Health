#!/usr/bin/env python3
"""
Evaluation entry point for RAPiD-Twin.
Usage: python -m rapid_twin.evaluate --checkpoint checkpoints/rapid_twin/best_rapid_twin.pt
"""

import argparse
import torch
import json
from pathlib import Path

from rapid_twin.configs.config import DEFAULT_CONFIG
from rapid_twin.models.rapid_twin_model import RAPiDTwin
from rapid_twin.datasets.retinal_datasets import get_synthetic_loaders
from rapid_twin.evaluation.evaluator import RAPiDTwinEvaluator
from rapid_twin.utils.metrics import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    set_seed(cfg.training.seed)

    model = RAPiDTwin(cfg)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint provided — evaluating with random weights")

    _, val_loader = get_synthetic_loaders(
        batch_size=args.batch_size, image_size=args.image_size, num_workers=0
    )

    evaluator = RAPiDTwinEvaluator(model, device=device)
    report = evaluator.full_evaluation(val_loader)

    # Serialize report
    report_dict = {"summary": report.summary}
    if report.disease_tasks:
        report_dict["disease_tasks"] = {
            k: {f: getattr(v, f) for f in v.__dataclass_fields__}
            for k, v in report.disease_tasks.items()
        }
    if report.twin_metrics:
        report_dict["twin_metrics"] = report.twin_metrics
    if report.segmentation:
        report_dict["segmentation"] = {
            f: getattr(report.segmentation, f)
            for f in report.segmentation.__dataclass_fields__
        }

    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nReport saved to {args.output}")
    print("\nSummary:")
    for k, v in report.summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
