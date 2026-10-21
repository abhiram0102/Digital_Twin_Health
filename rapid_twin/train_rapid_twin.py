#!/usr/bin/env python3
"""
Entry point: Train full RAPiD-Twin model end-to-end.
Usage: python -m rapid_twin.train_rapid_twin [--synthetic] [--epochs N]
"""

import argparse
import torch

from rapid_twin.configs.config import DEFAULT_CONFIG
from rapid_twin.models.rapid_twin_model import RAPiDTwin
from rapid_twin.datasets.retinal_datasets import get_synthetic_loaders
from rapid_twin.training.trainer import RAPiDTwinTrainer
from rapid_twin.utils.metrics import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train RAPiD-Twin Full Model")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.training.checkpoint_dir = args.checkpoint_dir
    cfg.training.log_dir = args.log_dir
    cfg.training.main_batch_size = args.batch_size
    if args.epochs:
        cfg.training.main_epochs = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    set_seed(cfg.training.seed)

    print(f"\nRAPiD-Twin Full Model Training")
    print(f"Device: {device} | Batch size: {args.batch_size}")

    model = RAPiDTwin(cfg)
    counts = model.count_parameters()
    print(f"\nParameter counts:")
    for k, v in counts.items():
        print(f"  {k:<25}: {v:>10,}")

    if args.synthetic:
        print("\nUsing synthetic dataset")
        train_loader, val_loader = get_synthetic_loaders(
            batch_size=args.batch_size, image_size=args.image_size, num_workers=0
        )
    else:
        # Add real dataset loading here
        train_loader, val_loader = get_synthetic_loaders(
            batch_size=args.batch_size, image_size=args.image_size, num_workers=0
        )

    trainer = RAPiDTwinTrainer(model, train_loader, val_loader, cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
