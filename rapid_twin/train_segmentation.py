#!/usr/bin/env python3
"""
Entry point: Train Attention U-Net vessel segmentation model.
Usage: python -m rapid_twin.train_segmentation [--config config.json]
"""

import argparse
import torch
from pathlib import Path

from rapid_twin.configs.config import DEFAULT_CONFIG
from rapid_twin.segmentation.attention_unet import AttentionUNet
from rapid_twin.datasets.retinal_datasets import get_segmentation_loaders, get_synthetic_loaders
from rapid_twin.preprocessing.pipeline import AdaptiveRetinalPreprocessor
from rapid_twin.training.trainer import SegmentationTrainer
from rapid_twin.utils.metrics import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train RAPiD-Twin Vessel Segmentation")
    parser.add_argument("--data_root",   type=str, default="data")
    parser.add_argument("--hrf_root",    type=str, default=None,
                        help="HRF dataset root (images/ + manual1/). "
                             "Omit to skip HRF.")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data only (testing/CI)")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--image_size",  type=int, default=512)
    parser.add_argument("--device",      type=str, default="auto")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir",     type=str, default="logs")
    parser.add_argument("--grad_accum",  type=int, default=1,
                        help="Gradient accumulation steps (use 4-8 for 512×512 on 8GB GPU)")
    parser.add_argument("--no_ema",      action="store_true",
                        help="Disable EMA weights")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.training.checkpoint_dir = args.checkpoint_dir
    cfg.training.log_dir        = args.log_dir
    if args.epochs:
        cfg.training.seg_epochs = args.epochs
    if args.batch_size:
        cfg.training.seg_batch_size = args.batch_size

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg.training.seed)
    print(f"Device      : {device}")
    print(f"Image size  : {args.image_size}×{args.image_size}")
    print(f"Grad accum  : {args.grad_accum}")
    print(f"EMA         : {not args.no_ema}")

    model = AttentionUNet(
        in_channels=cfg.segmentation.in_channels,
        out_channels=cfg.segmentation.out_channels,
        features=cfg.segmentation.features,
        dropout=cfg.segmentation.dropout,
        use_attention=cfg.segmentation.attention,
        deep_supervision=cfg.segmentation.deep_supervision,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters  : {n_params:,}")

    preprocessor = AdaptiveRetinalPreprocessor(
        target_size=(args.image_size, args.image_size), device=device
    )

    if args.synthetic:
        print("Using synthetic data")
        from torch.utils.data import DataLoader
        import torch as _torch

        class SimpleSegDataset(_torch.utils.data.Dataset):
            def __init__(self, size, image_size):
                _rng = _torch.Generator().manual_seed(42)
                self.images = _torch.rand(size, 3, image_size, image_size,
                                          generator=_rng)
                self.masks  = (_torch.rand(size, 1, image_size, image_size,
                                           generator=_rng) > 0.85).float()
            def __len__(self): return len(self.images)
            def __getitem__(self, i):
                return {"image": self.images[i], "mask": self.masks[i],
                        "dataset": "synthetic"}

        train_ds = SimpleSegDataset(200, args.image_size)
        val_ds   = SimpleSegDataset(50,  args.image_size)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.seg_batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.training.seg_batch_size,
                                  num_workers=0)
    else:
        roots = {
            "drive": str(Path(args.data_root) / "DRIVE"),
            "stare": str(Path(args.data_root) / "STARE"),
            "chase": str(Path(args.data_root) / "CHASE_DB1"),
        }
        if args.hrf_root:
            roots["hrf"] = args.hrf_root
        elif (Path(args.data_root) / "HRF").exists():
            roots["hrf"] = str(Path(args.data_root) / "HRF")

        train_loader, val_loader = get_segmentation_loaders(
            roots,
            batch_size=cfg.training.seg_batch_size,
            image_size=args.image_size,
            preprocessor=preprocessor,
            num_workers=args.num_workers,
        )

    trainer = SegmentationTrainer(
        model, train_loader, val_loader, cfg, device,
        grad_accum_steps=args.grad_accum,
        use_ema=not args.no_ema,
    )
    trainer.train()


if __name__ == "__main__":
    main()
