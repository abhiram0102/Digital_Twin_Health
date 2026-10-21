"""
Retinal Dataset Loaders — production-grade PyTorch Dataset / DataLoader wrappers.

Supported datasets
──────────────────
Vessel segmentation:
  DRIVE         — 40 images (20 train / 20 test), 565 × 584
  STARE         — 20 images, 700 × 605
  CHASE_DB1     — 28 images, 999 × 960
  CombinedSeg   — union of all three with harmonised splits

Disease classification:
  APTOS 2019    — 3662 images, 5-class DR (Kaggle format)
  EyePACS       — large-scale DR dataset (Kaggle format)
  Messidor-2    — 1748 images, 4-class DR

Glaucoma:
  REFUGE        — 1200 images, binary glaucoma/normal
  RIM-ONE DL    — 485 images, binary glaucoma/normal

Development / testing:
  SyntheticRAPiDDataset — deterministic random tensors (unit-test only)

All real datasets fall back gracefully to synthetic samples when files are
absent, ensuring the training loop never crashes on a missing dataset.
"""

from __future__ import annotations

import csv
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ─── Normalisation ────────────────────────────────────────────────────────────

# ImageNet statistics (used for TIMM backbone fine-tuning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_normalize(img: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) float32 [0,1] → normalised float32."""
    return (img - IMAGENET_MEAN) / IMAGENET_STD


# ─── Augmentation pipeline ────────────────────────────────────────────────────

class RetinalAugmentation:
    """
    Augmentation pipeline for retinal fundus images.

    Applies (each independently gated by its own probability):
      - Random horizontal / vertical flip
      - Random rotation ±rotation°
      - Random brightness / contrast jitter
      - CLAHE contrast enhancement (retinal-specific)
      - Gamma correction (illumination robustness)
      - Gaussian blur (camera-defocus simulation)
      - Optional elastic deformation (for segmentation)
      - Optional Gaussian noise injection

    Reproducibility:
      Pass `seed` to pin the internal numpy Generator.  All random decisions
      (flip, angle, clip, grid, gamma, kernel, noise, elastic) are drawn from
      `self.rng` — a `numpy.random.default_rng(seed)` instance — instead of
      the global `random` / `np.random` state.  The default `seed=None` gives
      non-deterministic behaviour suitable for training.
    """

    def __init__(
        self,
        rotation:   float = 30.0,
        brightness: float = 0.2,
        contrast:   float = 0.2,
        flip:       bool  = True,
        elastic:    bool  = False,
        noise_std:  float = 0.02,
        p:          float = 0.5,
        p_clahe:    float = 0.5,
        p_gamma:    float = 0.5,
        p_blur:     float = 0.3,
        seed:       Optional[int] = None,
    ):
        self.rotation   = rotation
        self.brightness = brightness
        self.contrast   = contrast
        self.flip       = flip
        self.elastic    = elastic
        self.noise_std  = noise_std
        self.p          = p
        self.p_clahe    = p_clahe
        self.p_gamma    = p_gamma
        self.p_blur     = p_blur
        # Seeded numpy Generator — sole source of randomness in this class.
        self.rng = np.random.default_rng(seed)

    # ── Transforms ───────────────────────────────────────────────────────────

    def _random_rotation(self, img: np.ndarray, mask: Optional[np.ndarray]):
        angle = float(self.rng.uniform(-self.rotation, self.rotation))
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)
        return img, mask

    def _random_flip(self, img: np.ndarray, mask: Optional[np.ndarray]):
        if self.flip:
            if self.rng.random() < 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy() if mask is not None else None
            if self.rng.random() < 0.3:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy() if mask is not None else None
        return img, mask

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        delta  = float(self.rng.uniform(-self.brightness, self.brightness))
        img_f  = np.clip(img.astype(np.float32) / 255.0 + delta, 0.0, 1.0)
        factor = float(self.rng.uniform(1 - self.contrast, 1 + self.contrast))
        mean   = img_f.mean()
        img_f  = np.clip((img_f - mean) * factor + mean, 0.0, 1.0)
        return (img_f * 255).astype(np.uint8)

    def _clahe_augment(self, img: np.ndarray) -> np.ndarray:
        clip  = float(self.rng.uniform(1.5, 3.5))
        grid  = int(self.rng.choice([6, 8, 10]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        channels = cv2.split(img)
        enhanced = [clahe.apply(c) for c in channels]
        return cv2.merge(enhanced)

    def _gamma_augment(self, img: np.ndarray) -> np.ndarray:
        gamma = float(self.rng.uniform(0.7, 1.4))
        lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        return lut[img]

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        k     = int(self.rng.choice([3, 5]))
        sigma = float(self.rng.uniform(0.5, 1.5))
        return cv2.GaussianBlur(img, (k, k), sigma)

    def _elastic_transform(self, img: np.ndarray,
                           mask: Optional[np.ndarray],
                           alpha: float = 50, sigma: float = 5):
        h, w = img.shape[:2]
        dx   = cv2.GaussianBlur(
            self.rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma) * alpha
        dy   = cv2.GaussianBlur(
            self.rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma) * alpha
        xs   = np.clip(np.meshgrid(np.arange(w), np.arange(h))[0] + dx, 0, w - 1).astype(np.float32)
        ys   = np.clip(np.meshgrid(np.arange(w), np.arange(h))[1] + dy, 0, h - 1).astype(np.float32)
        img  = cv2.remap(img, xs, ys, cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.remap(mask, xs, ys, cv2.INTER_NEAREST)
        return img, mask

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        noise = self.rng.standard_normal(img.shape).astype(np.float32) * self.noise_std * 255
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def __call__(
        self,
        image: np.ndarray,
        mask:  Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.rng.random() > self.p:
            return image, mask
        image, mask = self._random_flip(image, mask)
        image, mask = self._random_rotation(image, mask)
        image       = self._color_jitter(image)
        if self.p_clahe > 0 and self.rng.random() < self.p_clahe:
            image = self._clahe_augment(image)
        if self.p_gamma > 0 and self.rng.random() < self.p_gamma:
            image = self._gamma_augment(image)
        if self.p_blur > 0 and self.rng.random() < self.p_blur:
            image = self._gaussian_blur(image)
        if self.elastic and self.rng.random() < 0.3:
            image, mask = self._elastic_transform(image, mask)
        if self.noise_std > 0 and self.rng.random() < 0.3:
            image = self._add_noise(image)
        return image, mask


# ─── Shared image / mask loaders ─────────────────────────────────────────────

def _load_rgb(path: Path, size: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))


def _load_mask(path: Path, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.float32)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((size, size), dtype=np.float32)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def _img_to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32, optionally ImageNet-normalised."""
    img_f = img.astype(np.float32) / 255.0
    if normalize:
        img_f = imagenet_normalize(img_f)
    return torch.from_numpy(img_f.transpose(2, 0, 1))


def _synthetic_image(size: int) -> torch.Tensor:
    """Synthetic fundus image normalised to ImageNet statistics.

    Real images go through _img_to_tensor(..., normalize=True) which applies
    (pixel/255 - mean) / std.  The synthetic fallback must live in the same
    numerical space so downstream layers see a consistent distribution.
    """
    img = torch.rand(3, size, size)                               # uniform [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img - mean) / std                                     # ≈ [-2.1, 2.6]


def _synthetic_mask(size: int) -> torch.Tensor:
    return (torch.rand(1, size, size) > 0.88).float()


# ─── DRIVE ───────────────────────────────────────────────────────────────────

class DRIVEDataset(Dataset):
    """
    DRIVE Retinal Vessel Segmentation Dataset.

    Expected directory layout:
        root/
          training/
            images/   *.tif
            1st_manual/ *.gif
          test/
            images/   *.tif
            1st_manual/ *.gif (if available)
    """

    def __init__(self, root: str, split: str = "training",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True):
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.7) if augment else None

        root_p = Path(root)
        img_dir  = root_p / split / "images"
        mask_dir = root_p / split / "1st_manual"

        self.samples: List[Tuple[Path, Optional[Path]]] = []
        if img_dir.exists():
            imgs  = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.png"))
            masks = (sorted(mask_dir.glob("*.gif")) + sorted(mask_dir.glob("*.png"))
                     if mask_dir.exists() else [])
            for i, img_p in enumerate(imgs):
                m = masks[i] if i < len(masks) else None
                self.samples.append((img_p, m))

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "synthetic_drive"}
        img_p, mask_p = self.samples[idx % len(self.samples)]
        img  = _load_rgb(img_p, self.image_size)
        if img is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "drive"}
        mask = _load_mask(mask_p, self.image_size)
        if self.aug:
            img, mask = self.aug(img, mask)
        img_t  = _img_to_tensor(img, self.normalize)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return {"image": img_t, "mask": mask_t,
                "image_path": str(img_p), "dataset": "drive"}


# ─── STARE ────────────────────────────────────────────────────────────────────

class STAREDataset(Dataset):
    """
    STARE dataset.

    Expected layout:
        root/
          images/  *.ppm or *.png
          labels/  *.ah.ppm or *.png  (Hoover labels)
    """

    def __init__(self, root: str, image_size: int = 512,
                 augment: bool = True, normalize: bool = True):
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        root_p = Path(root)
        imgs  = sorted((root_p / "images").glob("*.ppm")) + \
                sorted((root_p / "images").glob("*.png")) if (root_p / "images").exists() else []
        masks = sorted((root_p / "labels").glob("*.ppm")) + \
                sorted((root_p / "labels").glob("*.png")) if (root_p / "labels").exists() else []

        self.samples = [(imgs[i], masks[i] if i < len(masks) else None)
                        for i in range(len(imgs))]

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "synthetic_stare"}
        img_p, mask_p = self.samples[idx % len(self.samples)]
        img  = _load_rgb(img_p, self.image_size)
        if img is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size), "dataset": "stare"}
        mask = _load_mask(mask_p, self.image_size)
        if self.aug:
            img, mask = self.aug(img, mask)
        return {"image": _img_to_tensor(img, self.normalize),
                "mask":  torch.from_numpy(mask).unsqueeze(0),
                "dataset": "stare"}


# ─── CHASE_DB1 ────────────────────────────────────────────────────────────────

class CHASE_DB1Dataset(Dataset):
    """
    CHASE_DB1 dataset.

    Expected layout:
        root/
          Image_*L.jpg, Image_*R.jpg
          Image_*L_1stHO.png  (manual labels)
    """

    def __init__(self, root: str, image_size: int = 512,
                 augment: bool = True, normalize: bool = True):
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        root_p = Path(root)
        imgs   = sorted(root_p.glob("Image_*.jpg")) + sorted(root_p.glob("Image_*.png")) \
                 if root_p.exists() else []
        # Match label by replacing extension with _1stHO.png
        self.samples: List[Tuple[Path, Optional[Path]]] = []
        for img_p in imgs:
            stem  = img_p.stem                  # e.g. Image_01L
            label = root_p / f"{stem}_1stHO.png"
            self.samples.append((img_p, label if label.exists() else None))

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "synthetic_chase"}
        img_p, mask_p = self.samples[idx % len(self.samples)]
        img  = _load_rgb(img_p, self.image_size)
        if img is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size), "dataset": "chase_db1"}
        mask = _load_mask(mask_p, self.image_size)
        if self.aug:
            img, mask = self.aug(img, mask)
        return {"image": _img_to_tensor(img, self.normalize),
                "mask":  torch.from_numpy(mask).unsqueeze(0),
                "dataset": "chase_db1"}


# ─── HRF ──────────────────────────────────────────────────────────────────────

class HRFDataset(Dataset):
    """
    High-Resolution Fundus (HRF) vessel segmentation dataset.

    Expected layout:
        root/
          images/   *.JPG  (45 images: 15 healthy, 15 diabetic, 15 glaucoma)
          manual1/  *.tif  (first manual segmentation)

    Reference: Budai et al., "Robust Vessel Segmentation in Fundus Images"
    International Journal of Biomedical Imaging, 2013.
    """

    def __init__(self, root: str, image_size: int = 512,
                 augment: bool = True, normalize: bool = True):
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.7, elastic=True) if augment else None

        root_p   = Path(root)
        img_dir  = root_p / "images"
        mask_dir = root_p / "manual1"

        self.samples: List[Tuple[Optional[Path], Optional[Path]]] = []
        if img_dir.exists():
            img_exts = ["*.JPG", "*.jpg", "*.png", "*.tif"]
            imgs: List[Path] = []
            for ext in img_exts:
                imgs += sorted(img_dir.glob(ext))
            for img_p in imgs:
                stem   = img_p.stem
                mask_p = None
                if mask_dir.exists():
                    for mext in [".tif", ".png", ".gif"]:
                        candidate = mask_dir / f"{stem}{mext}"
                        if candidate.exists():
                            mask_p = candidate
                            break
                self.samples.append((img_p, mask_p))

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "synthetic_hrf"}
        img_p, mask_p = self.samples[idx % len(self.samples)]
        img = _load_rgb(img_p, self.image_size)
        if img is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "hrf"}
        mask = _load_mask(mask_p, self.image_size)
        if self.aug:
            img, mask = self.aug(img, mask)
        return {
            "image":   _img_to_tensor(img, self.normalize),
            "mask":    torch.from_numpy(mask).unsqueeze(0),
            "dataset": "hrf",
        }


# ─── Combined segmentation dataset ───────────────────────────────────────────

class CombinedSegDataset(Dataset):
    """
    Merges DRIVE + STARE + CHASE_DB1 + HRF.
    Falls back to synthetic samples for any missing datasets.

    Stores raw file-path pairs (img_path, mask_path) and loads on demand so
    that augmentation is always applied to uint8 numpy arrays — never to
    pre-normalised tensors (which was the v1 bug: multiplying an
    ImageNet-normalised tensor by 255 before passing to OpenCV transforms).
    """

    def __init__(self, roots: Dict[str, str], split: str = "training",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True):
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(
            p=0.7, elastic=True, p_clahe=0.5, p_gamma=0.5, p_blur=0.3,
        ) if augment else None

        self.samples: List[Tuple[Optional[Path], Optional[Path]]] = []
        self._dataset_labels: List[str] = []

        if "drive" in roots and roots["drive"]:
            ds = DRIVEDataset(roots["drive"], split=split,
                              image_size=image_size, augment=False, normalize=normalize)
            self.samples.extend(ds.samples)
            self._dataset_labels.extend(["drive"] * len(ds.samples))

        if "stare" in roots and roots["stare"]:
            ds = STAREDataset(roots["stare"], image_size=image_size,
                              augment=False, normalize=normalize)
            self.samples.extend(ds.samples)
            self._dataset_labels.extend(["stare"] * len(ds.samples))

        if "chase" in roots and roots["chase"]:
            ds = CHASE_DB1Dataset(roots["chase"], image_size=image_size,
                                  augment=False, normalize=normalize)
            self.samples.extend(ds.samples)
            self._dataset_labels.extend(["chase"] * len(ds.samples))

        if "hrf" in roots and roots["hrf"]:
            ds = HRFDataset(roots["hrf"], image_size=image_size,
                            augment=False, normalize=normalize)
            self.samples.extend(ds.samples)
            self._dataset_labels.extend(["hrf"] * len(ds.samples))

        if not self.samples:
            warnings.warn("CombinedSegDataset: no real data found — using synthetic samples.")
            self.samples = [(None, None)] * 200
            self._dataset_labels = ["synthetic"] * 200

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx  = idx % len(self.samples)
        img_p, mask_p = self.samples[real_idx]
        ds_name   = self._dataset_labels[real_idx] if self._dataset_labels else "combined_seg"

        if img_p is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": "synthetic"}

        img = _load_rgb(img_p, self.image_size)
        if img is None:
            return {"image": _synthetic_image(self.image_size),
                    "mask":  _synthetic_mask(self.image_size),
                    "dataset": ds_name}

        mask = _load_mask(mask_p, self.image_size)

        if self.aug:
            img, mask = self.aug(img, mask)

        return {
            "image":   _img_to_tensor(img, self.normalize),
            "mask":    torch.from_numpy(mask).unsqueeze(0),
            "dataset": ds_name,
        }


# ─── APTOS 2019 ───────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    APTOS 2019 Diabetic Retinopathy Detection.

    Expected layout:
        root/
          train.csv    columns: id_code, diagnosis
          test.csv     columns: id_code
          train_images/  *.png
          test_images/   *.png

    Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
    """

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None
        self.split      = split

        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        csv_path = self.root / f"{self.split}.csv"
        img_dir  = self.root / f"{self.split}_images"
        if not csv_path.exists():
            warnings.warn(f"APTOS: {csv_path} not found — using synthetic labels.")
            return [{"path": None, "label": random.randint(0, 4)} for _ in range(200)]

        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code  = row.get("id_code", row.get("image", ""))
                label = int(row.get("diagnosis", row.get("level", 0)))
                for ext in (".png", ".jpeg", ".jpg"):
                    p = img_dir / f"{code}{ext}"
                    if p.exists():
                        samples.append({"path": p, "label": label})
                        break
                else:
                    samples.append({"path": None, "label": label})
        return samples if samples else [{"path": None, "label": 0} for _ in range(200)]

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted sampling."""
        from collections import Counter
        counts = Counter(s["label"] for s in self.samples)
        n      = len(self.samples)
        weights = torch.zeros(5)
        for cls, cnt in counts.items():
            weights[cls] = n / (cnt * 5 + 1e-6)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s     = self.samples[idx]
        label = s["label"]

        if s["path"] is None or not Path(str(s["path"])).exists():
            img_t = _synthetic_image(self.image_size)
        else:
            img = _load_rgb(Path(str(s["path"])), self.image_size)
            if img is None:
                img_t = _synthetic_image(self.image_size)
            else:
                if self.aug:
                    img, _ = self.aug(img)
                img_t = _img_to_tensor(img, self.normalize)

        return {
            "image":    img_t,
            "labels": {
                "diabetic_retinopathy": torch.tensor(label, dtype=torch.long),
            },
            "sample_id": idx,
            "dataset":   "aptos",
        }


# ─── EyePACS ──────────────────────────────────────────────────────────────────

class EyePACSDataset(Dataset):
    """
    EyePACS Diabetic Retinopathy dataset (Kaggle format).

    Expected layout:
        root/
          train.csv    columns: image, level
          train_images/ *.jpeg

    Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
    """

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True, max_samples: Optional[int] = None):
        self.root       = Path(root)
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        self.samples = self._load_metadata(split, max_samples)

    def _load_metadata(self, split: str,
                       max_samples: Optional[int]) -> List[Dict]:
        csv_path = self.root / f"{split}.csv"
        img_dir  = self.root / f"{split}_images"
        if not csv_path.exists():
            n = min(max_samples or 200, 200)
            warnings.warn(f"EyePACS: {csv_path} not found — using synthetic labels.")
            return [{"path": None, "label": random.randint(0, 4)} for _ in range(n)]

        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code  = row.get("image", "")
                label = int(row.get("level", 0))
                p     = img_dir / f"{code}.jpeg"
                samples.append({"path": p if p.exists() else None, "label": label})
                if max_samples and len(samples) >= max_samples:
                    break
        return samples or [{"path": None, "label": 0} for _ in range(100)]

    def get_class_weights(self) -> torch.Tensor:
        from collections import Counter
        counts = Counter(s["label"] for s in self.samples)
        n      = len(self.samples)
        weights = torch.zeros(5)
        for cls, cnt in counts.items():
            weights[cls] = n / (cnt * 5 + 1e-6)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s     = self.samples[idx]
        label = s["label"]

        if s["path"] is None or not Path(str(s["path"])).exists():
            img_t = _synthetic_image(self.image_size)
        else:
            img = _load_rgb(Path(str(s["path"])), self.image_size)
            if img is None:
                img_t = _synthetic_image(self.image_size)
            else:
                if self.aug:
                    img, _ = self.aug(img)
                img_t = _img_to_tensor(img, self.normalize)

        return {
            "image":  img_t,
            "labels": {
                "diabetic_retinopathy": torch.tensor(label, dtype=torch.long),
            },
            "sample_id": idx,
            "dataset":   "eyepacs",
        }


# ─── Messidor-2 ───────────────────────────────────────────────────────────────

class Messidor2Dataset(Dataset):
    """
    Messidor-2 Diabetic Retinopathy dataset.

    Expected layout:
        root/
          messidor_data.csv   columns: image_id, adjudicated_dr_grade, adjudicated_gradable
          IMAGES/  *.tif or *.png

    Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Vision-threatening DR
    (grade 3+4 merged to class 3)
    """

    def __init__(self, root: str, image_size: int = 512,
                 augment: bool = True, normalize: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        self.samples = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        csv_path = self.root / "messidor_data.csv"
        img_dir  = self.root / "IMAGES"
        if not csv_path.exists():
            warnings.warn("Messidor-2: CSV not found — using synthetic.")
            return [{"path": None, "label": random.randint(0, 3)} for _ in range(100)]

        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gradable = int(row.get("adjudicated_gradable", 1))
                if gradable == 0:
                    continue
                img_id = row.get("image_id", "")
                grade  = int(row.get("adjudicated_dr_grade", 0))
                label  = min(grade, 3)

                for ext in (".tif", ".png", ".jpg", ".jpeg"):
                    p = img_dir / f"{img_id}{ext}"
                    if not p.exists():
                        p = img_dir / f"{img_id}"
                    if p.exists():
                        samples.append({"path": p, "label": label})
                        break
                else:
                    samples.append({"path": None, "label": label})
        return samples or [{"path": None, "label": 0} for _ in range(100)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s     = self.samples[idx]
        label = s["label"]
        if s["path"] is None or not Path(str(s["path"])).exists():
            img_t = _synthetic_image(self.image_size)
        else:
            img = _load_rgb(Path(str(s["path"])), self.image_size)
            img_t = _img_to_tensor(img, self.normalize) if img is not None \
                    else _synthetic_image(self.image_size)
            if self.aug and img is not None:
                img, _ = self.aug(img)
                img_t  = _img_to_tensor(img, self.normalize)

        return {
            "image":  img_t,
            "labels": {
                "diabetic_retinopathy": torch.tensor(label, dtype=torch.long),
            },
            "dataset": "messidor2",
        }


# ─── REFUGE ───────────────────────────────────────────────────────────────────

class REFUGEDataset(Dataset):
    """
    REFUGE Glaucoma Challenge dataset.

    Expected layout:
        root/
          Training400/
            Images/    *.jpg
            Glaucoma/  *.jpg  (positive cases)
            Non-Glaucoma/ *.jpg
          Validation400/ (same structure)
          REFUGE-Test400/ (same structure)
    """

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        self.samples = self._collect(split)

    def _collect(self, split: str) -> List[Dict]:
        split_map = {
            "train": "Training400",
            "val":   "Validation400",
            "test":  "REFUGE-Test400",
        }
        folder = self.root / split_map.get(split, "Training400")
        samples = []

        glaucoma_dir    = folder / "Glaucoma"
        non_glaucoma_dir = folder / "Non-Glaucoma"

        for d, lbl in [(glaucoma_dir, 1), (non_glaucoma_dir, 0)]:
            if d.exists():
                for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
                    samples.append({"path": p, "label": lbl})

        if not samples:
            warnings.warn(f"REFUGE: {folder} not found — using synthetic.")
            return [{"path": None, "label": random.randint(0, 1)} for _ in range(100)]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s     = self.samples[idx]
        label = s["label"]
        if s["path"] is None or not Path(str(s["path"])).exists():
            img_t = _synthetic_image(self.image_size)
        else:
            img = _load_rgb(Path(str(s["path"])), self.image_size)
            if img is None:
                img_t = _synthetic_image(self.image_size)
            else:
                if self.aug:
                    img, _ = self.aug(img)
                img_t = _img_to_tensor(img, self.normalize)
        return {
            "image":  img_t,
            "labels": {
                "glaucoma": torch.tensor(label, dtype=torch.long),
            },
            "dataset": "refuge",
        }


# ─── RIM-ONE DL ───────────────────────────────────────────────────────────────

class RIMONEDataset(Dataset):
    """
    RIM-ONE DL Glaucoma dataset.

    Expected layout:
        root/
          Normal/   *.png
          Glaucoma/ *.png
    """

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 512, augment: bool = True,
                 normalize: bool = True, val_ratio: float = 0.2):
        self.root       = Path(root)
        self.image_size = image_size
        self.normalize  = normalize
        self.aug        = RetinalAugmentation(p=0.6) if augment else None

        all_samples = self._collect()
        # Use a local Random instance to avoid contaminating the global random state.
        random.Random(42).shuffle(all_samples)
        n_val = int(len(all_samples) * val_ratio)
        if split == "train":
            self.samples = all_samples[n_val:]
        else:
            self.samples = all_samples[:n_val]

    def _collect(self) -> List[Dict]:
        samples = []
        for lbl, dirname in [(0, "Normal"), (1, "Glaucoma")]:
            d = self.root / dirname
            if d.exists():
                for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                    samples.append({"path": p, "label": lbl})
        if not samples:
            warnings.warn("RIM-ONE DL: data not found — using synthetic.")
            _rng = random.Random(42)
            return [{"path": None, "label": _rng.randint(0, 1)} for _ in range(100)]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        label = s["label"]
        if s["path"] is None or not Path(str(s["path"])).exists():
            img_t = _synthetic_image(self.image_size)
        else:
            img = _load_rgb(Path(str(s["path"])), self.image_size)
            if img is None:
                img_t = _synthetic_image(self.image_size)
            else:
                if self.aug:
                    img, _ = self.aug(img)
                img_t = _img_to_tensor(img, self.normalize)
        return {
            "image":  img_t,
            "labels": {
                "glaucoma": torch.tensor(label, dtype=torch.long),
            },
            "dataset": "rimone",
        }


# ─── Synthetic dataset (unit tests / CI) ─────────────────────────────────────

class SyntheticRAPiDDataset(Dataset):
    """
    Deterministic synthetic dataset for unit tests and CI.
    Uses fixed random seed — reproducible across runs.

    NOT suitable for actual model training/evaluation.
    """

    def __init__(self, size: int = 200, image_size: int = 128,
                 physio_len: int = 24, seq_len: int = 12, seed: int = 42):
        self.size       = size
        self.image_size = image_size
        self.physio_len = physio_len
        self.seq_len    = seq_len

        rng = torch.Generator().manual_seed(seed)
        self.images            = torch.rand(size, 3, image_size, image_size, generator=rng)
        self.physio            = torch.rand(size, physio_len, 8, generator=rng)
        self.dr_labels         = torch.randint(0, 5,  (size,), generator=rng)
        self.glaucoma_labels   = torch.randint(0, 2,  (size,), generator=rng)
        self.amd_labels        = torch.randint(0, 4,  (size,), generator=rng)
        self.rvo_labels        = torch.randint(0, 2,  (size,), generator=rng)
        self.diabetes_labels   = torch.randint(0, 3,  (size,), generator=rng)
        self.hypert_labels     = torch.randint(0, 3,  (size,), generator=rng)
        self.cv_labels         = torch.randint(0, 4,  (size,), generator=rng)
        self.vessel_masks      = (torch.rand(size, 1, image_size, image_size, generator=rng) > 0.85).float()
        self.hist_states       = torch.randn(size, seq_len, 256, generator=rng)

        _rng = random.Random(seed)
        self.node_features = [torch.randn(_rng.randint(5, 50), 16) for _ in range(size)]
        self.edge_features = [torch.randn(_rng.randint(4, 100), 8)  for _ in range(size)]
        self.edge_indices  = []
        for i in range(size):
            n = self.node_features[i].shape[0]
            e = self.edge_features[i].shape[0]
            src = torch.randint(0, n, (e,))
            dst = torch.randint(0, n, (e,))
            self.edge_indices.append(torch.stack([src, dst]))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image":            self.images[idx],
            "vessel_mask":      self.vessel_masks[idx],
            "physio_signals":   self.physio[idx],
            "historical_states": self.hist_states[idx],
            "node_features":    self.node_features[idx],
            "edge_features":    self.edge_features[idx],
            "edge_index":       self.edge_indices[idx],
            "dataset":          "synthetic",
            "labels": {
                "diabetic_retinopathy":  self.dr_labels[idx],
                "glaucoma":              self.glaucoma_labels[idx],
                "macular_degeneration":  self.amd_labels[idx],
                "retinal_vein_occlusion": self.rvo_labels[idx],
                "diabetes":              self.diabetes_labels[idx],
                "hypertension":          self.hypert_labels[idx],
                "cardiovascular":        self.cv_labels[idx],
            },
        }


# ─── Collate function ─────────────────────────────────────────────────────────

def collate_graph_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate handling variable-size graphs.
    Graph tensors (node_features, edge_index, edge_features) are kept as lists.
    Everything else is stacked normally.
    """
    GRAPH_KEYS  = {"node_features", "edge_features", "edge_index"}
    LABEL_KEY   = "labels"
    STRING_KEYS = {"image_path", "dataset"}

    out: Dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        items = [b[key] for b in batch]

        if key in GRAPH_KEYS:
            out[key] = items                          # list of tensors

        elif key == LABEL_KEY:
            # Stack each label type
            label_keys = items[0].keys()
            out[LABEL_KEY] = {
                lk: torch.stack([it[lk] for it in items])
                for lk in label_keys
                if isinstance(items[0][lk], torch.Tensor)
            }

        elif key in STRING_KEYS:
            out[key] = items                          # list of strings

        elif isinstance(items[0], torch.Tensor):
            out[key] = torch.stack(items)

        else:
            out[key] = items

    return out


# ─── Weighted sampler helper ──────────────────────────────────────────────────

def make_weighted_sampler(dataset: Dataset,
                          label_key: str = "diabetic_retinopathy",
                          num_classes: int = 5) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for class-balanced training.

    Fast path: if the dataset exposes a ``samples`` attribute whose items are
    dicts with a ``"label"`` key (all classification datasets here do), labels
    are read directly from metadata without loading a single image.

    Slow path fallback: iterate __getitem__ for arbitrary datasets.
    """
    from collections import Counter

    labels_list: List[int] = []

    # Fast path — avoids loading all images (critical for EyePACS / APTOS).
    if (hasattr(dataset, "samples")
            and dataset.samples
            and isinstance(dataset.samples[0], dict)
            and "label" in dataset.samples[0]):
        labels_list = [int(s.get("label", 0)) for s in dataset.samples]
    else:
        # Slow path — only used for datasets without pre-loaded metadata.
        for i in range(len(dataset)):
            sample = dataset[i]
            lbl = sample.get("labels", {}).get(label_key)
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            labels_list.append(int(lbl) if lbl is not None else 0)

    counts  = Counter(labels_list)
    total   = len(labels_list)
    class_w = {cls: total / (cnt * num_classes + 1e-6)
               for cls, cnt in counts.items()}
    sample_weights = torch.tensor([class_w.get(lbl, 1.0) for lbl in labels_list])

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


# ─── DataLoader factories ─────────────────────────────────────────────────────

def get_segmentation_loaders(
    roots:        Dict[str, str],
    batch_size:   int  = 8,
    image_size:   int  = 512,
    num_workers:  int  = 4,
    normalize:    bool = True,
    preprocessor  = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for combined vessel segmentation.

    roots may contain any subset of: 'drive', 'stare', 'chase', 'hrf'.
    preprocessor (AdaptiveRetinalPreprocessor) is accepted for API compatibility
    but augmentation is handled per-sample inside CombinedSegDataset using
    RetinalAugmentation (CLAHE, gamma, blur, flip, rotation, elastic, noise).
    """
    train_ds = CombinedSegDataset(roots, split="training", image_size=image_size,
                                  augment=True, normalize=normalize)
    val_ds   = CombinedSegDataset(roots, split="test", image_size=image_size,
                                  augment=False, normalize=normalize)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    ds_counts = {}
    for lbl in train_ds._dataset_labels:
        ds_counts[lbl] = ds_counts.get(lbl, 0) + 1
    print(f"  CombinedSegDataset train={n_train} val={n_val}  "
          f"sources={dict(sorted(ds_counts.items()))}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    return train_loader, val_loader


def get_classification_loaders(
    roots:           Dict[str, str],
    dataset_name:    str = "aptos",
    batch_size:      int = 16,
    image_size:      int = 512,
    num_workers:     int = 4,
    normalize:       bool = True,
    class_balance:   bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for a classification dataset.

    dataset_name: one of 'aptos', 'eyepacs', 'messidor2', 'refuge', 'rimone'
    """
    ds_map = {
        "aptos":    APTOSDataset,
        "eyepacs":  EyePACSDataset,
        "messidor2": Messidor2Dataset,
        "refuge":   REFUGEDataset,
        "rimone":   RIMONEDataset,
    }
    if dataset_name not in ds_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(ds_map.keys())}")

    root = roots.get(dataset_name, "")
    Cls  = ds_map[dataset_name]

    if dataset_name in ("refuge", "rimone"):
        train_ds = Cls(root, split="train", image_size=image_size,
                       augment=True, normalize=normalize)
        val_ds   = Cls(root, split="val",   image_size=image_size,
                       augment=False, normalize=normalize)
    else:
        train_ds = Cls(root, split="train", image_size=image_size,
                       augment=True, normalize=normalize)
        val_ds   = Cls(root, split="test",  image_size=image_size,
                       augment=False, normalize=normalize)

    sampler = None
    shuffle = True
    if class_balance and hasattr(train_ds, "get_class_weights"):
        sampler = make_weighted_sampler(train_ds)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True,
                              collate_fn=collate_graph_batch)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_graph_batch)
    return train_loader, val_loader


def get_synthetic_loaders(
    batch_size:  int = 8,
    image_size:  int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Synthetic loaders for unit tests and CI pipelines."""
    train_ds = SyntheticRAPiDDataset(size=200, image_size=image_size, seed=42)
    val_ds   = SyntheticRAPiDDataset(size=50,  image_size=image_size, seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_graph_batch, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_graph_batch, num_workers=num_workers)
    return train_loader, val_loader
