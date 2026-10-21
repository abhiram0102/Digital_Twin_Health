"""
Adaptive Retinal Preprocessing Pipeline.
Steps: green channel → illumination normalization → CLAHE →
       noise removal → vessel enhancement (Frangi) → multi-scale composite.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .quality_assessment import AdaptiveQualityAssessor, QualityMetrics


def frangi_filter(image: np.ndarray, sigmas=(1, 2, 4, 8),
                  beta1: float = 0.5, beta2: float = 15.0) -> np.ndarray:
    """
    Frangi vessel filter using Hessian eigenvalue analysis.
    Highlights tubular (vessel) structures in 2D grayscale image.
    """
    img = image.astype(np.float64)
    result = np.zeros_like(img)

    for sigma in sigmas:
        # Smooth image at this scale
        smoothed = cv2.GaussianBlur(img, (0, 0), sigma)

        # Hessian via sequential first-order Sobel derivatives
        Ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)

        # Scale by sigma^2 (Lindeberg normalization)
        Ixx *= sigma ** 2
        Iyy *= sigma ** 2
        Ixy *= sigma ** 2

        # Eigenvalues of 2x2 Hessian at each pixel
        tmp = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
        lam1 = 0.5 * (Ixx + Iyy + tmp)
        lam2 = 0.5 * (Ixx + Iyy - tmp)

        # Sort so |lam1| <= |lam2|
        swap = np.abs(lam1) > np.abs(lam2)
        lam1[swap], lam2[swap] = lam2[swap].copy(), lam1[swap].copy()

        # Vesselness: Rb = lam1/lam2, S = Frobenius norm
        with np.errstate(divide='ignore', invalid='ignore'):
            Rb = np.where(lam2 != 0, np.abs(lam1) / (np.abs(lam2) + 1e-10), 0)
        S = np.sqrt(lam1**2 + lam2**2)

        vesselness = np.where(
            lam2 < 0,
            np.exp(-Rb**2 / (2 * beta1**2)) * (1 - np.exp(-S**2 / (2 * beta2**2))),
            0.0
        )
        result = np.maximum(result, vesselness)

    return result.astype(np.float32)


def estimate_background(green: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Morphological opening → background illumination estimate."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    bg_blur = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), kernel_size // 4)
    return bg_blur


def normalize_illumination(green: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Subtract background illumination, rescale to [0, 255]."""
    bg = estimate_background(green)
    normalized = green.astype(np.float32) - strength * bg
    normalized = normalized - normalized.min()
    denom = normalized.max()
    if denom > 0:
        normalized = normalized / denom * 255.0
    return normalized.astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def remove_noise(image: np.ndarray, gaussian_k: int = 5,
                 use_median: bool = True) -> np.ndarray:
    """Gaussian + optional median filtering for noise removal."""
    denoised = cv2.GaussianBlur(image, (gaussian_k, gaussian_k), 0)
    if use_median:
        denoised = cv2.medianBlur(denoised, 3)
    return denoised


def tophat_enhancement(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Morphological top-hat enhances thin vessel structures."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(image, tophat)
    return enhanced


@dataclass
class ProcessedRetinalImage:
    composite: np.ndarray           # (H, W, 3) float32 — 3-channel composite
    green_channel: np.ndarray       # (H, W) uint8
    clahe_enhanced: np.ndarray      # (H, W) uint8
    vessel_enhanced: np.ndarray     # (H, W) float32 — Frangi output
    quality_metrics: QualityMetrics
    preprocessing_params: Dict


class AdaptiveRetinalPreprocessor:
    """
    Complete adaptive preprocessing pipeline for fundus images.
    Automatically adjusts parameters based on image quality assessment.
    Outputs a 3-channel composite: [green, CLAHE, Frangi_vessel].
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512),
                 use_quality_cnn: bool = False, device: str = "cpu"):
        self.target_size = target_size
        self.assessor = AdaptiveQualityAssessor(use_cnn=use_quality_cnn, device=device)

    def preprocess(self, image_rgb: np.ndarray,
                   override_params: Optional[Dict] = None) -> ProcessedRetinalImage:
        """
        Full preprocessing pipeline.

        Args:
            image_rgb: (H, W, 3) uint8 RGB fundus image
            override_params: optional dict to override adaptive parameters

        Returns:
            ProcessedRetinalImage with all intermediate outputs
        """
        # Resize to target
        img = cv2.resize(image_rgb, self.target_size[::-1])  # (W, H)

        # 1. Quality assessment
        quality = self.assessor.assess(img)
        params = override_params or self.assessor.recommend_params(quality)

        # 2. Green channel extraction
        green = img[:, :, 1].copy()

        # 3. Illumination normalization
        norm = normalize_illumination(green, strength=params.get("illum_strength", 1.0))

        # 4. CLAHE contrast enhancement
        clahe = apply_clahe(norm,
                            clip_limit=params.get("clahe_clip_limit", 2.0),
                            tile_grid=params.get("clahe_tile_grid", (8, 8)))

        # 5. Noise removal
        denoised = remove_noise(clahe,
                                gaussian_k=params.get("gaussian_kernel", 5),
                                use_median=params.get("use_median_filter", True))

        # 6. Top-hat + Frangi vessel enhancement
        tophat = tophat_enhancement(denoised)
        vessel = frangi_filter(tophat.astype(np.float64) / 255.0,
                               sigmas=params.get("frangi_sigmas", [1, 2, 4, 8]))

        # 7. Build 3-channel composite: [green_norm, clahe, vessel_map]
        g_norm = (norm.astype(np.float32)) / 255.0
        c_norm = (denoised.astype(np.float32)) / 255.0
        v_norm = vessel / (vessel.max() + 1e-8)

        composite = np.stack([g_norm, c_norm, v_norm], axis=-1)  # (H, W, 3) float32

        return ProcessedRetinalImage(
            composite=composite,
            green_channel=norm,
            clahe_enhanced=denoised,
            vessel_enhanced=vessel,
            quality_metrics=quality,
            preprocessing_params=params
        )

    def preprocess_batch(self, images):
        """Process a list of RGB images, return list of ProcessedRetinalImage."""
        return [self.preprocess(img) for img in images]

    def to_tensor(self, processed: ProcessedRetinalImage) -> "torch.Tensor":
        """Convert composite to CHW float32 tensor in [0,1]."""
        import torch
        arr = processed.composite.transpose(2, 0, 1)  # (3, H, W)
        return torch.from_numpy(arr.copy()).float()
