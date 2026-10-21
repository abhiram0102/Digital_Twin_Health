"""
Retinal vessel segmentation postprocessing.

Topology-preserving cleanup of raw neural network predictions.
All functions operate on numpy arrays (H, W) of float32 in [0, 1].
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ─── Thresholding ─────────────────────────────────────────────────────────────

def threshold_otsu(prob: np.ndarray) -> np.ndarray:
    """Otsu adaptive threshold — no manual tuning required."""
    prob_u8 = (prob * 255).astype(np.uint8)
    _, binary = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.float32)


def threshold_fixed(prob: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (prob > t).astype(np.float32)


# ─── Morphological cleanup ────────────────────────────────────────────────────

def remove_small_components(mask: np.ndarray, min_area_px: int = 50) -> np.ndarray:
    """
    Remove connected components smaller than min_area_px pixels.
    Eliminates isolated noise dots while preserving vessel structure.
    """
    binary = (mask > 0.5).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    out = np.zeros_like(binary)
    for comp_id in range(1, n_comp):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            out[labels == comp_id] = 1
    return out.astype(np.float32)


def fill_vessel_holes(mask: np.ndarray, max_hole_area_px: int = 100) -> np.ndarray:
    """
    Fill small holes inside vessel segments (caused by reflection artefacts).
    """
    binary = (mask > 0.5).astype(np.uint8)
    inv    = 1 - binary
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = binary.copy()
    for comp_id in range(1, n_comp):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area <= max_hole_area_px:
            out[labels == comp_id] = 1
    return out.astype(np.float32)


def morphological_cleanup(
    mask:         np.ndarray,
    close_radius: int = 2,
    open_radius:  int = 1,
) -> np.ndarray:
    """
    Closing (fills small gaps) followed by opening (removes thin noise).
    Radius units: pixels.
    """
    binary = (mask > 0.5).astype(np.uint8)
    if close_radius > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    if open_radius > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    return binary.astype(np.float32)


# ─── Vessel continuity refinement ─────────────────────────────────────────────

def bridge_gaps(mask: np.ndarray, gap_px: int = 5) -> np.ndarray:
    """
    Dilate → close gaps → erode back to approximate original width.
    Connects vessel fragments separated by small gaps.
    """
    binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_px * 2 + 1, gap_px * 2 + 1))
    dilated  = cv2.dilate(binary, kernel)
    closed   = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    eroded   = cv2.erode(closed, kernel)
    combined = np.maximum(binary, eroded)
    return combined.astype(np.float32)


# ─── Full postprocessing pipeline ─────────────────────────────────────────────

def postprocess(
    prob:              np.ndarray,
    threshold:         float = 0.5,
    use_otsu:          bool  = True,
    min_component_px:  int   = 50,
    max_hole_px:       int   = 100,
    close_radius:      int   = 2,
    open_radius:       int   = 0,
    bridge_gap_px:     int   = 0,
) -> np.ndarray:
    """
    Full postprocessing pipeline for vessel probability maps.

    Steps (in order):
      1. Thresholding (Otsu or fixed)
      2. Remove small noise components
      3. Fill small holes
      4. Morphological close / open
      5. Bridge small gaps (optional)

    Args:
        prob:             (H, W) float32 sigmoid output in [0, 1]
        threshold:        fixed threshold (used if use_otsu=False)
        use_otsu:         use Otsu adaptive thresholding
        min_component_px: remove components smaller than this
        max_hole_px:      fill holes smaller than this
        close_radius:     morphological closing radius (0 = skip)
        open_radius:      morphological opening radius (0 = skip)
        bridge_gap_px:    vessel gap bridging radius (0 = skip)

    Returns:
        (H, W) float32 binary mask {0.0, 1.0}
    """
    mask = threshold_otsu(prob) if use_otsu else threshold_fixed(prob, threshold)

    if min_component_px > 0:
        mask = remove_small_components(mask, min_component_px)

    if max_hole_px > 0:
        mask = fill_vessel_holes(mask, max_hole_px)

    if close_radius > 0 or open_radius > 0:
        mask = morphological_cleanup(mask, close_radius, open_radius)

    if bridge_gap_px > 0:
        mask = bridge_gaps(mask, bridge_gap_px)

    return mask


# ─── Test-time augmentation (TTA) ────────────────────────────────────────────

def tta_predict(
    model,
    image: np.ndarray,
    image_size: int = 512,
    use_flips:   bool = True,
    device: str  = "cpu",
    normalize:   bool = True,
) -> np.ndarray:
    """
    Test-time augmentation for a single retinal image.

    Averages predictions over:
      - Original image
      - Horizontal flip
      - Vertical flip
      - 90° rotation
      - 180° rotation
      - 270° rotation
      (if use_flips=True: all 8 D4 symmetry group members)

    Args:
        model:      AttentionUNet (in eval mode)
        image:      (H, W, 3) uint8 RGB image
        image_size: target size for resizing
        use_flips:  include flip variants
        device:     torch device string
        normalize:  apply ImageNet normalisation

    Returns:
        (H, W) float32 probability map averaged over all variants
    """
    import torch
    from rapid_twin.datasets.retinal_datasets import _img_to_tensor

    img_resized = cv2.resize(image, (image_size, image_size))

    def _variants(img):
        vs = [img]
        if use_flips:
            vs.append(np.fliplr(img).copy())
            vs.append(np.flipud(img).copy())
            vs.append(np.rot90(img, 1).copy())
            vs.append(np.rot90(img, 2).copy())
            vs.append(np.rot90(img, 3).copy())
            vs.append(np.flipud(np.fliplr(img)).copy())
            vs.append(np.fliplr(np.rot90(img, 1)).copy())
        return vs

    def _invert(pred: np.ndarray, idx: int, orig_img: np.ndarray) -> np.ndarray:
        if idx == 0: return pred
        if idx == 1: return np.fliplr(pred).copy()
        if idx == 2: return np.flipud(pred).copy()
        if idx == 3: return np.rot90(pred, -1).copy()
        if idx == 4: return np.rot90(pred, -2).copy()
        if idx == 5: return np.rot90(pred, -3).copy()
        if idx == 6: return np.fliplr(np.flipud(pred)).copy()
        if idx == 7: return np.rot90(np.fliplr(pred), -1).copy()
        return pred

    variants = _variants(img_resized)
    preds    = []

    model.eval()
    with torch.no_grad():
        for i, v in enumerate(variants):
            t = _img_to_tensor(v, normalize=normalize).unsqueeze(0).to(device)
            out = model(t)
            prob = out["segmentation"].squeeze().cpu().numpy()
            preds.append(_invert(prob, i, v))

    return np.mean(preds, axis=0)
