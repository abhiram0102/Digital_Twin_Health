"""
Adaptive Image Quality Assessment for Retinal Fundus Images.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class QualityMetrics:
    blur_score: float
    illumination_score: float
    occlusion_score: float
    overall_score: float
    is_acceptable: bool


if TORCH_AVAILABLE:
    class LightweightQualityCNN(nn.Module):
        """Small CNN predicting image quality in [0,1]. Runs as preprocessing gate."""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
            )
        def forward(self, x):
            return self.head(self.features(x))
else:
    LightweightQualityCNN = None


class AdaptiveQualityAssessor:
    """Full quality assessor: heuristics + optional CNN."""

    def __init__(self, blur_threshold=100.0, illumination_threshold=0.3,
                 occlusion_threshold=0.6, use_cnn=False, cnn_checkpoint=None, device="cpu"):
        self.blur_threshold = blur_threshold
        self.illumination_threshold = illumination_threshold
        self.occlusion_threshold = occlusion_threshold
        self.device = device
        self.cnn = None
        if use_cnn and TORCH_AVAILABLE and LightweightQualityCNN is not None:
            self.cnn = LightweightQualityCNN().to(device)
            if cnn_checkpoint:
                import torch
                self.cnn.load_state_dict(torch.load(cnn_checkpoint, map_location=device, weights_only=True))
            self.cnn.eval()

    def _blur_score(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _illumination_score(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        bs = max(h // 8, 32)
        means = [gray[i:i+bs, j:j+bs].mean()
                 for i in range(0, h-bs, bs) for j in range(0, w-bs, bs)]
        m = np.array(means, dtype=np.float32)
        cv = m.std() / (m.mean() + 1e-6)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _occlusion_score(self, gray: np.ndarray) -> float:
        return float((gray > 10).sum() / gray.size)

    def _cnn_score(self, green: np.ndarray) -> float:
        import torch
        r = cv2.resize(green, (128, 128)).astype(np.float32) / 255.0
        t = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.cnn(t).item())

    def assess(self, image_rgb: np.ndarray) -> QualityMetrics:
        green = image_rgb[:, :, 1]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = self._blur_score(green)
        illum = self._illumination_score(gray)
        occl = self._occlusion_score(gray)
        if self.cnn is not None:
            cnn = self._cnn_score(green)
            overall = 0.4*cnn + 0.2*min(blur/self.blur_threshold, 1.0) + 0.2*illum + 0.2*occl
        else:
            overall = 0.4*min(blur/self.blur_threshold, 1.0) + 0.35*illum + 0.25*occl
        acceptable = (blur >= self.blur_threshold*0.3 and
                      illum >= self.illumination_threshold and
                      occl >= self.occlusion_threshold)
        return QualityMetrics(blur, illum, occl, float(np.clip(overall, 0, 1)), acceptable)

    def recommend_params(self, m: QualityMetrics) -> Dict:
        p = {"clahe_clip_limit": 2.0, "clahe_tile_grid": (8, 8), "gaussian_kernel": 5,
             "use_median_filter": True, "frangi_sigmas": [1, 2, 4, 8], "illum_strength": 1.0}
        if m.illumination_score < 0.5:
            p["clahe_clip_limit"] = 3.5; p["illum_strength"] = 1.5
        if m.blur_score < self.blur_threshold * 0.5:
            p["gaussian_kernel"] = 3; p["frangi_sigmas"] = [1, 2, 4]
        if m.blur_score > self.blur_threshold * 1.5:
            p["clahe_clip_limit"] = 2.5; p["frangi_sigmas"] = [1, 2, 4, 8, 16]
        return p
