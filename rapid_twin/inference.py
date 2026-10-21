#!/usr/bin/env python3
"""
RAPiD-Twin Inference Engine.
Processes a single patient's retinal image + physiological signals
and returns a complete health assessment with explainability.

Security notes:
    torch.load uses weights_only=True (requires PyTorch >= 2.1) to prevent
    arbitrary code execution via pickle in checkpoint files.

AV classification:
    If rapid_twin.av_classification.av_classifier is available at runtime,
    AVClassifier is instantiated once in __init__ and run after graph
    construction to provide per-node artery/vein labels.  The AV statistics
    (artery_fraction, vein_fraction, uncertain_fraction, mean_av_confidence)
    are appended to the vascular_graph section of the result dict.
    AV failure is non-fatal: the exception is caught and av_stats is empty.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Union

from rapid_twin.configs.config import DEFAULT_CONFIG
from rapid_twin.models.rapid_twin_model import RAPiDTwin
from rapid_twin.preprocessing.pipeline import AdaptiveRetinalPreprocessor
from rapid_twin.segmentation.attention_unet import AttentionUNet
from rapid_twin.graph.graph_builder import RetinalGraphBuilder
from rapid_twin.explainability.gradcam import ExplainabilityEngine
from rapid_twin.utils.visualization import format_prediction_report


class RAPiDTwinInference:
    """
    End-to-end inference engine for RAPiD-Twin.
    Accepts raw fundus image + optional physiological signals.
    Returns structured health assessment.
    """

    def __init__(self, model_checkpoint: Optional[str] = None,
                 seg_checkpoint: Optional[str] = None,
                 device: str = "auto", image_size: int = 512):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.cfg = DEFAULT_CONFIG
        self.image_size = image_size

        # Build and load models
        self.model = RAPiDTwin(self.cfg).to(self.device)
        self.seg_model = AttentionUNet(
            in_channels=3, out_channels=1,
            features=[64, 128, 256, 512]
        ).to(self.device)

        if model_checkpoint and Path(model_checkpoint).exists():
            ckpt = torch.load(model_checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"Loaded RAPiD-Twin from {model_checkpoint}")

        if seg_checkpoint and Path(seg_checkpoint).exists():
            ckpt = torch.load(seg_checkpoint, map_location=self.device, weights_only=True)
            self.seg_model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"Loaded segmentation model from {seg_checkpoint}")

        self.model.eval()
        self.seg_model.eval()

        # Preprocessing and graph builder
        self.preprocessor = AdaptiveRetinalPreprocessor(
            target_size=(image_size, image_size), device=str(self.device)
        )
        self.graph_builder = RetinalGraphBuilder(min_branch_length=10, max_nodes=512)

        # Explainability engine
        self.explainer = ExplainabilityEngine(self.model, self.seg_model, device=str(self.device))

        # AV classifier — optional, degrades gracefully if unavailable
        self._av_classifier   = None
        self._av_available    = False
        self._av_label_names  = {0: "ARTERY", 1: "VEIN", 2: "UNCERTAIN"}
        try:
            from rapid_twin.av_classification.av_classifier import AVClassifier
            self._av_classifier = AVClassifier(
                node_in_dim = 16,
                edge_in_dim = 8,
                hidden_dim  = 64,
                n_heads     = 4,
                n_layers    = 3,
            ).to(self.device)
            self._av_classifier.eval()
            self._av_available = True
        except Exception as exc:
            print(f"AV classifier not available (non-fatal): {exc}")

        # Digital twin state (patient session)
        self._twin_state: Optional[torch.Tensor] = None
        self._state_history = []

    # ── AV classification helper ───────────────────────────────────────────────

    def _run_av_classification(
        self,
        nf: torch.Tensor,   # (N, 16)
        ei: torch.Tensor,   # (2, 2E)
        ef: torch.Tensor,   # (2E, 8)
    ) -> Dict:
        """
        Run AV classification on the vascular graph.

        Returns a dict with per-class fractions and mean confidence,
        or an empty dict when AV classification is unavailable or fails.
        """
        if not self._av_available or self._av_classifier is None:
            return {}
        if nf.shape[0] < 3:
            return {}   # too few nodes for meaningful classification

        try:
            with torch.no_grad():
                av_logits, av_conf = self._av_classifier(nf, ei, ef)
                # av_logits: (N, 3), av_conf: (N, 1)

            probs      = torch.softmax(av_logits, dim=-1).cpu().numpy()   # (N, 3)
            av_labels  = probs.argmax(axis=1)                              # (N,) in {0,1,2}
            n_nodes    = max(len(av_labels), 1)
            mean_conf  = float(av_conf.mean().item())

            return {
                "artery_fraction":    round(float((av_labels == 0).sum() / n_nodes), 4),
                "vein_fraction":      round(float((av_labels == 1).sum() / n_nodes), 4),
                "uncertain_fraction": round(float((av_labels == 2).sum() / n_nodes), 4),
                "mean_av_confidence": round(mean_conf, 4),
            }
        except Exception as exc:
            print(f"AV classification failed (non-fatal): {exc}")
            return {}

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def process_image(self, image_path: Union[str, np.ndarray]) -> Dict:
        """
        Full pipeline for one retinal scan.

        Args:
            image_path: path to fundus image OR numpy array (H, W, 3) RGB

        Returns:
            Complete health assessment dict
        """
        # Load image
        if isinstance(image_path, np.ndarray):
            image_rgb = image_path
        else:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: Adaptive preprocessing
        processed = self.preprocessor.preprocess(image_rgb)
        quality = processed.quality_metrics

        print(f"\nImage quality: {quality.overall_score:.2f} "
              f"(blur={quality.blur_score:.1f}, illum={quality.illumination_score:.2f})")
        if not quality.is_acceptable:
            print("Warning: Image quality below threshold — results may be less reliable")

        img_tensor = self.preprocessor.to_tensor(processed).unsqueeze(0).to(self.device)

        # Step 2: Vessel segmentation
        with torch.no_grad():
            seg_out = self.seg_model(img_tensor)
            vessel_mask = seg_out["segmentation"].squeeze().cpu().numpy()

        print(f"Vessel coverage: {vessel_mask.mean():.3f}")

        # Step 3: Vascular graph construction
        vessel_map = processed.vessel_enhanced
        graph = self.graph_builder.build(vessel_mask, vessel_map)
        print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        # Prepare graph tensors
        has_graph = len(graph.nodes) > 0
        nf = torch.from_numpy(graph.node_features).to(self.device) if has_graph \
             else torch.zeros(1, 16, device=self.device)
        ei = torch.from_numpy(graph.edge_index).long().to(self.device) if has_graph \
             and graph.edge_index.shape[1] > 0 \
             else torch.zeros(2, 0, dtype=torch.long, device=self.device)
        ef = torch.from_numpy(graph.edge_features).to(self.device) if has_graph \
             and len(graph.edges) > 0 \
             else torch.zeros(0, 8, device=self.device)

        # Step 3b: AV classification (optional)
        av_stats = self._run_av_classification(nf, ei, ef)

        # Step 4: Full model forward pass
        with torch.no_grad():
            outputs = self.model(
                retinal_image     = img_tensor,
                node_features     = nf if has_graph else None,
                edge_index        = ei if has_graph else None,
                edge_features     = ef if has_graph else None,
                historical_states = self._twin_state.unsqueeze(0)
                                    if self._twin_state is not None else None,
            )

        # Update digital twin state
        new_state = outputs["updated_state"].squeeze(0).detach()
        self._twin_state = new_state
        self._state_history.append(new_state.cpu().numpy())

        # Step 5: Build result
        vascular_graph_info = {
            "num_nodes":        len(graph.nodes),
            "num_edges":        len(graph.edges),
            "bifurcations":     sum(1 for n in graph.nodes if n.is_bifurcation),
            "endpoints":        sum(1 for n in graph.nodes if n.is_endpoint),
            "mean_tortuosity":  float(np.mean([n.tortuosity for n in graph.nodes])
                                      if graph.nodes else 0),
            "mean_diameter":    float(np.mean([e.mean_diameter for e in graph.edges])
                                      if graph.edges else 0),
        }
        if av_stats:
            vascular_graph_info.update(av_stats)

        result = {
            "quality_metrics": {
                "blur_score":         quality.blur_score,
                "illumination_score": quality.illumination_score,
                "occlusion_score":    quality.occlusion_score,
                "overall_score":      quality.overall_score,
                "is_acceptable":      quality.is_acceptable,
            },
            "vascular_graph": vascular_graph_info,
            "disease_risks":  {},
            "health_state":   {},
            "twin_update": {
                "anomaly_score": float(outputs["anomaly_score"][0].item())
                                 if outputs.get("anomaly_score") is not None else 0.0,
                "update_index":  len(self._state_history),
            },
        }

        # Disease predictions
        preds = outputs["disease_predictions"]
        for task, logits in preds.items():
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            result["disease_risks"][task] = {
                "predicted_class": int(proba.argmax()),
                "probabilities":   proba.tolist(),
                "max_probability": float(proba.max()),
                "risk_level":      self._risk_level(proba),
            }

        # Health state scores
        if outputs.get("health_scores"):
            for dim, score in outputs["health_scores"].items():
                if isinstance(score, torch.Tensor) and score.shape[-1] == 1:
                    result["health_state"][dim] = float(score[0].item())

        # Modality weights
        if outputs.get("modality_weights") is not None:
            mw = outputs["modality_weights"][0].cpu().numpy()
            result["modality_weights"] = {
                "vascular_graph": float(mw[0]),
                "retinal_image":  float(mw[1]),
                "physiological":  float(mw[2]) if len(mw) > 2 else 0.0,
            }

        print(format_prediction_report(outputs))
        return result

    def process_with_physio(self, image_path: Union[str, np.ndarray],
                             physio_signals: np.ndarray) -> Dict:
        """
        Full pipeline with physiological signals in a single forward pass.

        physio_signals: (T, 8) array — HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, steps

        A single forward pass includes image + graph + physio together,
        avoiding the double-preprocessing cost of the previous two-pass approach.
        """
        physio_tensor = torch.from_numpy(
            physio_signals.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        # ── Step 1: load & preprocess once ────────────────────────────────────
        if isinstance(image_path, np.ndarray):
            image_rgb = image_path
        else:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        processed  = self.preprocessor.preprocess(image_rgb)
        quality    = processed.quality_metrics

        print(f"\nImage quality: {quality.overall_score:.2f} "
              f"(blur={quality.blur_score:.1f}, illum={quality.illumination_score:.2f})")
        if not quality.is_acceptable:
            print("Warning: Image quality below threshold — results may be less reliable")

        img_tensor = self.preprocessor.to_tensor(processed).unsqueeze(0).to(self.device)

        # ── Step 2: vessel segmentation ────────────────────────────────────────
        with torch.no_grad():
            seg_out     = self.seg_model(img_tensor)
            vessel_mask = seg_out["segmentation"].squeeze().cpu().numpy()

        print(f"Vessel coverage: {vessel_mask.mean():.3f}")

        # ── Step 3: vascular graph ─────────────────────────────────────────────
        vessel_map = processed.vessel_enhanced
        graph      = self.graph_builder.build(vessel_mask, vessel_map)
        print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        has_graph = len(graph.nodes) > 0
        nf = torch.from_numpy(graph.node_features).to(self.device) if has_graph \
             else torch.zeros(1, 16, device=self.device)
        ei = torch.from_numpy(graph.edge_index).long().to(self.device) \
             if (has_graph and graph.edge_index.shape[1] > 0) \
             else torch.zeros(2, 0, dtype=torch.long, device=self.device)
        ef = torch.from_numpy(graph.edge_features).to(self.device) \
             if (has_graph and len(graph.edges) > 0) \
             else torch.zeros(0, 8, device=self.device)

        # ── Step 3b: AV classification (optional) ─────────────────────────────
        av_stats = self._run_av_classification(nf, ei, ef)

        # ── Step 4: single forward pass — image + graph + physio ──────────────
        with torch.no_grad():
            outputs = self.model(
                retinal_image     = img_tensor,
                node_features     = nf if has_graph else None,
                edge_index        = ei if has_graph else None,
                edge_features     = ef if has_graph else None,
                physio_signals    = physio_tensor,
                historical_states = self._twin_state.unsqueeze(0)
                                    if self._twin_state is not None else None,
            )

        # ── Step 5: update twin state ──────────────────────────────────────────
        new_state = outputs["updated_state"].squeeze(0).detach()
        self._twin_state = new_state
        self._state_history.append(new_state.cpu().numpy())

        # ── Step 6: build result ───────────────────────────────────────────────
        vascular_graph_info = {
            "num_nodes":        len(graph.nodes),
            "num_edges":        len(graph.edges),
            "bifurcations":     sum(1 for n in graph.nodes if n.is_bifurcation),
            "endpoints":        sum(1 for n in graph.nodes if n.is_endpoint),
            "mean_tortuosity":  float(np.mean([n.tortuosity for n in graph.nodes])
                                      if graph.nodes else 0),
            "mean_diameter":    float(np.mean([e.mean_diameter for e in graph.edges])
                                      if graph.edges else 0),
        }
        if av_stats:
            vascular_graph_info.update(av_stats)

        result = {
            "quality_metrics": {
                "blur_score":         quality.blur_score,
                "illumination_score": quality.illumination_score,
                "occlusion_score":    quality.occlusion_score,
                "overall_score":      quality.overall_score,
                "is_acceptable":      quality.is_acceptable,
            },
            "vascular_graph": vascular_graph_info,
            "disease_risks":  {},
            "health_state":   {},
            "twin_update": {
                "anomaly_score": float(outputs["anomaly_score"][0].item())
                                 if outputs.get("anomaly_score") is not None else 0.0,
                "update_index":  len(self._state_history),
            },
        }

        preds = outputs["disease_predictions"]
        for task, logits in preds.items():
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            result["disease_risks"][task] = {
                "predicted_class": int(proba.argmax()),
                "probabilities":   proba.tolist(),
                "max_probability": float(proba.max()),
                "risk_level":      self._risk_level(proba),
            }

        if outputs.get("health_scores"):
            for dim, score in outputs["health_scores"].items():
                if isinstance(score, torch.Tensor) and score.shape[-1] == 1:
                    result["health_state"][dim] = float(score[0].item())

        if outputs.get("modality_weights") is not None:
            mw = outputs["modality_weights"][0].cpu().numpy()
            result["modality_weights"] = {
                "vascular_graph": float(mw[0]),
                "retinal_image":  float(mw[1]),
                "physiological":  float(mw[2]) if len(mw) > 2 else 0.0,
            }

        from rapid_twin.utils.visualization import format_prediction_report
        print(format_prediction_report(outputs))
        return result

    def explain_prediction(self, image_rgb: np.ndarray,
                            target_task: str = "diabetic_retinopathy") -> Dict:
        """Generate full explainability report."""
        processed  = self.preprocessor.preprocess(image_rgb)
        img_tensor = self.preprocessor.to_tensor(processed).unsqueeze(0).to(self.device)
        return self.explainer.explain(img_tensor, target_task=target_task)

    def reset_twin_state(self):
        """Reset digital twin state (new patient session).

        Clears both the in-memory state variable *and* the model's persistent
        ring buffer — without this the ring buffer keeps accumulating states
        from previous patients even after the high-level state pointer is reset.
        """
        self._twin_state = None
        self._state_history = []
        self.model.reset_memory()
        print("Digital twin state reset.")

    @staticmethod
    def _risk_level(proba: np.ndarray) -> str:
        cls   = proba.argmax()
        n     = len(proba)
        ratio = cls / max(n - 1, 1)
        if ratio < 0.25:
            return "LOW"
        elif ratio < 0.5:
            return "MILD"
        elif ratio < 0.75:
            return "MODERATE"
        return "HIGH"

    def get_twin_state_history(self) -> np.ndarray:
        """Return history of health state vectors (N_visits, state_dim)."""
        if not self._state_history:
            return np.array([])
        return np.array(self._state_history)


def demo():
    """Quick inference demo with a synthetic fundus image."""
    print("\n" + "="*60)
    print("RAPiD-Twin Demo — Synthetic Patient")
    print("="*60)

    # Create synthetic fundus image with realistic vessel structure
    np.random.seed(42)
    import cv2
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Dark background
    img[:] = (40, 30, 20)
    # Fundus circular field
    cv2.circle(img, (256, 256), 245, (70, 120, 60), -1)
    # Optic disc
    cv2.circle(img, (320, 256), 28, (220, 200, 150), -1)
    cv2.circle(img, (320, 256), 14, (240, 230, 180), -1)
    # Major vessel tree from optic disc
    vessels = [
        [(320, 256), (256, 200), (180, 160), (120, 140)],
        [(320, 256), (256, 300), (180, 340), (120, 370)],
        [(320, 256), (380, 200), (440, 170), (490, 150)],
        [(320, 256), (380, 310), (440, 350), (490, 380)],
        [(256, 200), (220, 160), (180, 130)],
        [(256, 300), (220, 340), (180, 370)],
    ]
    for path in vessels:
        for i in range(len(path) - 1):
            thickness = max(1, 4 - i)
            cv2.line(img, path[i], path[i+1], (50, 100, 50), thickness * 2)
            cv2.line(img, path[i], path[i+1], (80, 160, 80), thickness)
    # Fovea (dark centre)
    cv2.circle(img, (200, 256), 18, (20, 15, 10), -1)
    # Slight noise for realism
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Synthetic physio signals: (24 hours, 8 channels)
    physio = np.column_stack([
        np.random.normal(72,   8,    24),   # HR
        np.random.normal(125,  12,   24),   # BP sys
        np.random.normal(82,   8,    24),   # BP dia
        np.random.normal(97.5, 1,    24),   # SpO2
        np.random.normal(15,   2,    24),   # RR
        np.random.normal(36.8, 0.3,  24),   # Temp
        np.random.normal(55,   15,   24),   # HRV
        np.random.normal(7000, 2000, 24),   # Steps/hr
    ]).astype(np.float32)

    engine = RAPiDTwinInference(device="cpu", image_size=256)
    result = engine.process_with_physio(img, physio)

    print("\n\nVascular Graph Summary:")
    for k, v in result["vascular_graph"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nDisease Risk Summary:")
    for task, risk in result["disease_risks"].items():
        print(f"  {task}: {risk['risk_level']} (p={risk['max_probability']:.3f})")

    print("\nHealth State:")
    for dim, val in result["health_state"].items():
        print(f"  {dim}: {val:.3f}")

    print("\nDemo complete")
    return result


if __name__ == "__main__":
    demo()
