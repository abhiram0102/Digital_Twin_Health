"""
Inference API routes for RAPiD-Twin.

Endpoints:
  POST /infer/image          — retinal image only
  POST /infer/image+physio   — image + physiological signals
  POST /infer/explain        — explainability report
  POST /infer/reset          — reset digital twin state
  GET  /infer/twin/state     — current twin memory statistics

Security:
  All POST endpoints require a valid X-API-Key header when
  RAPID_TWIN_API_KEYS is configured (see rapid_twin/api/auth.py).

Input validation:
  - Magic-byte check: only JPEG (FF D8 FF) and PNG (89 PNG) files accepted.
  - Dimension cap: images wider or taller than 4096 px are rejected (HTTP 422).
  - patient_id: restricted to [A-Za-z0-9_-], max 64 characters.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..auth import verify_api_key
from ..schemas import (
    DiseaseRiskResponse,
    ExplainabilityResponse,
    ExplainRequest,
    HealthStateResponse,
    InferenceResponse,
    ModalityWeightsResponse,
    QualityMetricsResponse,
    ResetResponse,
    TwinUpdateResponse,
    VascularGraphResponse,
)

router = APIRouter(prefix="/infer", tags=["inference"])

# ── Image validation constants ────────────────────────────────────────────────

_JPEG_MAGIC: bytes = b"\xff\xd8\xff"
_PNG_MAGIC:  bytes = b"\x89PNG"
_MAX_DIMENSION: int = 4096   # px — images beyond this are implausibly large

# ── patient_id allow-list (alphanumeric plus underscore and hyphen, 1–64 chars)
_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_engine(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(503, "Inference engine not initialised — check logs for startup errors")
    return engine


def _validate_image_bytes(data: bytes) -> None:
    """
    Check file magic bytes to ensure the upload is a JPEG or PNG image.

    Raises:
        HTTPException(422): if the file is too small or does not match JPEG/PNG
                            magic bytes.
    """
    if len(data) < 8:
        raise HTTPException(422, "File too small to be a valid image.")
    if not (data[:3] == _JPEG_MAGIC or data[:4] == _PNG_MAGIC):
        raise HTTPException(
            422,
            "File must be a JPEG or PNG image. "
            "Invalid magic bytes — ensure the file was not corrupted or renamed.",
        )


def _decode_image(data: bytes) -> np.ndarray:
    """
    Validate, decode, and dimension-check an uploaded image.

    Steps:
      1. Magic-byte validation (JPEG / PNG only).
      2. OpenCV decode.
      3. Dimension cap (≤ 4096 × 4096).

    Returns:
        (H, W, 3) uint8 RGB array.

    Raises:
        HTTPException(422): on format violation, decode failure, or oversized image.
    """
    _validate_image_bytes(data)

    arr     = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(422, "Cannot decode image — file may be corrupted.")

    h, w = img_bgr.shape[:2]
    if h > _MAX_DIMENSION or w > _MAX_DIMENSION:
        raise HTTPException(
            422,
            f"Image dimensions {w}×{h} exceed the maximum allowed {_MAX_DIMENSION}×{_MAX_DIMENSION}. "
            "Please downsample the image before uploading.",
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _sanitize_patient_id(patient_id: Optional[str]) -> Optional[str]:
    """
    Validate and return patient_id restricted to [A-Za-z0-9_-], max 64 chars.

    Returns None when patient_id is None (field is optional).

    Raises:
        HTTPException(422): when patient_id does not conform to the allow-list.
    """
    if patient_id is None:
        return None
    if not _PATIENT_ID_RE.match(patient_id):
        raise HTTPException(
            422,
            "patient_id must contain only letters, digits, underscores, or hyphens "
            "(max 64 characters). Received: " + repr(patient_id[:80]),
        )
    return patient_id


def _build_response(
    result:     dict,
    patient_id: Optional[str],
    elapsed_ms: float,
) -> InferenceResponse:
    quality = QualityMetricsResponse(**result["quality_metrics"])

    vg = result["vascular_graph"]
    vascular = VascularGraphResponse(
        num_nodes       = int(vg["num_nodes"]),
        num_edges       = int(vg["num_edges"]),
        bifurcations    = int(vg.get("bifurcations", 0)),
        endpoints       = int(vg.get("endpoints", 0)),
        mean_tortuosity = float(vg.get("mean_tortuosity", 0.0)),
        mean_diameter   = float(vg.get("mean_diameter", 0.0)),
    )

    disease_risks = {
        task: DiseaseRiskResponse(
            predicted_class = info["predicted_class"],
            probabilities   = info["probabilities"],
            max_probability = float(info["max_probability"]),
            risk_level      = info["risk_level"],
        )
        for task, info in result.get("disease_risks", {}).items()
    }

    hs = result.get("health_state", {})
    health = HealthStateResponse(
        vascular_health     = hs.get("vascular_health"),
        retinal_integrity   = hs.get("retinal_integrity"),
        metabolic_risk      = hs.get("metabolic_risk"),
        cardiovascular_risk = hs.get("cardiovascular_risk"),
        lifestyle_health    = hs.get("lifestyle_health"),
    )

    tw = result.get("twin_update", {})
    twin = TwinUpdateResponse(
        anomaly_score = float(tw.get("anomaly_score", 0.0)),
        update_index  = int(tw.get("update_index", 0)),
        memory_filled = int(tw.get("memory_filled", 0)),
    )

    mw = result.get("modality_weights")
    modality_weights = ModalityWeightsResponse(
        vascular_graph = float(mw.get("vascular_graph", 0.0)),
        retinal_image  = float(mw.get("retinal_image", 0.0)),
        physiological  = float(mw.get("physiological", 0.0)),
    ) if mw else None

    return InferenceResponse(
        patient_id          = patient_id,
        quality_metrics     = quality,
        vascular_graph      = vascular,
        disease_risks       = disease_risks,
        health_state        = health,
        twin_update         = twin,
        modality_weights    = modality_weights,
        processing_time_ms  = round(elapsed_ms, 2),
    )


def _get_memory_fill(engine) -> int:
    try:
        return int(engine.model.state_tracker._buffer_fill.item())
    except Exception:
        return 0


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/image",
    response_model=InferenceResponse,
    summary="Analyse a retinal fundus image",
)
async def infer_image(
    request:    Request,
    file:       UploadFile = File(..., description="Fundus image (JPEG or PNG, max 20 MB)"),
    patient_id: Optional[str] = Form(None, description="Optional patient identifier"),
    _:          None = Depends(verify_api_key),
):
    """
    Full retinal health assessment from a fundus photograph.

    Returns:
    - Image quality score
    - Vascular graph statistics (including AV classification if available)
    - Disease risk predictions (DR, glaucoma, AMD, RVO, diabetes, HTN, CVD)
    - Digital twin state update
    """
    patient_id = _sanitize_patient_id(patient_id)
    engine     = _get_engine(request)
    t0         = time.time()
    data       = await file.read()
    img        = _decode_image(data)
    result     = engine.process_image(img)
    result.setdefault("twin_update", {})["memory_filled"] = _get_memory_fill(engine)
    return _build_response(result, patient_id, (time.time() - t0) * 1000)


@router.post(
    "/image+physio",
    response_model=InferenceResponse,
    summary="Analyse image fused with physiological signals",
)
async def infer_image_physio(
    request:    Request,
    file:       UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    signals:    str = Form(
        ...,
        description="JSON-encoded (T, 8) array — HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, Steps/hr",
    ),
    _:          None = Depends(verify_api_key),
):
    """
    Analyse retinal image fused with physiological time-series.

    `signals` must be a JSON string of shape (T, 8), e.g.:
    `[[72, 125, 82, 97.5, 15, 36.8, 55, 7000], ...]`
    """
    import json as _json
    patient_id = _sanitize_patient_id(patient_id)
    engine     = _get_engine(request)
    t0         = time.time()
    data       = await file.read()
    img        = _decode_image(data)

    try:
        sig_list  = _json.loads(signals)
        physio_np = np.array(sig_list, dtype=np.float32)
    except Exception:
        raise HTTPException(422, "signals must be a valid JSON array of shape (T, 8)")

    if physio_np.ndim != 2 or physio_np.shape[1] != 8:
        raise HTTPException(
            422,
            f"signals must have shape (T, 8). Got {physio_np.shape}. "
            "Columns: HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, Steps/hr",
        )

    result = engine.process_with_physio(img, physio_np)
    result.setdefault("twin_update", {})["memory_filled"] = _get_memory_fill(engine)
    return _build_response(result, patient_id, (time.time() - t0) * 1000)


@router.post(
    "/explain",
    response_model=ExplainabilityResponse,
    summary="Explainability report (Grad-CAM++, Integrated Gradients, GNNExplainer)",
)
async def explain_prediction(
    request:     Request,
    file:        UploadFile = File(...),
    target_task: str = Form("diabetic_retinopathy"),
    _:           None = Depends(verify_api_key),
):
    """
    Generate a full explainability report for a fundus image prediction.

    Returns spatial Grad-CAM++ heatmap summary, Integrated Gradients importance
    scores, GNNExplainer edge rankings, and modality ablation deltas.
    """
    valid_tasks = {
        "diabetic_retinopathy", "glaucoma", "macular_degeneration",
        "retinal_vein_occlusion", "diabetes", "hypertension", "cardiovascular",
    }
    if target_task not in valid_tasks:
        raise HTTPException(422, f"target_task must be one of {sorted(valid_tasks)}")

    engine = _get_engine(request)
    data   = await file.read()
    img    = _decode_image(data)
    report = engine.explain_prediction(img, target_task=target_task)

    return ExplainabilityResponse(
        task                      = report.get("task", target_task),
        predicted_class           = report.get("predicted_class"),
        prediction_probabilities  = report.get("prediction_probabilities"),
        anomaly_score             = report.get("anomaly_score"),
        gradcam_summary           = report.get("gradcam_summary"),
        intgrad_importance        = report.get("intgrad_importance"),
        physio_importance         = report.get("physio_importance"),
        physio_ablation_delta     = report.get("physio_ablation_delta"),
        gnn_explanation           = report.get("gnn_explanation"),
        health_state_vector       = report.get("health_state_vector"),
    )


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Reset digital twin state for a new patient",
)
async def reset_twin(
    request:    Request,
    patient_id: Optional[str] = Form(None),
    _:          None = Depends(verify_api_key),
):
    """
    Clear the persistent digital twin memory buffer.
    Call this before processing a new patient session.
    """
    patient_id = _sanitize_patient_id(patient_id)
    engine     = _get_engine(request)
    engine.reset_twin_state()
    return ResetResponse(
        patient_id     = patient_id,
        message        = "Digital twin state reset successfully.",
        memory_cleared = True,
    )


@router.get("/twin/state", summary="Current digital twin memory statistics")
async def twin_state(request: Request):
    """Return statistics about the rolling twin state ring buffer."""
    engine = _get_engine(request)
    try:
        tracker = engine.model.state_tracker
        stats   = tracker.get_memory_statistics()
        return {
            "memory_size":          int(tracker.memory_size),
            "buffer_filled":        int(tracker._buffer_fill.item()),
            "has_baseline":         bool(stats),
            "state_history_length": len(engine._state_history),
            "state_mean_norm": float(
                stats["mean"].norm().item()) if "mean" in stats else None,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
