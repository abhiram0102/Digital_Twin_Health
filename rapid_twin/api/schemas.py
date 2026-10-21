"""
Pydantic v2 schemas for the RAPiD-Twin FastAPI backend.

All request / response models live here to keep routes lean.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Requests ─────────────────────────────────────────────────────────────────

class PhysioSignalRequest(BaseModel):
    """
    Physiological signal time-series for one patient.
    Shape: (T, 8) — rows are timesteps, columns are:
      HR, BP_sys, BP_dia, SpO2, RR, Temp, HRV, Steps/hr
    """
    signals: List[List[float]] = Field(
        ...,
        description="(T, 8) physiological signal array — T timesteps × 8 channels",
        examples=[[[72.0, 125.0, 82.0, 97.5, 15.0, 36.8, 55.0, 7000.0]] * 24],
    )
    patient_id: Optional[str] = None

    @field_validator("signals")
    @classmethod
    def validate_shape(cls, v):
        if not v:
            raise ValueError("signals array must not be empty")
        expected_cols = 8
        for row in v:
            if len(row) != expected_cols:
                raise ValueError(
                    f"Each row must have {expected_cols} channels (HR, BP_sys, BP_dia, "
                    f"SpO2, RR, Temp, HRV, Steps). Got {len(row)} columns."
                )
        return v


class ExplainRequest(BaseModel):
    """Request explainability report for a specific task."""
    target_task: str = Field(
        default="diabetic_retinopathy",
        description="Disease task to explain",
    )
    export_visuals: bool = Field(
        default=False,
        description="Export Grad-CAM overlay images to disk",
    )

    @field_validator("target_task")
    @classmethod
    def validate_task(cls, v):
        valid = {
            "diabetic_retinopathy", "glaucoma", "macular_degeneration",
            "retinal_vein_occlusion", "diabetes", "hypertension", "cardiovascular",
        }
        if v not in valid:
            raise ValueError(f"target_task must be one of {sorted(valid)}")
        return v


# ─── Responses ────────────────────────────────────────────────────────────────

class QualityMetricsResponse(BaseModel):
    blur_score:         float
    illumination_score: float
    occlusion_score:    float
    overall_score:      float
    is_acceptable:      bool


class VascularGraphResponse(BaseModel):
    num_nodes:       int
    num_edges:       int
    bifurcations:    int
    endpoints:       int
    mean_tortuosity: float
    mean_diameter:   float


class DiseaseRiskResponse(BaseModel):
    predicted_class:  int
    probabilities:    List[float]
    max_probability:  float
    risk_level:       str = Field(
        description="LOW | MILD | MODERATE | HIGH"
    )


class HealthStateResponse(BaseModel):
    vascular_health:    Optional[float] = None
    retinal_integrity:  Optional[float] = None
    metabolic_risk:     Optional[float] = None
    cardiovascular_risk: Optional[float] = None
    lifestyle_health:   Optional[float] = None


class TwinUpdateResponse(BaseModel):
    anomaly_score:  float
    update_index:   int
    memory_filled:  int = Field(description="Number of states in the ring buffer")


class ModalityWeightsResponse(BaseModel):
    vascular_graph: float = 0.0
    retinal_image:  float = 0.0
    physiological:  float = 0.0


class InferenceResponse(BaseModel):
    patient_id:          Optional[str]            = None
    quality_metrics:     QualityMetricsResponse
    vascular_graph:      VascularGraphResponse
    disease_risks:       Dict[str, DiseaseRiskResponse]
    health_state:        HealthStateResponse
    twin_update:         TwinUpdateResponse
    modality_weights:    Optional[ModalityWeightsResponse] = None
    processing_time_ms:  float


class ExplainabilityResponse(BaseModel):
    task:                  str
    predicted_class:       Optional[int]            = None
    prediction_probabilities: Optional[List[float]] = None
    anomaly_score:         Optional[float]           = None
    gradcam_summary:       Optional[Dict]            = None
    intgrad_importance:    Optional[float]           = None
    physio_importance:     Optional[float]           = None
    physio_ablation_delta: Optional[float]           = None
    gnn_explanation:       Optional[Dict]            = None
    health_state_vector:   Optional[Dict[str, float]] = None
    export_paths:          Optional[Dict[str, str]]  = None


class HealthCheckResponse(BaseModel):
    status:          str
    model_loaded:    bool
    seg_model_loaded: bool
    device:          str
    version:         str = "1.0.0"


class ResetResponse(BaseModel):
    patient_id:     Optional[str]
    message:        str
    memory_cleared: bool
