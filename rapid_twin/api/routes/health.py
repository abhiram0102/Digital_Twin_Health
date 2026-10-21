"""
Health / status routes for the RAPiD-Twin API.
"""

from fastapi import APIRouter
from ..schemas import HealthCheckResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthCheckResponse, summary="Service health check")
async def health_check(request=None):
    """Returns service status, model load state, and device info."""
    from fastapi import Request
    try:
        engine = request.app.state.engine
        return HealthCheckResponse(
            status           = "ok",
            model_loaded     = engine is not None and engine.model is not None,
            seg_model_loaded = engine is not None and engine.seg_model is not None,
            device           = str(engine.device) if engine else "unknown",
        )
    except Exception:
        return HealthCheckResponse(
            status           = "degraded",
            model_loaded     = False,
            seg_model_loaded = False,
            device           = "unknown",
        )


@router.get("/", summary="Root")
async def root():
    return {
        "service":     "RAPiD-Twin API",
        "version":     "1.0.0",
        "description": "Retina-Anchored Physiological Digital Twin",
        "docs":        "/docs",
    }
