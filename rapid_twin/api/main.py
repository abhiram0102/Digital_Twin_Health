"""
RAPiD-Twin FastAPI Production Backend.

Startup:
  uvicorn rapid_twin.api.main:app --host 0.0.0.0 --port 8000 --workers 1

Or via the CLI helper:
  python -m rapid_twin.api.main

Routes:
  GET  /                    — service info
  GET  /health              — health check
  POST /infer/image         — analyse retinal image
  POST /infer/image+physio  — image + physiological signals
  POST /infer/explain       — explainability report
  POST /infer/reset         — reset digital twin state
  GET  /infer/twin/state    — twin memory statistics
  GET  /docs                — Swagger UI (auto-generated)
  GET  /redoc               — ReDoc UI

CORS:
  Set RAPID_TWIN_CORS_ORIGINS to a comma-separated list of allowed origins.
  Example: RAPID_TWIN_CORS_ORIGINS="https://app.example.com,http://localhost:3000"
  When unset, the wildcard "*" is used (open access — dev only).
  NOTE: allow_credentials=True requires explicit origins (not "*") per the CORS
  spec (https://fetch.spec.whatwg.org/#cors-check). This is enforced automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("rapid_twin.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")


# ─── Lifespan (model loading) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; clean up on shutdown."""
    cfg = app.state.api_cfg
    logger.info("Loading RAPiD-Twin inference engine …")

    try:
        from rapid_twin.inference import RAPiDTwinInference
        engine = RAPiDTwinInference(
            model_checkpoint = cfg.get("model_checkpoint") or None,
            seg_checkpoint   = cfg.get("seg_checkpoint")   or None,
            device           = cfg.get("device", "auto"),
            image_size       = int(cfg.get("image_size", 512)),
        )
        app.state.engine = engine
        logger.info("Inference engine ready. Device: %s", engine.device)
    except Exception as exc:
        logger.error("Failed to load inference engine: %s", exc)
        app.state.engine = None

    yield   # ← application runs here

    logger.info("Shutting down RAPiD-Twin API.")
    app.state.engine = None


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(
    model_checkpoint: Optional[str] = None,
    seg_checkpoint:   Optional[str] = None,
    device:           str = "auto",
    image_size:       int = 512,
    max_upload_mb:    int = 20,
    allow_origins:    Optional[List[str]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title       = "RAPiD-Twin API",
        description = "Retina-Anchored Physiological Digital Twin — clinical health assessment",
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # Store config in app state (accessible inside lifespan)
    app.state.api_cfg = {
        "model_checkpoint": model_checkpoint or os.getenv("RAPID_TWIN_CHECKPOINT", ""),
        "seg_checkpoint":   seg_checkpoint   or os.getenv("RAPID_SEG_CHECKPOINT", ""),
        "device":           device,
        "image_size":       image_size,
    }
    app.state.engine = None

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Resolve origins: caller arg → env var → open wildcard (dev fallback).
    # The CORS spec forbids allow_credentials=True with a wildcard origin, so we
    # only enable credentials when at least one explicit origin is provided.
    _env_origins: List[str] = [
        o.strip()
        for o in os.getenv("RAPID_TWIN_CORS_ORIGINS", "").split(",")
        if o.strip()
    ]
    _cors_origins: List[str] = allow_origins or _env_origins or ["*"]
    _has_wildcard: bool      = "*" in _cors_origins
    _allow_creds:  bool      = (not _has_wildcard) and bool(_cors_origins)

    if _has_wildcard:
        logger.warning(
            "CORS is open to all origins ('*'). "
            "Set RAPID_TWIN_CORS_ORIGINS to restrict access in production."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = _cors_origins,
        allow_credentials = _allow_creds,   # True only when origins are explicit
        allow_methods     = ["GET", "POST", "OPTIONS"],
        allow_headers     = ["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request size limit ────────────────────────────────────────────────────
    max_bytes = max_upload_mb * 1024 * 1024

    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Upload exceeds {max_upload_mb} MB limit"},
                )
        return await call_next(request)

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    from .routes.health    import router as health_router
    from .routes.inference import router as inference_router

    app.include_router(health_router)
    app.include_router(inference_router)

    return app


# ─── Default app instance ─────────────────────────────────────────────────────

app = create_app(
    model_checkpoint = os.getenv("RAPID_TWIN_CHECKPOINT", ""),
    seg_checkpoint   = os.getenv("RAPID_SEG_CHECKPOINT", ""),
    device           = os.getenv("RAPID_DEVICE", "auto"),
    image_size       = int(os.getenv("RAPID_IMAGE_SIZE", "512")),
    max_upload_mb    = int(os.getenv("RAPID_MAX_UPLOAD_MB", "20")),
)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="RAPiD-Twin API Server")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--workers",    type=int, default=1)
    parser.add_argument("--reload",     action="store_true")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seg-checkpoint", default="")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    os.environ["RAPID_TWIN_CHECKPOINT"] = args.checkpoint
    os.environ["RAPID_SEG_CHECKPOINT"]  = args.seg_checkpoint
    os.environ["RAPID_DEVICE"]          = args.device
    os.environ["RAPID_IMAGE_SIZE"]      = str(args.image_size)

    uvicorn.run(
        "rapid_twin.api.main:app",
        host    = args.host,
        port    = args.port,
        workers = args.workers,
        reload  = args.reload,
        log_level = "info",
    )


if __name__ == "__main__":
    main()
