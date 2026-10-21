"""
API Key Authentication for RAPiD-Twin.

Configuration:
    Set RAPID_TWIN_API_KEYS to a comma-separated list of valid API keys.
    If the env var is unset or empty, the API runs in open-access mode (development).

    Example:
        export RAPID_TWIN_API_KEYS="prod-key-abc123,dev-key-xyz789"

Usage:
    from rapid_twin.api.auth import verify_api_key

    @router.post("/infer/image")
    async def endpoint(
        request: Request,
        file: UploadFile = File(...),
        _: None = Depends(verify_api_key),
    ):
        ...
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("rapid_twin.api.auth")

# ── Key registry ──────────────────────────────────────────────────────────────

_VALID_KEYS: frozenset = frozenset(
    k.strip()
    for k in os.getenv("RAPID_TWIN_API_KEYS", "").split(",")
    if k.strip()
)

_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

if not _VALID_KEYS:
    logger.warning(
        "RAPID_TWIN_API_KEYS is not configured — API is running in open-access "
        "mode. Set RAPID_TWIN_API_KEYS=<key1,key2,...> to require authentication "
        "in production."
    )


# ── Dependency ────────────────────────────────────────────────────────────────

async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_scheme),
) -> None:
    """
    FastAPI security dependency that validates the X-API-Key request header.

    Behavior:
        - RAPID_TWIN_API_KEYS unset / empty  → allow all requests (dev mode).
        - RAPID_TWIN_API_KEYS configured     → reject requests with an invalid key.

    Raises:
        HTTPException(403): when keys are configured and the provided key is absent
                            or not in the valid set.
    """
    if not _VALID_KEYS:
        return  # open-access mode — no keys configured

    if api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Provide a valid X-API-Key header.",
        )
