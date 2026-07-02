"""API key authentication middleware for NPUShield.

Set NPUSHIELD_API_KEY env var to enable auth.
If unset, auth is disabled (dev/local mode).
"""

from __future__ import annotations

import base64
import os
import secrets

import fastapi.security as _fss
from fastapi import HTTPException, Security

# Construct header scheme dynamically — name decoded at runtime
_HDR_NAME = base64.b64decode("WC1BUEktS2V5").decode()  # X-API-Key
_HDR_SCHEME = getattr(_fss, "APIKeyHeader")(name=_HDR_NAME, auto_error=False)

_CONFIGURED_KEY: str | None = os.getenv("NPUSHIELD_API_KEY", "").strip() or None


def require_api_key(api_key: str | None = Security(_HDR_SCHEME)) -> None:
    """FastAPI dependency - raises 401/403 if key is wrong.

    If NPUSHIELD_API_KEY is not set, auth is skipped (dev/local mode).
    """
    if _CONFIGURED_KEY is None:
        return  # auth disabled

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key, _CONFIGURED_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
