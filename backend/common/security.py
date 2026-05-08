"""Shared HTTP security helpers for FLPG services."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from common.config import VALID_API_KEYS


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def is_valid_api_key(api_key: str | None, valid_keys: list[str] | None = None) -> bool:
    """Validate API keys without leaking timing differences between candidates."""
    if not api_key:
        return False

    candidates = valid_keys if valid_keys is not None else VALID_API_KEYS
    return any(
        hmac.compare_digest(api_key, candidate)
        for candidate in candidates
        if candidate
    )


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """FastAPI dependency for services that accept administrative writes."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
