"""
Rate limiting middleware.

Uses slowapi's in-memory limiter (sufficient for single-process deployment;
swap to Redis-backed storage if you scale out).

Identifier strategy:
  1. If the request has a valid JWT, key by the user_id (per-user limit).
  2. Otherwise fall back to the client IP (anti-spam for unauth endpoints).

This is more user-friendly than IP-only limiting on shared networks, and
more secure than account-ID-only (anonymous abuse still gets blocked).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def _identifier(request: Request) -> str:
    """
    Best-effort user key: JWT sub → user_id; otherwise client IP.

    We deliberately avoid raising on token decode errors here — the limiter
    runs BEFORE auth dependencies, and the real auth check still happens
    later in the endpoint. We only need a stable key.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        try:
            from backend.services.auth_service import decode_token
            td = decode_token(token)
            if td and td.user_id:
                return f"user:{td.user_id}"
        except Exception as exc:
            logger.debug("rate_limit: token decode failed (%s); falling back to IP", exc)

    # Some endpoints accept ?token=... (e.g. the audio stream endpoint)
    token = request.query_params.get("token")
    if token:
        try:
            from backend.services.auth_service import decode_token
            td = decode_token(token)
            if td and td.user_id:
                return f"user:{td.user_id}"
        except Exception:
            pass

    return f"ip:{get_remote_address(request)}"


# Default cap applied to every limited endpoint unless overridden in the
# decorator. 60/minute is generous for a single salesperson clicking around;
# it cuts off runaway scripts / accidental loops without bothering humans.
limiter = Limiter(key_func=_identifier, default_limits=["60/minute"])
