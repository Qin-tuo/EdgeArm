"""Optional API-key auth for public voice endpoints.

Disabled when ``OVS_API_KEYS`` is unset, empty, whitespace, or
comma-only. When enabled, public voice endpoints require either
``Authorization: Bearer <key>`` (HTTP and WS) or, for WebSockets only,
``?token=<key>``. Admin endpoints keep their own auth (see
``server/core/admin_auth.py``); probes (``/health``, ``/livez``, ``/readyz``)
stay open.

Env is read on every check (mirrors ``admin_auth._admin_key``) so
operators can rotate keys without a restart; only new requests/sessions
see the new value.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Iterable

from fastapi import HTTPException, Request, WebSocket, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _parse_keys(raw: str | None) -> list[str]:
    """Parse the comma-separated ``OVS_API_KEYS`` env value.

    Strips whitespace around each entry, drops empty entries, allows
    duplicates. Returns ``[]`` when auth should be disabled.
    """
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _configured_keys() -> list[str]:
    return _parse_keys(os.environ.get("OVS_API_KEYS"))


def is_enabled() -> bool:
    """Return True when API-key auth is active for public voice endpoints."""
    return len(_configured_keys()) > 0


def _matches(candidate: str, keys: Iterable[str]) -> bool:
    """Constant-time match against any configured key."""
    if not candidate:
        return False
    matched = False
    for k in keys:
        # Always run compare_digest to keep timing uniform.
        if hmac.compare_digest(candidate, k):
            matched = True
    return matched


# ---------------------------------------------------------------------------
# Credential extraction
# ---------------------------------------------------------------------------

def _extract_bearer(authorization: str | None) -> str | None:
    """Return the bearer token from a raw ``Authorization`` header value.

    Returns ``None`` when the header is missing or not a Bearer scheme.
    Returns an empty string when the scheme is Bearer but the token is
    empty (caller treats this as failure when auth is enabled).
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return None
    return token.strip()


# Cap input length defensively: an attacker (or buggy client) could
# submit a multi-megabyte Authorization header. We never want to hash
# arbitrary-size data on the auth-rejection path. The SHA-256 of any
# 4 KiB prefix is just as opaque to log readers as the full value.
_MASK_KEY_MAX_LEN = 4096


def mask_key(value: str | bytes | None) -> str:
    """Return a log-safe mask of a key/token value.

    Never exposes any prefix of the raw token. Format:
      - ``None`` or empty (incl. whitespace-only)  → ``"<missing>"``
      - non-empty → ``"<masked:" + sha256(value)[:6] + ">"``

    The 6-hex-digit truncated SHA-256 is one-way but stable for the same
    input, so operators can still correlate auth-rejection log lines
    across a single rotation window without ever seeing the raw key.
    See codex MUST-FIX 3.

    Codex Week 3 NIT:
      - Accept ``bytes`` and decode safely (don't fall through to
        ``str(b"")`` which returns the truthy string ``"b''"``).
      - Cap input length at 4 KiB so a multi-MB header doesn't waste
        cycles hashing on the failure path.
    """
    if value is None:
        return "<missing>"
    # Normalize to str. bytes → utf-8 with replacement so we never raise
    # on weird bytes during a log call.
    if isinstance(value, (bytes, bytearray)):
        if len(value) == 0:
            return "<missing>"
        s = bytes(value)[:_MASK_KEY_MAX_LEN].decode("utf-8", errors="replace")
    elif isinstance(value, str):
        s = value[:_MASK_KEY_MAX_LEN]
    else:
        # Other types: take repr but capped — also routes through the
        # empty-check below.
        s = str(value)[:_MASK_KEY_MAX_LEN]
    if not s.strip():
        return "<missing>"
    digest = hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:6]
    return f"<masked:{digest}>"


# ---------------------------------------------------------------------------
# HTTP check
# ---------------------------------------------------------------------------

_UNAUTHORIZED_BODY = {"error": "unauthorized", "detail": "missing_or_invalid_api_key"}
_WWW_AUTH = "Bearer"


def check_http(request: Request) -> None:
    """FastAPI dependency: raises 401 when auth is enabled and missing/wrong.

    Disabled (no keys configured): always passes.
    """
    keys = _configured_keys()
    if not keys:
        return

    authz = request.headers.get("authorization")
    token = _extract_bearer(authz)
    if not token or not _matches(token, keys):
        try:
            from server.core import metrics
            metrics.inc_auth_rejected(request.url.path or "unknown")
        except Exception:
            pass
        client_host = request.client.host if request.client else None
        logger.warning(
            "api_auth: HTTP 401 endpoint=%s client=%s supplied=%s",
            request.url.path,
            client_host,
            mask_key(token),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED_BODY,
            headers={"WWW-Authenticate": _WWW_AUTH},
        )


def unauthorized_response(endpoint: str = "unknown") -> JSONResponse:
    """Build a manual 401 JSON response (for code paths that don't raise)."""
    try:
        from server.core import metrics
        metrics.inc_auth_rejected(endpoint)
    except Exception:
        pass
    return JSONResponse(
        _UNAUTHORIZED_BODY,
        status_code=401,
        headers={"WWW-Authenticate": _WWW_AUTH},
    )


# ---------------------------------------------------------------------------
# WebSocket check
# ---------------------------------------------------------------------------

async def check_ws(ws: WebSocket) -> bool:
    """Validate a WS connection before accept.

    When disabled: returns True without touching the socket.
    When enabled and credential is valid: returns True.
    When enabled and credential missing/invalid: accepts then closes
    with code 4401 and returns False — caller MUST return immediately
    without registering session/backend resources.

    Header precedence: ``Authorization: Bearer ...`` wins over
    ``?token=...``. An invalid header fails even if a query token would
    have matched.
    """
    keys = _configured_keys()
    if not keys:
        return True

    # Header has priority. Starlette lowercases header names.
    authz = ws.headers.get("authorization")
    if authz is not None:
        token = _extract_bearer(authz)
    else:
        # Fall back to query token only when no Authorization header at all.
        token = ws.query_params.get("token")
        if token is not None:
            token = token.strip()

    if not token or not _matches(token, keys):
        try:
            from server.core import metrics
            metrics.inc_auth_rejected(ws.url.path or "unknown")
        except Exception:
            pass
        client_host = ws.client.host if ws.client else None
        logger.warning(
            "api_auth: WS 4401 endpoint=%s client=%s supplied=%s",
            ws.url.path,
            client_host,
            mask_key(token),
        )
        try:
            await ws.accept()
        except Exception:
            return False
        try:
            # Reason text must not contain token values.
            await ws.close(code=4401, reason='{"error":"unauthorized"}')
        except Exception:
            pass
        return False
    return True
