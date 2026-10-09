"""FastAPI dependency: gate admin routes by loopback or shared secret.

Policy
------
* If the request's ``client.host`` is a loopback address (``127.0.0.0/8`` or
  ``::1``), the request is always allowed. Local operators on the box don't
  need a key.
* Otherwise, ``OVS_ADMIN_KEY`` must be set in the environment, and the
  ``X-Admin-Key`` request header must match it (constant-time compare).
* ``X-Forwarded-For`` is intentionally NOT consulted — a remote attacker could
  forge it.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _admin_key() -> str | None:
    raw = os.environ.get("OVS_ADMIN_KEY")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


async def require_admin(
    request: Request,
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> None:
    """FastAPI dependency that authorizes admin requests.

    Raises:
        HTTPException(403): non-loopback request, but ``OVS_ADMIN_KEY`` is not set.
        HTTPException(401): non-loopback request, key set, but header mismatched.
    """
    client_host = request.client.host if request.client else None
    if _is_loopback(client_host):
        return

    expected = _admin_key()
    if expected is None:
        logger.warning(
            "Admin route refused: non-loopback client %s and OVS_ADMIN_KEY unset",
            client_host,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin routes require OVS_ADMIN_KEY when accessed remotely",
        )

    provided = x_admin_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid X-Admin-Key",
        )
