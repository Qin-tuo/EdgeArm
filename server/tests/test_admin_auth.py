"""Tests for server.core.admin_auth.require_admin dependency."""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from server.core import admin_auth
from server.core.admin_auth import require_admin


def _make_request(host: str | None) -> MagicMock:
    req = MagicMock()
    if host is None:
        req.client = None
    else:
        req.client = MagicMock()
        req.client.host = host
    return req


def _run(coro):
    return asyncio.run(coro)


def test_loopback_127_allowed(monkeypatch):
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    req = _make_request("127.0.0.1")
    _run(require_admin(req, x_admin_key=None))


def test_loopback_ipv6_allowed(monkeypatch):
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    req = _make_request("::1")
    _run(require_admin(req, x_admin_key=None))


def test_remote_without_env_key_forbidden(monkeypatch):
    monkeypatch.delenv("OVS_ADMIN_KEY", raising=False)
    req = _make_request("10.1.2.3")
    with pytest.raises(HTTPException) as exc:
        _run(require_admin(req, x_admin_key=None))
    assert exc.value.status_code == 403


def test_remote_with_correct_key_allowed(monkeypatch):
    monkeypatch.setenv("OVS_ADMIN_KEY", "secret-xyz")
    req = _make_request("10.1.2.3")
    _run(require_admin(req, x_admin_key="secret-xyz"))


def test_remote_with_wrong_key_unauthorized(monkeypatch):
    monkeypatch.setenv("OVS_ADMIN_KEY", "secret-xyz")
    req = _make_request("10.1.2.3")
    with pytest.raises(HTTPException) as exc:
        _run(require_admin(req, x_admin_key="not-the-key"))
    assert exc.value.status_code == 401


def test_remote_with_no_header_unauthorized(monkeypatch):
    monkeypatch.setenv("OVS_ADMIN_KEY", "secret-xyz")
    req = _make_request("10.1.2.3")
    with pytest.raises(HTTPException) as exc:
        _run(require_admin(req, x_admin_key=None))
    assert exc.value.status_code == 401


def test_constant_time_compare_used(monkeypatch):
    """Verify hmac.compare_digest is what's actually used for key comparison."""
    monkeypatch.setenv("OVS_ADMIN_KEY", "expected")
    calls: list[tuple[str, str]] = []

    import hmac as real_hmac
    real_cd = real_hmac.compare_digest

    def fake_compare_digest(a, b):
        calls.append((a, b))
        return real_cd(a, b)

    monkeypatch.setattr(admin_auth.hmac, "compare_digest", fake_compare_digest)

    req = _make_request("10.1.2.3")
    _run(require_admin(req, x_admin_key="expected"))
    assert calls, "hmac.compare_digest was not called"
    assert calls[0] == ("expected", "expected")


def test_source_uses_hmac_compare_digest():
    """Defensive check: source code calls hmac.compare_digest, not ==."""
    src = inspect.getsource(require_admin)
    assert "hmac.compare_digest" in src


def test_is_loopback_helper():
    assert admin_auth._is_loopback("127.0.0.1") is True
    assert admin_auth._is_loopback("127.5.5.5") is True
    assert admin_auth._is_loopback("::1") is True
    assert admin_auth._is_loopback("10.0.0.1") is False
    assert admin_auth._is_loopback(None) is False
    assert admin_auth._is_loopback("not-an-ip") is False
