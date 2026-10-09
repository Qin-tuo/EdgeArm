"""Unit tests for server.core.api_auth."""

import os
import pytest

from server.core import api_auth, metrics


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    metrics._reset_for_tests()


def test_unset_env_disables_auth():
    assert api_auth.is_enabled() is False
    assert api_auth._configured_keys() == []


def test_empty_env_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "")
    assert api_auth.is_enabled() is False


def test_whitespace_only_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "   ")
    assert api_auth.is_enabled() is False


def test_comma_only_disables_auth(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", ",,,")
    assert api_auth.is_enabled() is False


def test_single_key_enables(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "abc")
    assert api_auth.is_enabled() is True
    assert api_auth._configured_keys() == ["abc"]


def test_multiple_keys_with_whitespace(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", " abc , def ,,ghi ")
    assert api_auth._configured_keys() == ["abc", "def", "ghi"]


def test_extract_bearer():
    assert api_auth._extract_bearer(None) is None
    assert api_auth._extract_bearer("") is None
    assert api_auth._extract_bearer("Basic abc") is None
    # "Bearer " with trailing whitespace has no second token after split.
    assert api_auth._extract_bearer("Bearer ") is None
    assert api_auth._extract_bearer("Bearer abc") == "abc"
    assert api_auth._extract_bearer("bearer ABC") == "ABC"
    assert api_auth._extract_bearer("Bearer  abc  ") == "abc"


def test_mask_key():
    import hashlib
    assert api_auth.mask_key(None) == "<missing>"
    assert api_auth.mask_key("") == "<missing>"
    # Whitespace-only is also treated as missing.
    assert api_auth.mask_key("   ") == "<missing>"
    # Non-empty: never exposes any prefix of the raw token.
    out_short = api_auth.mask_key("short")
    assert out_short.startswith("<masked:") and out_short.endswith(">")
    assert "short" not in out_short

    raw = "supersecretkey1234567"
    out = api_auth.mask_key(raw)
    # No raw substring ever leaks.
    for n in range(3, len(raw) + 1):
        assert raw[:n] not in out
    assert "1234567" not in out
    # Stable: same input → same masked output.
    assert api_auth.mask_key(raw) == out
    # Deterministic hash check.
    expected = hashlib.sha256(raw.encode()).hexdigest()[:6]
    assert out == f"<masked:{expected}>"


def test_mask_key_does_not_expose_token_prefix():
    """Codex MUST-FIX 3: mask_key must never return a multi-char prefix of raw."""
    raw = "secret123abcdef"
    masked = api_auth.mask_key(raw)
    # Walk prefixes of length 2+ — single-char "s" can incidentally appear
    # in the literal "<masked:...>" format, which is fine.
    for n in range(2, len(raw) + 1):
        assert raw[:n] not in masked, (
            f"mask_key leaked prefix {raw[:n]!r} via output {masked!r}"
        )
    assert "secret" not in masked
    assert "secret123" not in masked
    assert "abcdef" not in masked
    # And the format is the fixed placeholder, never resembling the raw.
    assert masked.startswith("<masked:") and masked.endswith(">")


def test_mask_key_consistent_hash():
    """Same token → same masked output (operator log correlation)."""
    raw = "another-very-long-api-key-zz"
    assert api_auth.mask_key(raw) == api_auth.mask_key(raw)
    # Different token → different masked output (overwhelmingly likely).
    assert api_auth.mask_key(raw) != api_auth.mask_key(raw + "x")


def test_mask_key_handles_bytes_input():
    """Codex Week 3 NIT: bytes input must be treated as the equivalent
    string, never via `str(b"...")` (which produces `"b'...'"`, leaking
    the original bytes literal into logs).
    """
    import hashlib
    # Empty bytes → <missing>, not `str(b"") == "b''"` (truthy).
    assert api_auth.mask_key(b"") == "<missing>"
    assert api_auth.mask_key(bytearray()) == "<missing>"

    raw = "secretbyteskey"
    bts = raw.encode("utf-8")
    masked = api_auth.mask_key(bts)
    # Format matches the str path.
    assert masked.startswith("<masked:") and masked.endswith(">")
    # No raw substring leaks.
    assert "secret" not in masked
    # Critically: the `b'...'` repr form must never appear.
    assert "b'" not in masked and "b\"" not in masked
    # And the digest matches what we'd get from the str path.
    assert masked == f"<masked:{hashlib.sha256(raw.encode()).hexdigest()[:6]}>"


def test_mask_key_caps_input_length():
    """A multi-MB token must not waste cycles hashing every byte. Cap at
    4 KiB — the masked output is identical for value vs value+padding
    beyond the cap, demonstrating the cap is applied before hashing."""
    base = "x" * 4096
    padded = base + ("y" * 100_000)
    assert api_auth.mask_key(base) == api_auth.mask_key(padded), (
        "input beyond _MASK_KEY_MAX_LEN must be truncated pre-hash"
    )
    # Same logic for bytes.
    base_b = b"x" * 4096
    padded_b = base_b + (b"y" * 100_000)
    assert api_auth.mask_key(base_b) == api_auth.mask_key(padded_b)


def test_mask_key_other_types_dont_crash():
    """A non-str/bytes value (e.g. an int passed by accident) should not
    raise — it's a log-only helper."""
    out = api_auth.mask_key(12345)
    assert out.startswith("<masked:") and out.endswith(">")


def test_matches_constant_time():
    keys = ["abc", "def"]
    assert api_auth._matches("abc", keys) is True
    assert api_auth._matches("def", keys) is True
    assert api_auth._matches("ghi", keys) is False
    assert api_auth._matches("", keys) is False


def test_hot_update_changes_check(monkeypatch):
    monkeypatch.setenv("OVS_API_KEYS", "k1")
    assert api_auth._matches("k1", api_auth._configured_keys())
    monkeypatch.setenv("OVS_API_KEYS", "k2")
    assert not api_auth._matches("k1", api_auth._configured_keys())
    assert api_auth._matches("k2", api_auth._configured_keys())


# ── HTTP integration via TestClient ────────────────────────────────────

def _build_app():
    from fastapi import FastAPI, Depends, Request
    from server.core.api_auth import check_http
    app = FastAPI()

    @app.get("/protected")
    def protected(request: Request, _: None = Depends(check_http)):
        return {"ok": True}

    return app


def test_http_disabled_lets_through(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("OVS_API_KEYS", raising=False)
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected")
        assert r.status_code == 200


def test_http_enabled_missing_key_401(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected")
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"
        body = r.json()
        # FastAPI nests HTTPException.detail under "detail"
        assert body["detail"]["error"] == "unauthorized"


def test_http_enabled_wrong_key_401(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401


def test_http_enabled_valid_key_passes(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Bearer mysecret"})
        assert r.status_code == 200


def test_http_non_bearer_scheme_fails(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OVS_API_KEYS", "mysecret")
    app = _build_app()
    with TestClient(app) as c:
        r = c.get("/protected", headers={"Authorization": "Basic mysecret"})
        assert r.status_code == 401
