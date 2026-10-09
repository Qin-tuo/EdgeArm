"""Week 2 structured JSON logging tests."""

from __future__ import annotations

import json
import logging

import pytest

from server.core import logging_config as lc


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)


def _capture_record_as_json(message: str = "hello") -> dict:
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(lc.OVSJsonFormatter())
    handler.addFilter(lc._ContextInjectorFilter())
    log = logging.getLogger("test.json")
    # Isolate this logger from the root handlers under test.
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.info(message)
    handler.flush()
    raw = stream.getvalue().strip().splitlines()[-1]
    return json.loads(raw)


def test_json_formatter_basic_fields():
    payload = _capture_record_as_json("hello world")
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.json"
    assert "ts" in payload
    # ts must look like ISO 8601 with ms + Z.
    assert payload["ts"].endswith("Z")
    assert "T" in payload["ts"]


def test_json_formatter_includes_request_id_when_set():
    tokens = lc.set_request_context(request_id="req-123")
    try:
        payload = _capture_record_as_json("with context")
    finally:
        lc.reset_request_context(tokens)
    assert payload["request_id"] == "req-123"


def test_json_formatter_no_context_keys_when_unset():
    payload = _capture_record_as_json("no ctx")
    # request_id / session_id / backend should be absent when contextvars are None.
    assert "request_id" not in payload
    assert "session_id" not in payload
    assert "backend" not in payload


def test_json_formatter_handles_printf_style():
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(lc.OVSJsonFormatter())
    handler.addFilter(lc._ContextInjectorFilter())
    log = logging.getLogger("test.json.printf")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.warning("x=%s y=%d", "foo", 42)
    handler.flush()
    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert payload["msg"] == "x=foo y=42"
    assert payload["level"] == "WARNING"


def test_json_formatter_handles_exception():
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(lc.OVSJsonFormatter())
    handler.addFilter(lc._ContextInjectorFilter())
    log = logging.getLogger("test.json.exc")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("caught it")
    handler.flush()
    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert payload["msg"] == "caught it"
    assert "exc_info" in payload
    assert "RuntimeError" in payload["exc_info"]


def test_text_formatter_includes_request_id_suffix():
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(lc.OVSTextFormatter())
    handler.addFilter(lc._ContextInjectorFilter())
    log = logging.getLogger("test.text")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    tokens = lc.set_request_context(request_id="rid-X")
    try:
        log.info("ping")
    finally:
        lc.reset_request_context(tokens)
    line = stream.getvalue().strip().splitlines()[-1]
    assert "ping" in line
    assert "request_id=rid-X" in line


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_mask_sensitive_value_default_missing():
    assert lc.mask_sensitive_value(None) == "<missing>"
    assert lc.mask_sensitive_value("") == "<missing>"


def test_mask_sensitive_value_never_exposes_prefix():
    raw = "abcdefghij"
    out = lc.mask_sensitive_value(raw)
    assert raw[:3] not in out and raw[:8] not in out
    assert out.startswith("<masked:") and out.endswith(">")


def test_mask_url_query_single_token():
    masked = lc.mask_url_query("/ws?foo=bar&token=secret-key&x=1")
    assert "secret-key" not in masked
    assert "token=<masked:" in masked
    assert "foo=bar" in masked


def test_mask_url_query_multiple_tokens():
    masked = lc.mask_url_query("/ws?token=alpha&token=beta")
    assert masked.count("<masked:") == 2
    assert "alpha" not in masked
    assert "beta" not in masked


def test_mask_url_query_no_token_unchanged():
    url = "/livez"
    assert lc.mask_url_query(url) == url


def test_sanitize_headers_masks_authorization():
    s = lc.sanitize_headers_for_log({"Authorization": "Bearer mysecrettoken123"})
    assert "mysecrettoken123" not in s["Authorization"]
    assert s["Authorization"].startswith("Bearer ")


# ---------------------------------------------------------------------------
# Inbound request id sanitisation
# ---------------------------------------------------------------------------

def test_request_id_from_headers_propagates():
    rid = lc.request_id_from_headers({"x-request-id": "abc-123"})
    assert rid == "abc-123"


def test_request_id_from_headers_strips_control_chars():
    rid = lc.request_id_from_headers({"x-request-id": "a\x00b\x1fc"})
    assert rid == "abc"


def test_request_id_from_headers_caps_length():
    big = "x" * 200
    rid = lc.request_id_from_headers({"x-request-id": big})
    assert len(rid) == 128


def test_request_id_from_headers_missing_returns_none():
    assert lc.request_id_from_headers({}) is None
    assert lc.request_id_from_headers(None) is None


def test_request_id_from_headers_empty_after_cleaning_returns_none():
    rid = lc.request_id_from_headers({"x-request-id": "\x00\x00"})
    assert rid is None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def test_setup_logging_text_default(monkeypatch):
    monkeypatch.delenv("OVS_LOG_FORMAT", raising=False)
    lc.setup_logging()
    root = logging.getLogger()
    assert root.handlers, "root logger should have a handler"
    fmt = root.handlers[0].formatter
    assert isinstance(fmt, lc.OVSTextFormatter)


def test_setup_logging_json(monkeypatch):
    monkeypatch.setenv("OVS_LOG_FORMAT", "json")
    lc.setup_logging()
    root = logging.getLogger()
    assert root.handlers
    fmt = root.handlers[0].formatter
    assert isinstance(fmt, lc.OVSJsonFormatter)


def test_setup_logging_unknown_falls_back_to_text(monkeypatch):
    monkeypatch.setenv("OVS_LOG_FORMAT", "yaml-or-whatever")
    lc.setup_logging()
    root = logging.getLogger()
    fmt = root.handlers[0].formatter
    assert isinstance(fmt, lc.OVSTextFormatter)


def test_context_var_reset_idempotent():
    tokens = lc.set_request_context(request_id="r1", session_id="s1", backend="be")
    assert lc.request_id_var.get() == "r1"
    assert lc.session_id_var.get() == "s1"
    assert lc.backend_var.get() == "be"
    lc.reset_request_context(tokens)
    assert lc.request_id_var.get() is None
    assert lc.session_id_var.get() is None
    assert lc.backend_var.get() is None
