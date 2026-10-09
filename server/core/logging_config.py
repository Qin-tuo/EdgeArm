"""Structured logging configuration (Week 2 production hardening).

Two formats are supported and selected via the ``OVS_LOG_FORMAT`` env
var:

  * ``text`` (default) — preserves the legacy line format so local
    grep/tail workflows keep working.
  * ``json`` — emits one JSON object per log record with stable keys
    (``ts``, ``level``, ``logger``, ``msg``, optional ``request_id`` /
    ``session_id`` / ``backend``). Production compose files set
    ``OVS_LOG_FORMAT=json``.

Context propagation: ``request_id_var`` / ``session_id_var`` /
``backend_var`` are ``contextvars.ContextVar`` instances populated by
HTTP middleware and WS handler glue. Values flow across ``await`` and
``asyncio.create_task`` boundaries automatically. Crossing into
``run_in_executor`` threads still needs explicit
``contextvars.copy_context()`` (best-effort; logs from executor
threads will simply lack the context fields when not wrapped).

Security: ``mask_url_query`` and ``mask_sensitive_value`` scrub
``Authorization`` headers and ``?token=...`` query parameters before
they hit a log line. The middleware never reads the request body and
never logs raw headers/queries.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    from pythonjsonlogger import jsonlogger  # type: ignore
    _HAS_JSON_LOGGER = True
except Exception:  # pragma: no cover - dependency missing in dev
    jsonlogger = None  # type: ignore
    _HAS_JSON_LOGGER = False


_REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_MAX_LEN = 128
_REQUEST_ID_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ovs_request_id", default=None
)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ovs_session_id", default=None
)
backend_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ovs_backend", default=None
)


@dataclass
class TokenBundle:
    """Opaque handle returned by ``set_request_context`` so callers can
    reset all three contextvars in a single ``reset_request_context``
    invocation (used in finally blocks).
    """
    request_id_token: contextvars.Token | None
    session_id_token: contextvars.Token | None
    backend_token: contextvars.Token | None


def set_request_context(
    request_id: str | None = None,
    session_id: str | None = None,
    backend: str | None = None,
) -> TokenBundle:
    return TokenBundle(
        request_id_token=request_id_var.set(request_id) if request_id is not None else None,
        session_id_token=session_id_var.set(session_id) if session_id is not None else None,
        backend_token=backend_var.set(backend) if backend is not None else None,
    )


def reset_request_context(tokens: TokenBundle) -> None:
    if tokens.request_id_token is not None:
        try:
            request_id_var.reset(tokens.request_id_token)
        except Exception:
            pass
    if tokens.session_id_token is not None:
        try:
            session_id_var.reset(tokens.session_id_token)
        except Exception:
            pass
    if tokens.backend_token is not None:
        try:
            backend_var.reset(tokens.backend_token)
        except Exception:
            pass


def get_request_id() -> str | None:
    return request_id_var.get()


# ---------------------------------------------------------------------------
# Request-id helpers
# ---------------------------------------------------------------------------

def generate_request_id() -> str:
    return uuid.uuid4().hex


def request_id_from_headers(headers: Mapping[str, str] | Iterable[tuple[str, str]] | None) -> str | None:
    """Extract a sanitised inbound request id, or None if missing/invalid.

    Strips control characters, caps length at 128 chars. Empty after
    cleaning → None so the caller falls back to a fresh UUID.
    """
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        raw = headers.get(_REQUEST_ID_HEADER) or headers.get("X-Request-ID")
    else:
        raw = None
        for k, v in headers:
            if k.lower() == _REQUEST_ID_HEADER:
                raw = v
                break
    if raw is None or not raw:
        return None
    cleaned = _REQUEST_ID_BAD_CHARS.sub("", str(raw))[:_REQUEST_ID_MAX_LEN].strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"(token=)([^&\s]+)", re.IGNORECASE)


def mask_sensitive_value(value: str | None) -> str:
    """Mask a secret-ish value.

    Delegates to :func:`server.core.api_auth.mask_key` so HTTP, WS, and log
    masking share a single non-reversible representation (sha256 prefix
    placeholder). See codex MUST-FIX 3.
    """
    from server.core.api_auth import mask_key
    return mask_key(value)


def mask_url_query(url: str) -> str:
    """Mask ``token=...`` (any number of occurrences) in a URL or query string."""
    if not url:
        return url
    from server.core.api_auth import mask_key

    def _sub(m: re.Match) -> str:
        prefix, val = m.group(1), m.group(2)
        return f"{prefix}{mask_key(val)}"

    return _TOKEN_RE.sub(_sub, url)


def sanitize_headers_for_log(headers: Mapping[str, str] | Iterable[tuple[str, str]] | None) -> dict:
    if headers is None:
        return {}
    items: Iterable[tuple[str, str]]
    if isinstance(headers, Mapping):
        items = headers.items()
    else:
        items = headers
    out: dict[str, str] = {}
    for k, v in items:
        lk = k.lower()
        if lk == "authorization":
            # Drop the raw value entirely; keep scheme if present.
            parts = (v or "").split(None, 1)
            scheme = parts[0] if parts else ""
            tok = parts[1] if len(parts) > 1 else ""
            out[k] = f"{scheme} {mask_sensitive_value(tok)}".strip() if scheme else "<masked>"
        elif lk == "cookie" or lk == "set-cookie":
            out[k] = "<masked>"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


class _ContextInjectorFilter(logging.Filter):
    """Attach contextvars to every record so formatters can read them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        record.backend = backend_var.get()
        return True


class OVSTextFormatter(logging.Formatter):
    """Text formatter that matches the legacy ``server/main.py:31`` format
    while appending optional ``request_id=...`` when set."""

    default_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(self.default_fmt)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        rid = getattr(record, "request_id", None)
        if rid:
            base = f"{base} request_id={rid}"
        return base


if _HAS_JSON_LOGGER:

    class OVSJsonFormatter(jsonlogger.JsonFormatter):  # type: ignore[misc]
        def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:  # type: ignore[override]
            super().add_fields(log_record, record, message_dict)
            # Stable iso-with-ms timestamp.
            ts = time.gmtime(record.created)
            ms = int((record.created - int(record.created)) * 1000)
            log_record["ts"] = time.strftime(_LOG_TIME_FORMAT, ts) + f".{ms:03d}Z"
            log_record["level"] = record.levelname
            log_record["logger"] = record.name
            log_record["msg"] = record.getMessage()
            # Context fields — only emitted when set so the line stays
            # compact in idle logging paths.
            # pythonjsonlogger's default add_fields auto-copies any
            # LogRecord attribute named in its known reserved set; the
            # context filter writes None into request_id/session_id/
            # backend, which would surface as null keys. Strip those
            # null entries explicitly.
            rid = getattr(record, "request_id", None)
            if rid is None:
                log_record.pop("request_id", None)
            else:
                log_record["request_id"] = rid
            sid = getattr(record, "session_id", None)
            if sid is None:
                log_record.pop("session_id", None)
            else:
                log_record["session_id"] = sid
            be = getattr(record, "backend", None)
            if be is None:
                log_record.pop("backend", None)
            else:
                log_record["backend"] = be
            # Drop the redundant pythonjsonlogger ``message`` field so
            # consumers can rely on ``msg``.
            log_record.pop("message", None)
            log_record.pop("asctime", None)
            log_record.pop("taskName", None)

else:

    class OVSJsonFormatter(logging.Formatter):  # type: ignore[no-redef]
        """Minimal hand-rolled JSON fallback used when python-json-logger
        is not installed (dev shells). Produces the same field schema."""

        def format(self, record: logging.LogRecord) -> str:
            ts = time.gmtime(record.created)
            ms = int((record.created - int(record.created)) * 1000)
            payload: dict[str, Any] = {
                "ts": time.strftime(_LOG_TIME_FORMAT, ts) + f".{ms:03d}Z",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            rid = getattr(record, "request_id", None)
            if rid is not None:
                payload["request_id"] = rid
            sid = getattr(record, "session_id", None)
            if sid is not None:
                payload["session_id"] = sid
            be = getattr(record, "backend", None)
            if be is not None:
                payload["backend"] = be
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            try:
                return json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                return json.dumps({"msg": str(record.getMessage())})


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _resolve_format() -> str:
    raw = (os.environ.get("OVS_LOG_FORMAT") or "").strip().lower()
    if raw == "json":
        return "json"
    if raw == "text":
        return "text"
    if raw:
        logging.getLogger(__name__).warning(
            "OVS_LOG_FORMAT=%r unknown; falling back to text", raw
        )
    return "text"


def configure_root_logger(format_name: str) -> None:
    """Replace root handlers with a single stream handler using the
    requested formatter."""
    root = logging.getLogger()
    # Drop existing handlers so basicConfig calls earlier in import
    # don't shadow our formatter.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if format_name == "json":
        handler.setFormatter(OVSJsonFormatter())
    else:
        handler.setFormatter(OVSTextFormatter())
    handler.addFilter(_ContextInjectorFilter())

    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Bring uvicorn loggers in line — JSON in production, text by
    # default. This avoids "half the lines are JSON, half are not".
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.addHandler(handler)
        lg.propagate = False


def setup_logging() -> None:
    """Entrypoint: choose format from env and configure root logger."""
    configure_root_logger(_resolve_format())
