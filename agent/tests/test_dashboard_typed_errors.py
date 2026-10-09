"""Issue #8 — A3 typed on_error payloads + A6 dashboard colour rendering.

Verifies:
  • app_base broadcasts ``TypedLLMError`` on each failure path
    (llm_unavailable / llm_timeout / llm_failure / llm_stream_error).
  • DebugDashboardPlugin.on_error converts those into a dict payload
    that browser clients can colour-code by ``type``.
  • Backward compatibility: bare ``RuntimeError`` still produces a
    legacy ``type=unknown`` dict (no crash).
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ovs_agent import Config
from ovs_agent.app_base import BaseApp, TypedLLMError
from ovs_agent.event_bus import EventBus
from ovs_agent.llm import LLMStreamError
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin
from ovs_agent.plugins.llm_availability import LLMUnavailable
from ovs_agent.state import ConvState
from ovs_agent.app_mode import LLMTimeoutError


def _fresh_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.THINKING
    app._slv_reconnect_count = 0
    app._sleep_task = None
    app.config = Config(pipeline_mode="always_on")
    return app


class _CaptureErrPlugin:
    name = "errp"

    def __init__(self) -> None:
        self.errors: list[BaseException] = []

    async def on_error(self, exc):  # noqa: ANN001
        self.errors.append(exc)


# ── app_base broadcast layer ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_generic_llm_failure_emits_typed_llm_failure():
    app = _fresh_app()
    cap = _CaptureErrPlugin()
    app.plugins.append(cap)

    async def boom(text: str, detected_language: str | None = None) -> None:
        raise RuntimeError("edge-llm down")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert len(cap.errors) == 1
    exc = cap.errors[0]
    # Backward compat: still a RuntimeError, str() still contains key markers.
    assert isinstance(exc, RuntimeError)
    assert "LLM 调用失败" in str(exc)
    # New: carries a typed payload dict for the dashboard.
    payload = getattr(exc, "payload", None)
    assert isinstance(payload, dict), f"no .payload on exception: {exc!r}"
    assert payload["type"] == "llm_failure"
    assert payload["exc_class"] == "RuntimeError"
    assert "edge-llm down" in payload["message"]
    assert isinstance(payload.get("timestamp"), float)


@pytest.mark.asyncio
async def test_llm_stream_error_emits_typed_llm_stream_error():
    app = _fresh_app()
    cap = _CaptureErrPlugin()
    app.plugins.append(cap)

    async def boom(text: str, detected_language: str | None = None) -> None:
        raise LLMStreamError("finish_reason=error")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert len(cap.errors) == 1
    payload = cap.errors[0].payload
    assert payload["type"] == "llm_stream_error"
    assert payload["exc_class"] == "LLMStreamError"


@pytest.mark.asyncio
async def test_llm_timeout_emits_typed_llm_timeout():
    app = _fresh_app()
    cap = _CaptureErrPlugin()
    app.plugins.append(cap)

    async def boom(text: str, detected_language: str | None = None) -> None:
        raise LLMTimeoutError("first_token", 15.0)

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert len(cap.errors) == 1
    exc = cap.errors[0]
    assert isinstance(exc, RuntimeError)
    payload = exc.payload
    assert payload["type"] == "llm_timeout"
    assert payload["exc_class"] == "LLMTimeoutError"
    assert payload.get("kind") == "first_token"
    assert payload.get("timeout_s") == 15.0


@pytest.mark.asyncio
async def test_llm_unavailable_emits_typed_llm_unavailable():
    app = _fresh_app()
    cap = _CaptureErrPlugin()
    app.plugins.append(cap)

    async def boom(text: str, detected_language: str | None = None) -> None:
        raise LLMUnavailable("breaker open: DOWN")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert len(cap.errors) == 1
    payload = cap.errors[0].payload
    assert payload["type"] == "llm_unavailable"
    assert "不可用" in payload["message"]


# ── dashboard.on_error → browser payload shape ─────────────────────────


def _bare_dashboard() -> DebugDashboardPlugin:
    """Minimal DebugDashboardPlugin instance without app/aiohttp wiring."""
    p = DebugDashboardPlugin.__new__(DebugDashboardPlugin)
    p._errors = []
    p._browser_clients = set()
    p._started = False
    return p


@pytest.mark.asyncio
async def test_dashboard_on_error_typed_payload_round_trip():
    p = _bare_dashboard()
    broadcasts: list[tuple[str, Any]] = []

    async def fake_bcast(event, data=None):
        broadcasts.append((event, data))

    p._broadcast = fake_bcast  # type: ignore[assignment]

    exc = TypedLLMError(
        "llm_failure",
        "LLM 调用失败（RuntimeError）：edge crashed",
        exc_class="RuntimeError",
    )
    await p.on_error(exc)

    assert broadcasts, "no broadcast emitted"
    event, data = broadcasts[-1]
    assert event == "on_error"
    assert isinstance(data, dict)
    assert data["type"] == "llm_failure"
    assert data["exc_class"] == "RuntimeError"
    assert "edge crashed" in data["message"]
    # The persistent error log must keep the typed fields for snapshot replay.
    assert len(p._errors) == 1
    entry = p._errors[0]
    assert entry["type"] == "llm_failure"
    assert entry["exc_class"] == "RuntimeError"
    assert entry["msg"] == data["message"]


@pytest.mark.asyncio
async def test_dashboard_on_error_legacy_runtimeerror_still_works():
    """A plain RuntimeError (no .payload) must not crash and should
    fall back to type=unknown so the browser renders it grey."""
    p = _bare_dashboard()
    broadcasts: list[tuple[str, Any]] = []

    async def fake_bcast(event, data=None):
        broadcasts.append((event, data))

    p._broadcast = fake_bcast  # type: ignore[assignment]

    await p.on_error(RuntimeError("plain old error"))

    event, data = broadcasts[-1]
    assert event == "on_error"
    assert isinstance(data, dict)
    assert data["type"] == "unknown"
    assert data["message"] == "plain old error"
    assert data["exc_class"] == "RuntimeError"


@pytest.mark.asyncio
async def test_dashboard_snapshot_contains_typed_error_entries():
    """When a late client joins, the seeded errors[] in the snapshot
    must carry type / exc_class so the browser can colour them."""
    p = _bare_dashboard()
    p._broadcast = lambda *a, **k: _noop()  # type: ignore

    exc = TypedLLMError("llm_timeout", "timeout", exc_class="LLMTimeoutError")
    await p.on_error(exc)

    # The dashboard sends snapshot["data"]["errors"] = list(self._errors)
    # — verify those entries are dicts with the typed fields.
    snap_errors = list(p._errors)
    assert snap_errors
    assert snap_errors[0]["type"] == "llm_timeout"
    assert snap_errors[0]["exc_class"] == "LLMTimeoutError"


async def _noop():
    return None
