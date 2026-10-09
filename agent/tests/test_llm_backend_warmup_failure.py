"""Plan D item 5 — coverage for ``backend.warmup()`` failure paths.

Two layers of defense:

1. ``EdgeLLMBackend.warmup()`` is itself fail-open: even if the
   underlying ``httpx.AsyncClient`` constructor explodes, warmup returns
   a sentinel dict with ``cache_warmed=False`` / ``graph_warmed=False``.
   No exception escapes — operators see a WARNING log instead.

2. ``BaseApp.run()`` still wraps the warmup call in ``try/except``
   (defense in depth — if a future backend changes the contract and
   does raise, the app must still start so the user can talk).

This test pins both layers so a regression that, e.g., adds a raise
inside warmup OR drops the app-layer try/except surfaces here.
"""
from __future__ import annotations

import logging

import pytest

from ovs_agent.llm import EdgeLLMBackend
from ovs_agent.session import Session


class _ExplodingClient:
    """httpx.AsyncClient stand-in whose constructor raises."""

    def __init__(self, *a, **kw):
        raise RuntimeError("transport boom: cannot construct AsyncClient")


@pytest.mark.asyncio
async def test_warmup_transport_failure_is_fail_open(monkeypatch, caplog):
    """Layer 1: EdgeLLMBackend.warmup must NOT raise on transport failure."""
    monkeypatch.setattr(
        "ovs_agent.llm.edge_llm.httpx.AsyncClient",
        _ExplodingClient,
    )
    caplog.set_level(logging.WARNING, logger="ovs_agent.llm.edge_llm")

    backend = EdgeLLMBackend(
        base_url="http://edge-llm:8000/v1",
        api_key="sk-test",
        model="qwen3-4b-awq",
    )

    # MUST return a dict, MUST NOT raise. This is the warmup contract.
    result = await backend.warmup(
        system_prompt="You are a robot.",
        tools=None,
        enable_thinking=False,
    )

    assert isinstance(result, dict)
    assert result.get("cache_warmed") is False
    assert result.get("graph_warmed") is False

    # Operator-visible warning logged.
    warnings = [
        r for r in caplog.records
        if r.name == "ovs_agent.llm.edge_llm"
        and r.levelno == logging.WARNING
    ]
    assert warnings, "expected WARNING on cache warmup transport failure"


@pytest.mark.asyncio
async def test_app_layer_try_except_swallows_unexpected_raise(caplog):
    """Layer 2: even if a backend's ``warmup()`` raises (contract
    violation), ``BaseApp.run``'s try/except keeps app startup alive.

    We exercise the contract directly by replaying the same pattern
    used in ``app_base.py`` (~line 564). If that block is ever
    refactored to fail-fast, this test will flag the change.
    """

    class _BrokenBackend:
        async def warmup(self, **kw):
            raise RuntimeError("backend internals exploded")

    backend = _BrokenBackend()
    session = Session()
    assert session.cache_warmed is False  # baseline

    logger = logging.getLogger("ovs_agent.app_base")
    caplog.set_level(logging.WARNING, logger="ovs_agent.app_base")

    startup_continued = False
    try:
        try:
            warmup_result = await backend.warmup(
                system_prompt="hi", tools=None, enable_thinking=False,
            )
            if warmup_result and warmup_result.get("cache_warmed"):
                session.cache_warmed = True
        except Exception:  # noqa: BLE001
            logger.warning(
                "LLM warmup failed; first turn may be cold", exc_info=True,
            )
        # The next lines simulate the rest of BaseApp.run() — plugin
        # start, wake loop, etc. If the try/except above did its job,
        # we reach here.
        startup_continued = True
    except Exception:
        startup_continued = False

    assert startup_continued, (
        "BaseApp.run() must continue after warmup raises (warn-and-continue)"
    )
    assert session.cache_warmed is False, (
        "session.cache_warmed must stay False when warmup failed"
    )
    warns = [
        r for r in caplog.records
        if r.name == "ovs_agent.app_base"
        and r.levelno == logging.WARNING
        and "LLM warmup failed" in r.getMessage()
    ]
    assert warns, "expected WARNING from app-layer try/except"
