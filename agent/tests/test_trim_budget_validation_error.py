"""Plan D item 5 — startup budget validator contract under tight prefix.

``BaseApp._validate_session_budget(system_prompt, tools)`` is a startup
sanity check (see ``app_base.py`` ~line 284). Its current contract is
**warn-and-continue**:
  * Logs ERROR when ``history_headroom < _MIN_HISTORY_HEADROOM`` (i.e.
    the fixed prefix leaves almost no room for any user turn).
  * Logs WARNING when fixed prefix > 60% of ``session_max_input_tokens``.
  * **Does NOT raise** — startup proceeds either way; the operator sees
    the ERROR in logs and re-tunes config.

This test pins the contract. If we ever flip to refuse-to-start (e.g.
because we keep getting paged by users who ignored the ERROR), this
test should fail loudly so we update it intentionally.
"""
from __future__ import annotations

import logging
import types

import pytest

from ovs_agent.app_base import BaseApp


def _make_app_with_config(max_input: int | None) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = types.SimpleNamespace(session_max_input_tokens=max_input)
    return app


def test_validate_session_budget_logs_error_on_tight_prefix(caplog):
    """Huge system_prompt → headroom < 1000 → ERROR + continue."""
    app = _make_app_with_config(max_input=2000)
    # _approx_tokens uses len(text) // 3, so 6000 chars ≈ 2000 tokens →
    # fixed_tokens >= max_input → headroom <= 0 → triggers ERROR.
    huge_prompt = "x" * 6000

    caplog.set_level(logging.ERROR, logger="ovs_agent.app_base")

    # Contract: must NOT raise. Validator is warn-and-continue.
    app._validate_session_budget(huge_prompt, tools=None)

    errors = [
        r for r in caplog.records
        if r.name == "ovs_agent.app_base"
        and r.levelno == logging.ERROR
    ]
    assert errors, "expected ERROR log for tight prefix budget"
    msg = errors[0].getMessage()
    assert "FIXED PREFIX" in msg
    assert "history headroom" in msg.lower() or "headroom" in msg.lower()


def test_validate_session_budget_does_not_raise_even_on_severe_overflow():
    """Even when fixed prefix vastly exceeds max_input, validator must
    return cleanly so app startup continues (warn-and-continue contract)."""
    app = _make_app_with_config(max_input=1000)
    insane_prompt = "y" * 100000  # ~33k token estimate vs 1000 budget

    # Should return None without raising.
    result = app._validate_session_budget(insane_prompt, tools=None)
    assert result is None


def test_validate_session_budget_skips_when_trim_disabled(caplog):
    """session_max_input_tokens=None → trim is disabled → no validation
    (just an INFO log noting that), regardless of how big the prompt is."""
    app = _make_app_with_config(max_input=None)
    caplog.set_level(logging.INFO, logger="ovs_agent.app_base")

    app._validate_session_budget("anything", tools=None)

    # Must not have an ERROR or WARNING — only INFO about trim disabled.
    errs = [
        r for r in caplog.records
        if r.name == "ovs_agent.app_base"
        and r.levelno >= logging.WARNING
    ]
    assert errs == []
