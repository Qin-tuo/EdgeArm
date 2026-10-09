"""Plan D item 5 + item 6 — corner case: history with no user turns.

When ``session.history`` carries a system prompt followed exclusively by
``tool`` (and/or orphan ``assistant``) messages, the regular turn-grouper
in ``_trim_to_budget`` produces zero turns (no user anchors). Plan D
item 6 has been fixed: the trim now degrades to a raw drop-oldest pass
on the non-system messages, logs an ERROR so the underlying invariant
violation gets noticed, and clears ``prefix_cache_warmed`` because the
upstream KV-cache key may no longer be valid.

Regression contract:
  * Oversized all-tool history is trimmed (NOT silently passed through).
  * An ERROR log explains the corner case.
  * If ``prefix_cache_warmed`` was True, it gets cleared on the
    fallback path.
  * If the all-tool history fits inside budget, it's returned unchanged
    (no spurious trim).
"""
from __future__ import annotations

import logging

import pytest

from ovs_agent.session import Session


def _approx_token_counter(text: str) -> int:
    """Char/3 estimator. Stable + tokenizer-free for tests."""
    return max(1, len(text) // 3)


def test_all_tool_history_oversize_drops_oldest_with_error_log(caplog):
    """Plan D item 6 fix: an oversized all-tool history is trimmed by
    raw drop-oldest, ERROR is logged, and prefix_cache_warmed cleared."""
    sess = Session(
        max_input_tokens=200,           # budget = 200 * 0.75 = 150 tokens
        token_counter=_approx_token_counter,
    )
    sess.prefix_cache_warmed = True
    # System prefix + 5 fat tool messages, no user/assistant text.
    # Each tool message ~200 tokens (600 chars / 3) → 1000+ tokens dynamic.
    sess.history = [
        {"role": "system", "content": "You are a robot."},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 600},
        {"role": "tool", "tool_call_id": "2", "content": "y" * 600},
        {"role": "tool", "tool_call_id": "3", "content": "z" * 600},
        {"role": "tool", "tool_call_id": "4", "content": "a" * 600},
        {"role": "tool", "tool_call_id": "5", "content": "b" * 600},
    ]
    caplog.set_level(logging.ERROR, logger="ovs_agent.session")

    trimmed = sess._trim_to_budget(sess.history, sess.max_input_tokens)

    # System prompt is always kept; some tool messages were dropped.
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) < len(sess.history), (
        "all-tool history over budget MUST be trimmed (Plan D item 6 fix)"
    )

    # The corner-case ERROR was logged so ops sees the invariant violation.
    errs = [
        r for r in caplog.records
        if r.name == "ovs_agent.session"
        and r.levelno == logging.ERROR
        and "no user-anchored turns" in r.getMessage()
    ]
    assert errs, "expected ERROR log on all-tool fallback trim path"

    # Fallback clears prefix_cache_warmed (the cached prefix likely no
    # longer matches the trimmed history).
    assert sess.prefix_cache_warmed is False


def test_all_tool_history_under_budget_returns_unchanged():
    """When the all-tool history already fits, no trim, no warning."""
    sess = Session(
        max_input_tokens=2000,          # budget = 1500 tokens
        token_counter=_approx_token_counter,
    )
    sess.history = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "tool_call_id": "1", "content": "small"},
        {"role": "tool", "tool_call_id": "2", "content": "still small"},
    ]
    trimmed = sess._trim_to_budget(sess.history, sess.max_input_tokens)
    assert trimmed == sess.history


def test_normal_history_still_trims_after_corner_case_coverage():
    """Guardrail: the regular trim path is still working."""
    sess = Session(
        max_input_tokens=200,           # budget = 150 tokens
        token_counter=_approx_token_counter,
    )
    sess.history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 200},
        {"role": "assistant", "content": "b" * 200},
        {"role": "user", "content": "c" * 200},
        {"role": "assistant", "content": "d" * 200},
        {"role": "user", "content": "e" * 200},
        {"role": "assistant", "content": "f" * 200},
    ]
    trimmed = sess._trim_to_budget(sess.history, sess.max_input_tokens)
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) < len(sess.history)
    assert trimmed[-1]["role"] == "assistant"
    assert trimmed[-1]["content"] == "f" * 200
