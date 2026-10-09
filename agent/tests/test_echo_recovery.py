"""Regression: when the small/quantised LLM falls into an in-context
echo loop (3+ consecutive identical short assistant replies), the
session must auto-clear history so the next turn starts fresh.

Without this, every subsequent turn — even legitimate user intent —
sees the same canned reply (e.g. "我在这里呢，想听什么就说吧。")
because the model pattern-matches the repetition in history.
"""
from __future__ import annotations

import pytest

from ovs_agent.session import Session


def test_three_identical_short_replies_auto_clear():
    sess = Session()
    sess.add_user("你好")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    sess.add_user("嗯")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    sess.add_user("你")
    # Third identical assistant — echo recovery should fire on append.
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    assert sess.history == [], "history should be auto-cleared on echo"
    assert sess.cache_warmed is False


def test_two_identical_replies_do_not_trigger():
    sess = Session()
    sess.add_user("hi")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    sess.add_user("hi again")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    # Two duplicates is normal small-talk; must not clear.
    assert len(sess.history) == 4


def test_long_identical_replies_are_not_treated_as_echo():
    """Two byte-identical 200-char replies are implausible in real
    conversation but length-cap protects against false positives just
    in case (e.g. cached deterministic answers to a FAQ)."""
    sess = Session()
    long = "今天天气不错适合出门散步" * 10  # ~120 chars
    for _ in range(3):
        sess.add_user("x")
        sess.add_assistant(long)
    assert len(sess.history) == 6


def test_non_consecutive_duplicates_do_not_trigger():
    sess = Session()
    sess.add_user("a")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    sess.add_user("b")
    sess.add_assistant("好啊，我们继续聊。")
    sess.add_user("c")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    sess.add_user("d")
    sess.add_assistant("我在这里呢，想听什么就说吧。")
    # 4 assistant turns; last 3 are NOT all identical (turn 2 differs).
    assert len(sess.history) == 8


def test_echo_recovery_emits_event():
    seen = []

    class _Bus:
        def emit(self, name, data):
            seen.append((name, data))

    sess = Session()
    sess.event_bus = _Bus()
    sess.add_user("a")
    sess.add_assistant("我在这里呢。")
    sess.add_user("b")
    sess.add_assistant("我在这里呢。")
    sess.add_user("c")
    sess.add_assistant("我在这里呢。")
    assert seen and seen[0][0] == "on_echo_recovery"
    assert seen[0][1]["window"] == 3
    assert "我在这里呢" in seen[0][1]["echo_text"]


def test_echo_recovery_resets_cache_warmed():
    sess = Session()
    sess.cache_warmed = True
    sess.prefix_cache_disabled = True
    for _ in range(3):
        sess.add_user("x")
        sess.add_assistant("短句")
    assert sess.cache_warmed is False
    assert sess.prefix_cache_disabled is False
