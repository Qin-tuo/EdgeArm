"""Tests for Session tool_calls/role:tool support + turn-aware trim."""
from __future__ import annotations

import pytest

from ovs_agent.session import Session


def _fake_counter(text: str) -> int:
    return max(1, len(text) // 4)


# ── add_assistant_tool_calls / add_tool_result shape ──────────────────


def test_add_assistant_tool_calls_shape_with_none_content():
    s = Session()
    tcs = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }]
    s.add_assistant_tool_calls(None, tcs)
    assert s.history[-1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": tcs,
    }
    # Crucially: the "content" key must be present (explicit None),
    # not omitted — OpenAI's wire format expects it.
    assert "content" in s.history[-1]


def test_add_assistant_tool_calls_with_preamble():
    s = Session()
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}]
    s.add_assistant_tool_calls("let me check", tcs)
    assert s.history[-1]["content"] == "let me check"


def test_add_tool_result_shape():
    s = Session()
    s.add_tool_result("call_1", '{"now": "2026-01-01T00:00:00"}')
    assert s.history[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"now": "2026-01-01T00:00:00"}',
    }


# ── rollback_to ───────────────────────────────────────────────────────


def test_rollback_to_truncates():
    s = Session()
    s.add_user("u1")
    s.add_assistant("a1")
    s.add_user("u2")
    s.add_assistant("a2")
    dropped = s.rollback_to(2)
    assert dropped == 2
    assert len(s.history) == 2
    assert [m["content"] for m in s.history] == ["u1", "a1"]


def test_rollback_to_idempotent_at_tail():
    s = Session()
    s.add_user("u1")
    s.add_assistant("a1")
    assert s.rollback_to(len(s.history)) == 0
    assert len(s.history) == 2


def test_rollback_to_zero_clears_history():
    s = Session()
    s.add_user("u1")
    s.add_assistant("a1")
    assert s.rollback_to(0) == 2
    assert s.history == []


# ── turn-aware trim with tool turns ───────────────────────────────────


def test_trim_drops_whole_tool_turn():
    """history layout (2 turns, each containing a tool round + final
    assistant text):

        turn 1: [user1, assistant_tc, tool_result, assistant_text]
        turn 2: [user2, assistant_tc, tool_result, assistant_text]

    A trim that needs to free one turn MUST drop all 4 messages of
    turn 1 — never just 3.
    """
    s = Session(max_input_tokens=200, token_counter=_fake_counter)
    # Build turn 1
    s.add_user("user message one with some words to inflate token count")
    s.add_assistant_tool_calls(None, [{
        "id": "c1", "type": "function",
        "function": {"name": "f", "arguments": '{"x":1}'},
    }])
    s.add_tool_result("c1", '{"result": "abcdef ghijkl mnopqr stuvwx yz"}')
    s.add_assistant("answer to turn one is here, padded out with more text")
    # Build turn 2
    s.add_user("user message two with some words to inflate token count")
    s.add_assistant_tool_calls(None, [{
        "id": "c2", "type": "function",
        "function": {"name": "f", "arguments": '{"x":2}'},
    }])
    s.add_tool_result("c2", '{"result": "abcdef ghijkl mnopqr stuvwx yz"}')
    s.add_assistant("answer to turn two is here, padded out with more text")

    msgs = s.messages("sys")
    history = msgs[1:]  # strip system
    # Whatever survived, no half-tool-turn should be present.
    # Either turn 1 (4 messages) fully present + turn 2 fully present (no
    # trim happened) OR turn 1 fully dropped and turn 2 fully present.
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    # Each turn starts with a user message; the count tells us how many.
    assert len(user_indices) >= 1
    # The last turn must end with a normal assistant(text).
    assert history[-1]["role"] == "assistant"
    assert history[-1].get("content") is not None
    assert not history[-1].get("tool_calls")
    # The very last assistant text is turn 2's answer.
    assert "turn two" in history[-1]["content"]
    # If trim happened, turn 1's user message must be gone entirely.
    if len(user_indices) == 1:
        assert all("turn one" not in (m.get("content") or "") for m in history)
        # And NO orphan tool/assistant_tc from turn 1 left behind.
        # Specifically check that the first message after system is a
        # `user` (not an orphan assistant or tool).
        assert history[0]["role"] == "user"


def test_trim_keeps_atomicity_with_many_tool_turns():
    """Stress: 10 tool turns, tight budget. After trim, every kept turn
    must start with `user` and end with `assistant(text)` — no orphans."""
    s = Session(max_input_tokens=150, token_counter=_fake_counter)
    for i in range(10):
        s.add_user(f"u{i} " * 6)
        s.add_assistant_tool_calls(None, [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }])
        s.add_tool_result(f"c{i}", '{"r": "xxxxxxxx"}')
        s.add_assistant(f"a{i} " * 6)
    msgs = s.messages("sys")
    history = msgs[1:]
    # Split by user boundaries and re-check each turn.
    boundaries = [i for i, m in enumerate(history) if m.get("role") == "user"]
    assert boundaries, "at least one turn must remain"
    boundaries.append(len(history))
    for start, end in zip(boundaries, boundaries[1:]):
        turn = history[start:end]
        assert turn[0]["role"] == "user", turn
        last = turn[-1]
        assert last["role"] == "assistant", turn
        assert last.get("content") is not None, turn
        assert not last.get("tool_calls"), turn


# ── echo recovery skips tool-call-only assistants ─────────────────────


def test_echo_recovery_skips_tool_call_only_assistants():
    """When the assistant has emitted several short identical *text*
    replies, recovery should fire. But if the recent assistant turns are
    a mix of text and tool-call-only ones, only the text ones count."""
    s = Session()
    # Three identical short text replies interleaved with tool-call-only
    # assistant messages. Recovery counts only the text ones.
    for _ in range(3):
        s.add_user("u")
        s.add_assistant_tool_calls(None, [{
            "id": "x", "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }])
        s.add_tool_result("x", '{"ok": true}')
        s.add_assistant("ok!")
    # After 3 identical "ok!" assistant replies, recovery should have
    # fired (history cleared).
    assert s.history == []


def test_echo_recovery_does_not_falsepositive_on_tool_calls_only():
    """Three tool-call-only assistant messages with different args do
    NOT trigger echo recovery (no natural-language content to echo)."""
    s = Session()
    for i in range(3):
        s.add_user(f"u{i}")
        s.add_assistant_tool_calls(None, [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "f", "arguments": f'{{"x":{i}}}'},
        }])
        s.add_tool_result(f"c{i}", '{"ok": true}')
        s.add_assistant(f"different answer {i}")
    # History must still be intact — no recovery.
    assert len(s.history) == 12


# ── trailing in-flight tool round survives trim ───────────────────────


def test_trim_pins_in_flight_tool_turn():
    """A turn whose last message is `tool` (waiting on next assistant)
    must be treated as trailing/incomplete and survive trim intact."""
    s = Session(max_input_tokens=120, token_counter=_fake_counter)
    # Several completed turns first
    for i in range(5):
        s.add_user(f"old user {i} padded out with text")
        s.add_assistant(f"old answer {i} padded out with text")
    # Now an in-flight tool round (no final assistant text)
    s.add_user("the latest question still in flight")
    s.add_assistant_tool_calls(None, [{
        "id": "pending", "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }])
    s.add_tool_result("pending", '{"ok": true}')
    msgs = s.messages("sys")
    history = msgs[1:]
    # The in-flight turn (3 messages) must be the tail of history.
    assert history[-3]["role"] == "user"
    assert history[-3]["content"] == "the latest question still in flight"
    assert history[-2]["role"] == "assistant"
    assert history[-2]["tool_calls"][0]["id"] == "pending"
    assert history[-1]["role"] == "tool"
