"""A1-step2: trim budget covers only dynamic turns, not system prefix.

These tests pin the contract:
  * Budget = ``session.max_input_tokens * 0.75`` and applies ONLY to
    user/assistant/tool messages in ``Session.history``.
  * System prompt size has zero influence on trim decisions.
  * An empty history is a no-op even with an enormous system prompt.
"""
from __future__ import annotations

from ovs_agent.session import Session


def _fake_counter(text: str) -> int:
    # ~1 token per 4 chars; matches test_session_trim.py convention.
    return max(1, len(text) // 4)


def _add_turn(session: Session, u: str, a: str) -> None:
    session.add_user(u)
    session.add_assistant(a)


def test_huge_system_prompt_does_not_trigger_trim_when_history_small() -> None:
    """A 50k-char system prompt must NOT cause trimming of a tiny history."""
    session = Session(max_input_tokens=200, token_counter=_fake_counter)
    _add_turn(session, "hi", "hello")
    huge_system = "S" * 50_000  # 12_500 tokens — dwarfs the 200 token max
    msgs = session.messages(huge_system)
    # No history was dropped: 1 system + 1 user + 1 assistant
    assert len(msgs) == 3
    assert msgs[0]["content"] == huge_system
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_trim_decision_independent_of_system_size() -> None:
    """Same history, two different system prompts → identical trim outcome."""
    def build(system_text: str) -> list[dict]:
        s = Session(max_input_tokens=200, token_counter=_fake_counter)
        for i in range(30):
            _add_turn(s, f"user message number {i:03d}", f"assistant reply {i:03d}")
        return s.messages(system_text)

    small = build("tiny")
    huge = build("X" * 100_000)
    # Drop system msg at idx 0 and compare the remaining dynamic tail.
    assert [m["content"] for m in small[1:]] == [m["content"] for m in huge[1:]]


def test_empty_history_never_trims_even_with_giant_system() -> None:
    session = Session(max_input_tokens=100, token_counter=_fake_counter)
    msgs = session.messages("S" * 1_000_000)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


def test_history_alone_charged_against_budget() -> None:
    """Sum of returned dynamic turns must fit max_input_tokens * 0.75."""
    max_tokens = 400
    session = Session(max_input_tokens=max_tokens, token_counter=_fake_counter)
    for i in range(20):
        _add_turn(session, "u" * 120 + f"{i}", "a" * 120 + f"{i}")
    msgs = session.messages("any system")

    def cost(m: dict) -> int:
        return _fake_counter(m.get("content") or "") + 4

    history_cost = sum(cost(m) for m in msgs if m["role"] != "system")
    budget = int(max_tokens * 0.75)
    assert history_cost <= budget, (
        f"history {history_cost} exceeds dynamic-turn budget {budget}"
    )


# ── Plan D item 6: all-tool-message history corner case ─────────────


def test_trim_all_tool_messages_fallback_drops_oldest() -> None:
    """All-tool-message history exceeding budget triggers fallback drop.

    Pre-fix behaviour: regular trim sees turns=[] and returns input as-is
    even when total tokens > budget. New behaviour: log ERROR + drop
    oldest non-system messages until under budget; clear cache_warmed.
    """
    session = Session(max_input_tokens=200, token_counter=_fake_counter)
    long_payload = "x" * 80
    for i in range(20):
        session.history.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"{long_payload}-{i}",
        })
    session.prefix_cache_warmed = True

    msgs = session.messages("system")
    assert msgs[0]["role"] == "system"
    # Some tool messages dropped.
    assert len(msgs) < 21
    assert session.prefix_cache_warmed is False
    # Backward-compat alias mirrors the underlying field.
    assert session.cache_warmed is False


def test_trim_all_tool_messages_under_budget_is_noop() -> None:
    """If the all-tool history already fits, return unchanged + don't
    clear cache flags."""
    session = Session(max_input_tokens=10_000, token_counter=_fake_counter)
    for i in range(3):
        session.history.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "tiny",
        })
    session.prefix_cache_warmed = True
    msgs = session.messages("sys")
    assert len(msgs) == 1 + 3
    assert session.prefix_cache_warmed is True


def test_trim_all_tool_messages_emits_event() -> None:
    """Fallback emits on_session_trimmed with fallback marker."""
    from ovs_agent.event_bus import EventBus

    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("on_session_trimmed", lambda payload: received.append(payload))

    session = Session(max_input_tokens=200, token_counter=_fake_counter)
    session.event_bus = bus
    payload = "y" * 80
    for i in range(20):
        session.history.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"{payload}-{i}",
        })

    session.messages("sys")
    assert len(received) == 1
    evt = received[0]
    assert evt.get("fallback") == "all_tool_messages"
    assert evt["dropped_messages"] > 0
    assert evt["budget"] == int(200 * 0.75)


def test_backward_compat_cache_warmed_property() -> None:
    """Old ``session.cache_warmed`` reads/writes still work post-split."""
    session = Session()
    assert session.cache_warmed is False
    assert session.prefix_cache_warmed is False
    session.cache_warmed = True
    assert session.prefix_cache_warmed is True
    assert session.cache_warmed is True
    session.prefix_cache_warmed = False
    assert session.cache_warmed is False
