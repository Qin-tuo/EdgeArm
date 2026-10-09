"""Per-conversation Session state.

History is normally append-only (edge-llm's prefix cache rewards stable
prefixes), but edge-llm engines have a hard ``max_seq_len`` ceiling
(e.g. 3072 tokens for engines-3072). Past that point the server crashes
with a CUDA fatal abort. We therefore support an *optional* token-aware
trim: when ``max_input_tokens`` is set, the oldest *full turns* are
dropped before the prompt is shipped to the LLM, while always preserving
the system prompt + the most recent user/assistant pair.

When ``max_input_tokens`` is ``None`` (the default), behaviour matches
the original invariant — no trimming, prefix cache stays warm.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Module-level tokenizer cache so we only pay HF load cost once per
# model name across the whole process (and across Session instances).
_TOKENIZER_CACHE: dict[str, Any] = {}


# Sentinel used to cache "tokenizer load failed; use conservative
# char-based fallback" (see _fallback_estimate for the actual formula).
_FALLBACK = object()


def _get_tokenizer(model_name: str) -> Any:
    """Lazy, cached HuggingFace tokenizer loader.

    Returns ``_FALLBACK`` (and warns once) when transformers isn't
    installed or the model can't be loaded — callers must handle this
    by falling back to ``_fallback_estimate`` (``ceil(chars * 1.5)``).
    We deliberately don't raise: a missing tokenizer should degrade
    gracefully, not crash the voice loop.
    """
    if model_name not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer  # local import: heavy
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(
                "tokenizer load failed for %r (%s); using char*1.5 fallback",
                model_name, e,
            )
            _TOKENIZER_CACHE[model_name] = _FALLBACK
    return _TOKENIZER_CACHE[model_name]


def _fallback_estimate(text: str) -> int:
    """Conservative char-based estimator.

    Use ``ceil(chars * 1.5)`` rather than ``chars // 4`` because Chinese
    text averages ~1.5 tokens per character (not 0.25). A too-low estimate
    causes long Chinese prompts to slip past the trim/guard and crash the
    engine on ``max_seq_len`` overflow.
    """
    return max(1, math.ceil(len(text) * 1.5))


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort token count. Falls back to a conservative char estimate."""
    if tokenizer is _FALLBACK:
        return _fallback_estimate(text)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:  # pragma: no cover - defensive
        return _fallback_estimate(text)


@dataclass
class Session:
    sid: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    locale: str = "zh"
    # Split warmup flags (was single ``cache_warmed`` field, kept as a
    # backward-compat property below):
    #   * prefix_cache_warmed — server holds a KV-prefix entry we can
    #     reuse on the next request (``prefix_cache=True``). Set after
    #     a successful warmup Step A OR a successful real turn (which
    #     also saves its prefix). Cleared by session reset / trim /
    #     echo recovery.
    #   * graph_warmed — TRT-LLM CUDA graph captured + JIT done for the
    #     real-shape forward path. Independent of prefix_cache (the
    #     server can have one without the other after a partial warmup
    #     failure). Currently informational only: edge_llm doesn't
    #     change the request based on this flag; the runner logs a hint
    #     when prefix is hot but graph isn't, because the first tool_call
    #     decode after a partial warmup may still pay JIT cost.
    prefix_cache_warmed: bool = False
    graph_warmed: bool = False
    # Token-aware trim controls (A2). None disables trimming entirely
    # and preserves the original append-only invariant.
    max_input_tokens: int | None = None
    tokenizer_model: str = "Qwen/Qwen3-4B-AWQ"
    # Optional override for tokenization (used by tests to avoid pulling
    # a real model from HF). Signature: (text: str) -> int.
    token_counter: Callable[[str], int] | None = None
    # A4 placeholder: when True, the LLM backend should skip prefix
    # cache hints. This task only wires the field; logic lands in A4.
    prefix_cache_disabled: bool = False
    # Optional EventBus injected by the app. Plain attribute (not a
    # dataclass field) so tests can construct Sessions cheaply without
    # importing the EventBus type.
    event_bus: Any = None

    # ── backward-compat alias ───────────────────────────────────────
    # Old field was a single ``cache_warmed`` bool that conflated prefix
    # cache and CUDA graph warm. We split into two; keep the old name
    # as a property targeting prefix_cache_warmed (the half that
    # actually drives ``prefix_cache=True`` on requests). Older callers
    # and tests that read or assign ``session.cache_warmed`` keep working.
    @property
    def cache_warmed(self) -> bool:
        return self.prefix_cache_warmed

    @cache_warmed.setter
    def cache_warmed(self, value: bool) -> None:
        self.prefix_cache_warmed = bool(value)

    def reset(self) -> None:
        """Clear conversation history and per-session cache latches.

        Called when a new dialogue starts (mode switch / explicit clear /
        user-driven reset). Both warmup flags and the A4
        ``prefix_cache_disabled`` flag get reset so the next turn can
        re-warm normally — otherwise a session that ever hit a prefix-cache
        failure would skip prefix_cache forever, even across logical
        conversations.
        """
        self.history.clear()
        self.prefix_cache_warmed = False
        self.graph_warmed = False
        self.prefix_cache_disabled = False

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._maybe_recover_from_echo()

    def add_assistant_tool_calls(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Append an assistant message that issues one or more tool_calls.

        ``content`` may be ``None`` (pure tool-call turn) or a short
        preamble ("好的，我查一下…"). We append ``content`` explicitly
        rather than omitting it — OpenAI's wire format expects the key
        to be present with an explicit ``null`` when there's no text."""
        self.history.append({
            "role": "assistant",
            "content": content,
            "tool_calls": list(tool_calls),
        })

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a ``role:tool`` message linked to a prior tool_call."""
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def rollback_to(self, anchor: int) -> int:
        """Truncate ``self.history`` back to ``anchor`` length.

        Returns the number of messages dropped. Idempotent when
        ``anchor >= len(history)``. Used on cancel / error / iteration
        cap to keep history strict-valid for OpenAI (no orphan
        ``assistant(tool_calls)`` without matching ``role:tool``
        followup, no truncated tool round).
        """
        if anchor < 0:
            anchor = 0
        current = len(self.history)
        if anchor >= current:
            return 0
        dropped = current - anchor
        del self.history[anchor:]
        return dropped

    # In-context learning latch: when the model (small / quantised) sees
    # a few short identical assistant replies in history, it pattern-matches
    # and keeps emitting that same canned response for every subsequent
    # turn no matter what the user actually said. Detect and self-recover.
    ECHO_WINDOW = 3
    ECHO_MAX_LEN = 40

    def _maybe_recover_from_echo(self) -> None:
        """Wipe history when the assistant has just emitted the same short
        reply ``ECHO_WINDOW`` times in a row — strong signal that an
        in-context echo loop is locked in.

        We restrict to short replies (``ECHO_MAX_LEN`` chars) because two
        long natural answers being byte-identical is implausible in normal
        conversation, but a hard cap is still nice insurance against very
        long deterministic re-runs ever counting as 'real' history.
        Keeps ``sid`` / ``metadata`` so dashboards don't get confused;
        clears ``cache_warmed`` because we just invalidated the prefix.
        """
        if len(self.history) < self.ECHO_WINDOW:
            return
        # Skip tool-call-only assistant messages (content is None) — they
        # carry no natural-language reply to echo on.
        assistant_turns = [
            m for m in self.history
            if m.get("role") == "assistant" and m.get("content") is not None
        ]
        if len(assistant_turns) < self.ECHO_WINDOW:
            return
        last_n = assistant_turns[-self.ECHO_WINDOW:]
        texts = [m.get("content", "") for m in last_n]
        first = texts[0]
        if len(first) > self.ECHO_MAX_LEN:
            return
        if any(t != first for t in texts):
            return
        logger.warning(
            "echo loop detected: %d consecutive identical assistant turns "
            "(text=%r); auto-clearing history",
            self.ECHO_WINDOW, first[:60],
        )
        self.history.clear()
        self.prefix_cache_warmed = False
        self.graph_warmed = False
        self.prefix_cache_disabled = False
        bus = getattr(self, "event_bus", None)
        if bus is not None:
            try:
                bus.emit(
                    "on_echo_recovery",
                    {
                        "window": self.ECHO_WINDOW,
                        "echo_text": first[:120],
                        "sid": self.sid,
                    },
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("event_bus emit on_echo_recovery failed", exc_info=True)

    def messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Return OpenAI-format messages with system prompt prepended.

        When ``max_input_tokens`` is set, oldest turns are dropped so the
        total token budget stays within ``max_input_tokens * 0.75``
        (25% margin reserved for the response).
        """
        base: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *self.history,
        ]
        if self.max_input_tokens is None:
            return base
        return self._trim_to_budget(base, self.max_input_tokens)

    # ── internal ────────────────────────────────────────────────────

    def _count(self, text: str) -> int:
        if self.token_counter is not None:
            return self.token_counter(text)
        tok = _get_tokenizer(self.tokenizer_model)
        return _count_tokens(tok, text)

    def _msg_tokens(self, msg: dict[str, Any]) -> int:
        # Add a small per-message overhead (~4 tokens) to mirror
        # OpenAI's chat template framing — keeps the estimate honest
        # without needing the full chat template here. For tool-call
        # carrying messages we also charge an estimate against the
        # JSON-serialized tool_calls payload so a fat function-call
        # turn isn't undercounted.
        content = msg.get("content") or ""
        tokens = self._count(content) if content else 0
        tcs = msg.get("tool_calls")
        if tcs:
            try:
                import json as _json
                tokens += self._count(_json.dumps(tcs, ensure_ascii=False))
            except Exception:  # pragma: no cover - defensive
                pass
        return tokens + 4

    def _trim_to_budget(
        self, messages: list[dict[str, Any]], max_tokens: int
    ) -> list[dict[str, Any]]:
        """Drop oldest *whole turns* until the dynamic history fits the budget.

        A *turn* is a ``user`` message followed by all contiguous
        non-user messages up to (but not including) the next ``user``.
        With tools, a turn can contain multiple
        ``assistant(tool_calls)`` + ``tool`` pairs followed by a final
        ``assistant(text)``. The trim never splits a turn — either the
        whole thing stays or the whole thing is dropped.

        A trailing turn whose final message is not a normal
        ``assistant(text)`` (e.g. user just sent a message and the
        assistant hasn't replied, or an in-flight tool round) is pinned
        to the tail and never dropped.

        Budget semantics (A1-step2):
          * Budget = ``max_tokens * 0.75`` and applies ONLY to dynamic
            turns (user / assistant / tool messages).
          * The system prompt (messages[0]) is a fixed prefix and is
            NOT charged against the budget. Tools schemas, which the
            LLM backend prepends outside of ``Session.history``, are
            likewise out of scope here.
          * This decouples trim decisions from system_prompt / tools
            growth: when those grow we adjust ``session_max_input_tokens``
            (or the engine ``max_seq_len``) at config time, not at trim
            time.

        Invariants:
          * messages[0] (system prompt) always kept.
          * Latest turn always kept.
        """
        budget = int(max_tokens * 0.75)
        if not messages:
            return messages

        system = messages[0]
        rest = list(messages[1:])
        if not rest:
            return messages

        # Group into turns: each starts at a `user` message and runs up to
        # the next `user` (exclusive). Defensively skip leading non-user
        # entries (orphan assistant/tool) — those couldn't form a valid
        # turn anyway.
        i = 0
        while i < len(rest) and rest[i].get("role") != "user":
            i += 1
        turns: list[list[dict[str, Any]]] = []
        while i < len(rest):
            start = i
            i += 1  # skip the user message itself
            while i < len(rest) and rest[i].get("role") != "user":
                i += 1
            turns.append(rest[start:i])

        # Corner case (Plan D item 6): no user-anchored turns at all
        # (history is exclusively assistant/tool messages — can happen
        # if an in-flight tool round was rolled back leaving orphan
        # entries, or a programmatic test populates only tool results).
        # In this case the regular path returns the input unchanged,
        # so an oversized all-tool history would silently slip past the
        # budget and overflow ``max_seq_len`` upstream. Degrade by
        # dropping oldest non-system messages until under budget, log
        # ERROR so the underlying invariant violation gets noticed.
        if not turns:
            rest_tokens = sum(self._msg_tokens(m) for m in rest)
            if rest_tokens <= budget:
                return messages
            logger.error(
                "session trim corner case: history has no user-anchored "
                "turns but total %d tokens > budget %d. Falling back to "
                "raw drop-oldest until under budget — investigate why "
                "history is all assistant/tool (orphan rollback?).",
                rest_tokens, budget,
            )
            kept = list(rest)
            dropped_msgs = 0
            while kept and rest_tokens > budget:
                removed = kept.pop(0)
                rest_tokens -= self._msg_tokens(removed)
                dropped_msgs += 1
            # Clear cache_warmed: prefix the upstream KV cache was keyed
            # against is no longer present.
            if self.prefix_cache_warmed:
                self.prefix_cache_warmed = False
            bus = getattr(self, "event_bus", None)
            if bus is not None:
                try:
                    bus.emit(
                        "on_session_trimmed",
                        {
                            "dropped_messages": dropped_msgs,
                            "kept_messages": len(kept),
                            "approx_tokens": rest_tokens,
                            "budget": budget,
                            "max_input_tokens": max_tokens,
                            "sid": self.sid,
                            "fallback": "all_tool_messages",
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.debug(
                        "event_bus emit on_session_trimmed (fallback) failed",
                        exc_info=True,
                    )
            return [system, *kept]

        # A trailing "incomplete" turn = doesn't end in a normal
        # assistant(text) message. Pin it; only completed turns are
        # candidates for dropping.
        trailing: list[dict[str, Any]] = []
        if turns:
            last = turns[-1]
            last_msg = last[-1] if last else None
            is_complete = (
                last_msg is not None
                and last_msg.get("role") == "assistant"
                and last_msg.get("content") is not None
                and not last_msg.get("tool_calls")
            )
            if not is_complete:
                trailing = turns.pop()

        # A1-step2: budget covers only dynamic turns (history). The
        # system prompt is a fixed prefix and is excluded so config
        # changes there don't silently shift trim behaviour.
        trailing_tokens = sum(self._msg_tokens(m) for m in trailing)
        turn_tokens = [sum(self._msg_tokens(m) for m in t) for t in turns]
        total = trailing_tokens + sum(turn_tokens)

        if total <= budget:
            return messages

        # Drop from the front (oldest) until under budget. Stop when:
        #   * only one completed turn remains AND no trailing turn (must
        #     keep at least one completed turn for context), or
        #   * trailing turn exists AND no completed turns left (we can
        #     drop everything completed; trailing is already pinned).
        kept_turns = list(turns)
        dropped = 0
        min_keep = 0 if trailing else 1
        while total > budget and len(kept_turns) > min_keep:
            removed = kept_turns.pop(0)
            total -= sum(self._msg_tokens(m) for m in removed)
            dropped += 1

        kept_count = len(kept_turns)
        approx_tokens = total

        if dropped > 0:
            logger.warning(
                "session trimmed: dropped %d turns, kept %d turns, ~%d tokens",
                dropped, kept_count, approx_tokens,
            )
            # MED-2: trimming changes the prefix that the upstream KV cache
            # is keyed against. If we leave ``cache_warmed=True`` the next
            # turn will send ``prefix_cache=True`` and the server-side KV
            # cache (built for the pre-trim history) will mismatch — A4
            # would then catch the resulting failure and pin the session
            # to no-prefix-cache for the rest of the conversation. That's
            # a wasted upstream round-trip *and* a permanent latch from
            # an avoidable mismatch. Clearing the flag here means the
            # very next call goes cold (saves the system prompt KV again)
            # without ever attempting the doomed prefix_cache path.
            #
            # We deliberately do NOT touch ``prefix_cache_disabled``: that
            # latch tracks "the engine reject prefix_cache for this
            # session" — independent of trim. Conflating them would mean
            # a single trim event permanently disables prefix_cache for
            # the rest of the dialogue, which we don't want.
            if self.prefix_cache_warmed:
                logger.info(
                    "session trim invalidates upstream KV cache, "
                    "clearing prefix_cache_warmed (prefix_cache_disabled untouched)"
                )
                self.prefix_cache_warmed = False
            bus = getattr(self, "event_bus", None)
            if bus is not None:
                try:
                    bus.emit(
                        "on_session_trimmed",
                        {
                            "dropped_turns": dropped,
                            "kept_turns": kept_count,
                            "approx_tokens": approx_tokens,
                            "budget": budget,
                            "max_input_tokens": max_tokens,
                            "sid": self.sid,
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.debug(
                        "event_bus emit on_session_trimmed failed", exc_info=True
                    )

        rebuilt: list[dict[str, str]] = [system]
        for t in kept_turns:
            rebuilt.extend(t)
        rebuilt.extend(trailing)
        return rebuilt
