"""edge-llm-chat-service backend (OpenAI-compatible + prefix-cache hooks)."""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

import httpx
from openai import APIError

from ..session import Session
from .base import LLMEvent
from .openai_compat import LLMStreamError, OpenAICompatBackend

logger = logging.getLogger(__name__)


class _PrefixCacheError(Exception):
    """Raised internally when an upstream failure is identified as
    prefix_cache-specific. We never let this leak out of EdgeLLMBackend —
    it is the trigger for the no-prefix retry path."""


# Substrings that indicate the upstream rejected (only) the prefix_cache
# request. Kept conservative so unrelated 4xx/5xx still propagate to A3
# (transient retry) / A1+A5 (fail-fast).
_PREFIX_CACHE_MARKERS = (
    "prefix_cache",
    "prefix cache",
    "kv cache",
    "kv_cache",
    "kv mismatch",
    "prefix_messages",
)


def _is_prefix_cache_failure(exc: BaseException) -> bool:
    """Heuristic: True when ``exc`` looks like a prefix_cache-only failure.

    Upstream (tensorrt-edge-llm api_server.py L207-L224) returns
    ``JSONResponse(status_code=400, content={"error": str(exc)})`` when
    ``_build_prefix_formatted_request`` raises. That surfaces here as an
    ``APIError`` whose ``str(exc)`` contains the original ValueError text
    (e.g. "prefix_cache requires prefix_messages or at least two messages").
    Mid-stream prefix-cache blowups arrive as ``LLMStreamError``; we apply
    the same substring check.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _PREFIX_CACHE_MARKERS)


class EdgeLLMBackend(OpenAICompatBackend):
    """Adds edge-llm's ``save_system_prompt_kv_cache`` / ``prefix_cache``
    flags, plus an A4 fallback: if the upstream rejects ``prefix_cache``
    we retry the *same* call without it and pin the session so subsequent
    calls also skip prefix_cache (until ``session.reset()`` clears it).

    Important: we deliberately do NOT reuse ``session.cache_warmed`` as
    the disable flag. A successful drain always re-sets ``cache_warmed``,
    which would loop us straight back into ``prefix_cache=True`` on the
    next turn. ``session.prefix_cache_disabled`` is an independent latch.
    """

    def _build_extra_body(self, session: Session | None) -> dict[str, Any]:
        # edge-llm prefix-cache flag matrix:
        #
        # | Scenario                          | prefix_cache | save_system_prompt_kv_cache |
        # |-----------------------------------|--------------|------------------------------|
        # | First-ever request (cold)         |    false     |             true             |
        # | Hot turn (prefix_cache_warmed)    |    true      |             true             |
        # | Multi-iteration tool loop iter>0  |    true      |             true             |
        # | backend.warmup() chat call (B)    |    true      |             true             |
        # | prefix_cache_disabled latched     |    false     |             true             |
        #
        # Semantics on the server side:
        #   * ``prefix_cache=true`` asks the server to LOOK UP an existing
        #     cached prefix matching this request's token prefix and
        #     skip prefill on the hit. Miss falls back to a normal cold
        #     prefill (api_server.py:_build_prefix_formatted_request).
        #   * ``save_system_prompt_kv_cache=true`` asks the server to
        #     SAVE this request's prefix into the cache for future
        #     lookup.
        #   * Sending both makes every turn both CONSUME and PRODUCE
        #     cache entries — which is exactly what the A1 design wants
        #     for multi-turn KV reuse: each turn's history grows by one
        #     turn, the new (larger) prefix is saved, the next turn hits
        #     it.
        #   * Server-side ``save_prefix_cache`` (api_server.py:739-740)
        #     is OR-combined with ``save_system_prompt_kv_cache``; we
        #     currently set only the latter. If a future server release
        #     splits the semantics, set both explicitly here.
        use_prefix_cache = (
            session is not None
            and session.prefix_cache_warmed
            and not session.prefix_cache_disabled
        )
        if use_prefix_cache:
            # A1: also save this turn's prefix so the *next* turn's
            # longer-history request can hit it (server cache is a
            # token-id-prefix map; multiple keys coexist).
            return {
                "prefix_cache": True,
                "save_system_prompt_kv_cache": True,
                "return_cache_metrics": True,
                "enable_thinking": False,
            }
        # Cold path OR warm-but-disabled path. Both ask edge-llm to cache
        # the system prompt KV (cheap, no prefix-formatting risk) and to
        # report cache metrics so the dashboard stays informative.
        return {
            "save_system_prompt_kv_cache": True,
            "return_cache_metrics": True,
            "enable_thinking": False,
        }

    def _disable_prefix_cache(
        self, session: Session | None, exc: BaseException
    ) -> None:
        """Latch ``prefix_cache_disabled`` and notify the event bus."""
        if session is None:
            return
        already = session.prefix_cache_disabled
        session.prefix_cache_disabled = True
        if already:
            return
        bus = getattr(session, "event_bus", None)
        if bus is None:
            return
        try:
            bus.emit(
                "on_prefix_cache_disabled",
                {
                    "reason": str(exc),
                    "sid": getattr(session, "sid", None),
                },
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "event_bus emit on_prefix_cache_disabled failed",
                exc_info=True,
            )

    async def _stream_once(
        self,
        messages: list[dict[str, Any]],
        session: Session | None,
        caller_kw: dict[str, Any],
        *,
        disable_inner_retry: bool = False,
    ) -> AsyncIterator[LLMEvent]:
        """One pass through the base streamer with our cache flags injected.

        ``disable_inner_retry`` is set when invoked from the A4 fallback
        path so that the A3 retry inside ``OpenAICompatBackend.stream`` is
        bypassed for this call — otherwise a prefix_cache failure that
        surfaces as 5xx would trigger A3 retry (1 retry) and *then* the
        A4 fallback would also trigger A3 retry, for a worst case of 4
        upstream calls per turn.
        """
        kw = dict(caller_kw)
        cache_flags = self._build_extra_body(session)
        caller_extra = dict(kw.pop("extra_body", None) or {})
        cache_flags.update(caller_extra)
        kw["extra_body"] = cache_flags
        if disable_inner_retry:
            kw["_retry_disabled"] = True
        async for ev in super().stream_events(messages, **kw):
            yield ev

    async def stream_events(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        session: Session | None = None,
        **kw: Any,
    ) -> AsyncIterator[LLMEvent]:
        used_prefix_cache = (
            session is not None
            and session.prefix_cache_warmed
            and not session.prefix_cache_disabled
        )

        yielded_any = False
        try:
            async for ev in self._stream_once(messages, session, kw):
                yielded_any = True
                yield ev
        except (APIError, LLMStreamError) as exc:
            if used_prefix_cache and _is_prefix_cache_failure(exc):
                # Always latch the flag so future turns skip prefix_cache,
                # even when we can't safely retry this turn (mid-stream
                # failure → tokens already shipped → retry would duplicate).
                self._disable_prefix_cache(session, exc)
                if yielded_any:
                    raise
                logger.warning(
                    "prefix_cache failed (%s); retrying without prefix_cache",
                    exc,
                )
                async for ev in self._stream_once(
                    messages, session, kw, disable_inner_retry=True
                ):
                    yield ev
                if session is not None:
                    session.prefix_cache_warmed = True
                return
            raise

        if session is not None:
            session.prefix_cache_warmed = True

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        session: Session | None = None,
        **kw: Any,
    ) -> AsyncIterator[str]:
        """Back-compat text-only iterator that preserves the ``session=``
        kwarg expected by existing callers (the base ``stream`` filter
        doesn't know about session)."""
        async for ev in self.stream_events(messages, session=session, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text

    # ── warmup ──────────────────────────────────────────────────────
    async def warmup(  # type: ignore[override]
        self,
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        timeout_s: float | None = 60.0,
    ) -> dict[str, Any]:
        """Two-step edge-llm cold-path elimination.

        Step A — POST ``/v1/cache/system_prompt`` with the *messages*
        branch so the cached KV prefix is byte-identical to the prefix
        a real turn will use (server reuses its own
        ``_build_prefix_formatted_request`` slice).

        Step B — POST ``/v1/chat/completions`` with the same prefix,
        ``prefix_cache=true``, ``max_tokens=1``, and the real tools
        list, then drain the SSE stream. The point isn't the output —
        it's to force the TRT-LLM engine to run a full forward pass
        with the real-shape inputs so its CUDA graph capture / kernel
        warm / JIT all happen here, not on the user's first command.

        Fail-open: any error is logged at WARNING and swallowed. The
        first real turn will simply pay cold-start cost.

        Returns a metadata dict (always — empty on full failure):
            {
              "cache_warmed": bool, "cache_warmup_ms": int,
              "graph_warmed": bool, "graph_warmup_ms": int,
              "messages_branch": bool,
              "engine_max_seq_len": int,   # only if /v1/info exposes it
              "prompt_chars": int,         # from cache endpoint metadata
            }
        """
        result: dict[str, Any] = {
            "cache_warmed": False,
            "graph_warmed": False,
            "cache_warmup_ms": 0,
            "graph_warmup_ms": 0,
        }
        if not system_prompt:
            return result
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        cache_url = base + "/v1/cache/system_prompt"
        chat_url = base + "/v1/chat/completions"
        info_url = base + "/v1/info"

        # Best-effort: probe ``/v1/info`` for engine metadata (max_seq_len
        # etc.) so the caller can sanity-check session_max_input_tokens
        # against the actual engine context. Upstream may not implement
        # this endpoint yet — 404/connect-error is silently ignored.
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                info_resp = await client.get(info_url)
                if info_resp.status_code == 200:
                    info_data = info_resp.json() or {}
                    max_seq = info_data.get("max_seq_len") or info_data.get(
                        "engine_max_seq_len"
                    )
                    if isinstance(max_seq, int) and max_seq > 0:
                        result["engine_max_seq_len"] = max_seq
        except Exception:  # pragma: no cover - optional probe
            logger.debug("edge-llm /v1/info probe failed (optional)", exc_info=True)

        # ── Step A: prefix KV cache ────────────────────────────────
        cache_t0 = time.perf_counter()
        cache_body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ""},
            ],
            "prefix_cache": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            cache_body["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=timeout_s or 60.0) as client:
                resp = await client.post(cache_url, json=cache_body)
                resp.raise_for_status()
                try:
                    info = resp.json() or {}
                except Exception:
                    info = {}
            result["cache_warmed"] = True
            result["messages_branch"] = bool(info.get("messages_branch", False))
            result["prompt_chars"] = info.get("prompt_chars")
            # Legacy fallback if server didn't take messages branch.
            if not result["messages_branch"]:
                legacy = {"system_prompt": system_prompt}
                if tools:
                    legacy["tools"] = tools
                try:
                    async with httpx.AsyncClient(timeout=timeout_s or 60.0) as client:
                        resp2 = await client.post(cache_url, json=legacy)
                        resp2.raise_for_status()
                except Exception as exc:
                    logger.debug("cache warmup legacy fallback failed: %s", exc)
        except Exception as exc:
            logger.warning(
                "edge-llm cache warmup failed: %s (first turn pays cold prefill)",
                exc,
            )
        result["cache_warmup_ms"] = int((time.perf_counter() - cache_t0) * 1000)

        # ── Step B: CUDA graph / kernel warm via real-shape decode ──
        # Only attempt if cache_warmed (otherwise prefix_cache=True
        # would be rejected — and a no-prefix call doesn't exercise
        # the same engine path).
        if result["cache_warmed"]:
            graph_t0 = time.perf_counter()
            chat_body: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    # Single-char user message is the cheapest valid
                    # completion request that still hits the real
                    # forward path.
                    {"role": "user", "content": "."},
                ],
                "stream": True,
                "max_tokens": 1,
                "temperature": 0.0,
                "prefix_cache": True,
                "return_cache_metrics": True,
                "enable_thinking": enable_thinking,
            }
            if tools:
                chat_body["tools"] = tools
            try:
                async with httpx.AsyncClient(timeout=timeout_s or 60.0) as client:
                    async with client.stream(
                        "POST", chat_url, json=chat_body
                    ) as resp:
                        resp.raise_for_status()
                        # Drain the SSE stream fully so the engine runs
                        # the whole 1-token forward pass before we move
                        # on. We discard payload — only the side effect
                        # (CUDA graph capture) matters here.
                        async for _ in resp.aiter_lines():
                            pass
                result["graph_warmed"] = True
            except Exception as exc:
                logger.warning(
                    "edge-llm graph warmup failed: %s "
                    "(first turn pays JIT / CUDA-graph capture cost)",
                    exc,
                )
            result["graph_warmup_ms"] = int(
                (time.perf_counter() - graph_t0) * 1000
            )

        return result
