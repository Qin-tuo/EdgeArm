"""No-op LLM backend (for translator-only apps)."""
from __future__ import annotations

from typing import Any, AsyncIterator

from .base import LLMBackend, LLMEvent


class NoopLLM(LLMBackend):
    """LLM backend that yields nothing (used when LLM is not needed)."""

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        if False:  # pragma: no cover
            yield LLMEvent(kind="text", text="")

    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        if False:  # pragma: no cover
            yield ""
