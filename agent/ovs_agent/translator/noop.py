"""Pass-through translator (no-op)."""
from __future__ import annotations

from .base import TranslatorBackend


class NoopTranslator(TranslatorBackend):
    """Returns input text unchanged."""

    async def translate(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> str:
        return text
