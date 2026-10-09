"""Abstract translator backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TranslatorBackend(ABC):
    """Text translation backend."""

    @abstractmethod
    async def translate(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> str:
        """Translate text from src_lang to tgt_lang. Returns translated string."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any held network/transport resources. Default: no-op."""
        return None
