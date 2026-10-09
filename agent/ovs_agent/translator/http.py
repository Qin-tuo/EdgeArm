"""HTTP translator client for CTranslate2 backend."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from .base import TranslatorBackend

logger = logging.getLogger(__name__)


class CTranslate2Translator(TranslatorBackend):
    """HTTP client for remote CTranslate2 translation service."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def translate(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> str:
        """POST to remote translator service."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.post(
                f"{self.base_url}/translate",
                json={
                    "text": text,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                },
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        "Translator service returned %d: %s",
                        resp.status,
                        await resp.text(),
                    )
                    raise RuntimeError(f"Translator service error: {resp.status}")
                data = await resp.json()
                return data["translation"]
        except asyncio.TimeoutError as e:
            logger.error("Translator request timeout after %.1fs", self._timeout)
            raise RuntimeError(f"Translator timeout: {e}") from e
        except aiohttp.ClientError as e:
            logger.error("Translator request failed: %s", e)
            raise RuntimeError(f"Translator service unavailable: {e}") from e

    async def aclose(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
