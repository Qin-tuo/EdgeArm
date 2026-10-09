"""LiveCaptionApp — real-time bilingual captions from streaming ASR.

Pipeline: ASR partial → SegmentCommitter (retranslation) → translate →
``on_translation`` broadcast (rendered side-by-side on the dashboard). No TTS,
no LLM.

The committed prefix is locked; the volatile tail is re-translated (debounced)
and re-emitted so the caption refreshes as the user keeps speaking. On ASRFinal
the remaining tail is force-committed with full-utterance context.

With ``translator_backend: noop`` the "translation" is a pass-through, so this
same app doubles as a **pure transcription** view (original == translated).
"""
from __future__ import annotations

import asyncio

from ovs_agent import BaseApp
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin
from ovs_agent.streaming_translate import SegmentCommitter
from ovs_agent.translator import asr_lang_to_flores


class LiveCaptionApp(BaseApp):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Dashboard WS is how the /caption page receives on_translation events.
        self.register(DebugDashboardPlugin(self))
        self._committer = SegmentCommitter(
            agreement_n=config.committer_agreement_n,
            strategy="retranslation",
            min_commit_chars=config.committer_min_commit_chars,
        )
        self._debounce_s = max(0.0, config.translate_debounce_ms / 1000.0)
        self._committed_source = ""
        self._committed_translation = ""
        self._tail_task: asyncio.Task | None = None

    # ── language / translation helpers ───────────────────────────────
    def _src_lang(self, detected_language: str | None) -> str:
        return asr_lang_to_flores(detected_language) or self.config.translator_src_lang

    async def _translate(self, text: str, detected_language: str | None) -> str:
        if not text.strip():
            return ""
        return await self.translator.translate(
            text, self._src_lang(detected_language), self.config.translator_tgt_lang
        )

    async def _emit(
        self,
        *,
        detected_language: str | None,
        is_final: bool,
        tail_source: str = "",
        tail_translation: str = "",
    ) -> None:
        await self._broadcast(
            "on_translation",
            {
                "original": self._committed_source,
                "translated": self._committed_translation,
                "tail_original": tail_source,
                "tail_translated": tail_translation,
                "src_lang": self._src_lang(detected_language),
                "tgt_lang": self.config.translator_tgt_lang,
                "detected_language": detected_language,
                "is_final": is_final,
            },
        )

    # ── BaseApp hooks ────────────────────────────────────────────────
    async def on_user_partial(self, text: str, detected_language: str | None = None) -> None:
        for ev in self._committer.push_partial(text, detected_language):
            if ev.kind == "commit":
                await self._handle_commit(ev, detected_language)
            else:  # "tail" — volatile preview, debounced
                self._schedule_tail(ev, detected_language)

    async def on_user_utterance(self, text: str, detected_language: str | None = None) -> None:
        self._cancel_tail()
        for ev in self._committer.finalize(text, detected_language):
            await self._handle_commit(ev, detected_language)
        # per-utterance accumulators reset for the next sentence
        self._committed_source = ""
        self._committed_translation = ""

    # ── internals ────────────────────────────────────────────────────
    async def _handle_commit(self, ev, detected_language: str | None) -> None:
        self._cancel_tail()
        translated = await self._translate(ev.source_text, detected_language)
        self._committed_source = ev.committed_source
        self._committed_translation += translated
        await self._emit(detected_language=detected_language, is_final=ev.is_final)

    def _schedule_tail(self, ev, detected_language: str | None) -> None:
        self._cancel_tail()
        self._tail_task = asyncio.create_task(
            self._tail_after_debounce(ev.tail_source, detected_language)
        )

    async def _tail_after_debounce(self, tail_source: str, detected_language: str | None) -> None:
        try:
            if self._debounce_s:
                await asyncio.sleep(self._debounce_s)
            translated = await self._translate(tail_source, detected_language)
            await self._emit(
                detected_language=detected_language,
                is_final=False,
                tail_source=tail_source,
                tail_translation=translated,
            )
        except asyncio.CancelledError:
            pass

    def _cancel_tail(self) -> None:
        if self._tail_task is not None and not self._tail_task.done():
            self._tail_task.cancel()
        self._tail_task = None


App = LiveCaptionApp

__all__ = ["App", "LiveCaptionApp"]
