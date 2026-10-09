"""SimulInterpretApp — simultaneous(-ish) speech interpretation.

Pipeline: ASR partial → SegmentCommitter (monotonic) → translate → TTS.
Monotonic means a committed clause is never retracted (spoken audio can't be
un-said). No LLM.

``overlap_mode``:
  - ``"off"`` (default, clause-lag): translate each clause incrementally during
    partials (low translate latency) but SPEAK the whole utterance once on
    ASRFinal — the proven, robust TTS path. Works on any hardware; the listener
    is ~one sentence behind.
  - ``"on"`` (full-duplex overlap, experimental): speak each clause as it
    commits, while the user keeps talking. Needs an AEC device / headphones; the
    EchoFilter drops self-echo partials that the hardware AEC leaks.

Barge-in is disabled (translation must never self-interrupt); with the gate off,
partials keep flowing even while TTS plays (required for overlap).
"""
from __future__ import annotations

from ovs_agent import BaseApp
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin
from ovs_agent.streaming_translate import EchoFilter, SegmentCommitter
from ovs_agent.translator import asr_lang_to_flores


class SimulInterpretApp(BaseApp):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Dashboard WS feeds the /caption page (optional subtitle for interpret).
        self.register(DebugDashboardPlugin(self))
        self._committer = SegmentCommitter(
            agreement_n=config.committer_agreement_n,
            strategy="monotonic",
            min_commit_chars=config.committer_min_commit_chars,
        )
        self._overlap = str(getattr(config, "overlap_mode", "off")).lower() == "on"
        self._echo = (
            EchoFilter(
                threshold=config.echo_similarity_threshold,
                window_s=config.echo_window_s,
            )
            if getattr(config, "echo_filter_enabled", True)
            else None
        )
        self._pending_translation = ""
        self._pending_source = ""

    # ── helpers ──────────────────────────────────────────────────────
    def _src_lang(self, detected_language: str | None) -> str:
        return asr_lang_to_flores(detected_language) or self.config.translator_src_lang

    async def _translate(self, text: str, detected_language: str | None) -> str:
        if not text.strip():
            return ""
        return await self.translator.translate(
            text, self._src_lang(detected_language), self.config.translator_tgt_lang
        )

    async def _speak_now(self, translated: str, source: str, detected_language: str | None) -> None:
        if not translated.strip():
            return
        if self._echo is not None:
            self._echo.add_tts(translated)
        await self.slv.send_text(translated)
        await self.slv.flush_tts()
        await self._broadcast(
            "on_translation",
            {
                "original": source,
                "translated": translated,
                "src_lang": self._src_lang(detected_language),
                "tgt_lang": self.config.translator_tgt_lang,
                "detected_language": detected_language,
                "is_final": False,
            },
        )

    # ── BaseApp hooks ────────────────────────────────────────────────
    async def on_user_partial(self, text: str, detected_language: str | None = None) -> None:
        if self._echo is not None and self._echo.is_echo(text):
            return  # our own translation leaking back through the mic
        for ev in self._committer.push_partial(text, detected_language):
            translated = await self._translate(ev.source_text, detected_language)
            if self._overlap:
                await self._speak_now(translated, ev.source_text, detected_language)
            else:
                self._pending_translation += translated
                self._pending_source = ev.committed_source

    async def on_user_utterance(self, text: str, detected_language: str | None = None) -> None:
        if self._echo is not None and self._echo.is_echo(text):
            self._committer.finalize(text, detected_language)
            self._reset_pending()
            return
        for ev in self._committer.finalize(text, detected_language):
            translated = await self._translate(ev.source_text, detected_language)
            if self._overlap:
                await self._speak_now(translated, ev.source_text, detected_language)
            else:
                self._pending_translation += translated
                self._pending_source = ev.committed_source
        if not self._overlap and self._pending_translation.strip():
            await self._flush_pending(detected_language)
        self._reset_pending()

    # ── internals ────────────────────────────────────────────────────
    async def _flush_pending(self, detected_language: str | None) -> None:
        if self._echo is not None:
            self._echo.add_tts(self._pending_translation)
        await self.slv.send_text(self._pending_translation)
        await self.slv.flush_tts()
        await self._broadcast(
            "on_translation",
            {
                "original": self._pending_source,
                "translated": self._pending_translation,
                "src_lang": self._src_lang(detected_language),
                "tgt_lang": self.config.translator_tgt_lang,
                "detected_language": detected_language,
                "is_final": True,
            },
        )

    def _reset_pending(self) -> None:
        self._pending_translation = ""
        self._pending_source = ""


App = SimulInterpretApp

__all__ = ["App", "SimulInterpretApp"]
