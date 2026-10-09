"""End-to-end tests for live_caption + simul_interpret against the live SLV.

Spawns the real app (BaseApp subclass) pointed at the orin-nx SLV, injects a
scripted Chinese utterance, and asserts the streaming-translate pipeline runs.

Uses ``translator_backend=noop`` so no NLLB service is required: on_translation
then carries the recognized text verbatim (transcription path), which exercises
ASR → SegmentCommitter → on_translation (live_caption) and
ASR → SegmentCommitter → TTS (simul_interpret) without a translator dependency.
"""
from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager

import pytest

from .conftest import SLV_URL, WAV_DIR
from .fake_audio import ScriptedAudioIO
from .probe import AgentProbe

# NLLB translator service URL for the real-translation e2e (set up separately).
TRANSLATOR_URL = os.getenv("OVS_TRANSLATOR_URL", "http://localhost:9001")


def _translator_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(TRANSLATOR_URL + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _app_config(app_extra: dict | None = None):
    from ovs_agent.config import Config, _default_slv_config

    slv_cfg = _default_slv_config()
    slv_cfg.update({"vad": "none", "asr_language": "auto"})
    fields = dict(
        slv_url=SLV_URL,
        slv_config=slv_cfg,
        llm_backend="noop",
        translator_backend="noop",
        translator_src_lang="zho_Hans",
        translator_tgt_lang="eng_Latn",
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
        client_vad_backend="energy",
        client_vad_threshold=0.005,
        client_vad_speech_min_ms=200,
        client_vad_silence_ms=600,
        client_vad_drive_eos=True,
        barge_in_enabled=False,
        metadata={"dashboard_port": _free_port()},
    )
    fields.update(app_extra or {})
    return Config(**fields)


@asynccontextmanager
async def _run(app, audio: ScriptedAudioIO):
    app.audio = audio
    run_task = asyncio.create_task(app.run(), name="app-run")
    probe = AgentProbe(port=app.config.metadata["dashboard_port"])
    try:
        await probe.connect()
        yield app, probe
    finally:
        try:
            app.request_shutdown()
        except Exception:
            pass
        try:
            await audio.close()
        except Exception:
            pass
        try:
            await probe.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(run_task, timeout=8)
        except (asyncio.TimeoutError, Exception):
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_live_caption_emits_translation_events():
    from ovs_agent.apps.live_caption.app import LiveCaptionApp

    cfg = _app_config()
    audio = ScriptedAudioIO([(800, WAV_DIR / "weather.wav")])
    async with _run(LiveCaptionApp(cfg), audio) as (app, probe):
        # ASR should produce partials → committer → on_translation events.
        await probe.wait_event("on_user_utterance", timeout=25)
        ev = await probe.wait_event("on_translation", is_final=True, timeout=15)
        data = ev.get("data") or {}
        assert (data.get("original") or "").strip(), (
            f"expected non-empty caption original; got {data!r}"
        )
        # noop translator → translated mirrors original (transcription view).
        assert data.get("translated") == data.get("original")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _translator_up(), reason=f"NLLB translator not reachable at {TRANSLATOR_URL}"
)
async def test_live_caption_real_translation():
    """ctranslate2 backend → captions carry an ACTUAL translation (zh→en)."""
    from ovs_agent.apps.live_caption.app import LiveCaptionApp

    cfg = _app_config(
        {
            "translator_backend": "ctranslate2",
            "translator_url": TRANSLATOR_URL,
            "translator_src_lang": "zho_Hans",
            "translator_tgt_lang": "eng_Latn",
        }
    )
    audio = ScriptedAudioIO([(800, WAV_DIR / "weather.wav")])  # 今天天气怎么样
    async with _run(LiveCaptionApp(cfg), audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=25)
        ev = await probe.wait_event("on_translation", is_final=True, timeout=20)
        d = ev.get("data") or {}
        original = (d.get("original") or "").strip()
        translated = (d.get("translated") or "").strip()
        assert original, f"expected non-empty original; got {d!r}"
        assert translated, f"expected non-empty translation; got {d!r}"
        assert translated != original, "translation must differ from source"
        # zh→en output should be (mostly) Latin/ASCII, not Chinese.
        assert translated.isascii(), f"expected English output; got {translated!r}"


@pytest.mark.asyncio
async def test_simul_interpret_produces_tts():
    from ovs_agent.apps.simul_interpret.app import SimulInterpretApp

    cfg = _app_config({"overlap_mode": "off"})
    audio = ScriptedAudioIO([(800, WAV_DIR / "weather.wav")])
    async with _run(SimulInterpretApp(cfg), audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=25)
        # noop "translation" of the recognized Chinese is spoken back via TTS.
        for _ in range(60):
            if len(audio.captured_tts) > 1000:
                break
            await asyncio.sleep(0.25)
        assert len(audio.captured_tts) > 1000, (
            f"expected TTS PCM bytes > 1000, got {len(audio.captured_tts)}"
        )


@pytest.mark.asyncio
async def test_simul_interpret_overlap_speaks_per_clause():
    """overlap_mode=on: clauses are spoken DURING partials (is_final=False
    on_translation events), not buffered to a single final flush — and SLV
    actually produces TTS from those mid-utterance send_text calls."""
    from ovs_agent.apps.simul_interpret.app import SimulInterpretApp

    cfg = _app_config({"overlap_mode": "on"})
    # Longer utterance → more partials → committer commits clauses mid-stream.
    audio = ScriptedAudioIO([(800, WAV_DIR / "story_request.wav")])
    async with _run(SimulInterpretApp(cfg), audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=30)
        # Give the overlap speaks + TTS a moment to flush through SLV.
        for _ in range(80):
            if len(audio.captured_tts) > 1000:
                break
            await asyncio.sleep(0.25)
        # Overlap path emits per-clause is_final=False on_translation events.
        evs = probe.events
        final_idx = next(
            (i for i, e in enumerate(evs) if e.get("event") == "on_user_utterance"),
            len(evs),
        )
        previews = [
            i for i, e in enumerate(evs)
            if e.get("event") == "on_translation"
            and isinstance(e.get("data"), dict)
            and e["data"].get("is_final") is False
        ]
        assert previews, "expected per-clause (is_final=False) overlap speaks"
        # At least one clause must be spoken DURING partials, i.e. before the
        # ASRFinal/on_user_utterance — that's what distinguishes overlap from
        # the off-mode single final flush.
        assert previews[0] < final_idx, (
            "expected a clause spoken before ASRFinal (during partials); "
            f"first preview idx={previews[0]} not before final idx={final_idx}"
        )
        assert len(audio.captured_tts) > 1000, (
            f"expected TTS PCM bytes > 1000 from overlap speaks, got {len(audio.captured_tts)}"
        )
