"""Per-turn SLV resilience: two utterances both complete on one WS.

NOTE (reconnect semantics changed — commit 07c6466 + the multi_utterance
invariant): SLVClient FORCES ``multi_utterance=True`` (slv_client.py:140), so
the WS is ONE persistent connection for the whole App lifetime and does NOT
close after each asr_eos. Proactive reconnect on tts_done was also
INTENTIONALLY removed in 07c6466 (it caused 4429 too_many_sessions); reconnect
now fires only on a session_complete=True final or genuine WS death. The old
"SLV closes WS after each asr_eos → ≥2 reconnects per session" premise
described the pre-multi_utterance behavior that no longer exists. This test
therefore asserts the real healthy contract: two consecutive utterances both
drive a full turn (user + assistant) on a WS that stays alive — i.e. the
session is resilient ACROSS turns without a per-turn reconnect storm.
"""
import asyncio
import time

import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_reconnect_per_turn(test_config):
    audio = ScriptedAudioIO([
        (800,  WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "hello.wav"),
    ])
    async with run_agent(test_config, audio) as (app, probe):
        # Turn 1.
        await probe.wait_event("on_user_utterance", timeout=20)
        # Wait until both utterances have driven a full turn (each producing
        # an assistant_done) — proves the second turn worked on the same WS.
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            u = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
            d = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
            if u >= 2 and d >= 2:
                break
            await asyncio.sleep(0.2)
        u = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
        d = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
        assert u >= 2, f"expected ≥2 utterances, got {u}"
        assert d >= 2, f"expected ≥2 assistant_done (both turns completed), got {d}"
        # The persistent WS must still be healthy after both turns, and there
        # must be NO per-turn reconnect storm (multi_utterance keeps one WS).
        assert app.slv.is_healthy(), "SLV WS should still be healthy after 2 turns"
        r = sum(1 for e in probe.events if e.get("event") == "on_slv_reconnect")
        assert r < 2, (
            f"expected the WS to persist across turns (no per-turn reconnect "
            f"storm), but saw {r} reconnects"
        )
