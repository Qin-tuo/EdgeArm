"""V2V barge-in latency probe.

Drives /v2v/stream against a running voice service, sends TTS-trigger
text, then after first audio chunk + ~300ms sends an abort frame. Measures
elapsed time between the abort send and the last audio byte received
(silence for 500ms = trailing-end). Runs N trials and prints p50 / p95 /
max.

Standalone driver — only depends on websockets + stdlib. Intended to be
run on the host (NOT inside the docker container) and hit the published
port of the voice service.

Env:
    V2V_URL    full ws:// URL (default ws://localhost:8621/v2v/stream)
    TRIALS     number of trials (default 10)
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

import websockets


URL = os.environ.get("V2V_URL", "ws://localhost:8621/v2v/stream")
TRIALS = int(os.environ.get("TRIALS", "10"))

# Long enough that TTS keeps generating audio past our abort point.
TRIGGER_TEXT = (
    "请你详细地讲一讲人工智能在最近十年里的发展历程，"
    "尤其是大语言模型、多模态模型、以及在边缘设备上的部署进展，"
    "尽可能讲得详细一些，包括代表性模型、关键技术、典型应用场景。"
)

# After abort, we treat the stream as "done" when audio has been idle for
# this many ms.
IDLE_DONE_MS = 500
# Wait before sending abort after first audio chunk arrives.
ABORT_DELAY_MS = 300
# Per-trial hard cap.
TRIAL_TIMEOUT_S = 20.0


async def one_trial(idx: int) -> float | None:
    """Return abort_to_last_audio_ms, or None on failure."""
    try:
        async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:
            # 1. Send config
            await ws.send(json.dumps({
                "type": "config",
                "asr_language": "zh",
                "tts_language": "zh",
                "vad": "silero",
            }))

            # 2. Send TTS trigger text (text-only barge-in scenario: drive TTS)
            await ws.send(json.dumps({"type": "text", "text": TRIGGER_TEXT}))

            # 3. Wait for first binary audio chunk
            first_audio_ts = None
            deadline = time.perf_counter() + TRIAL_TIMEOUT_S
            while first_audio_ts is None:
                if time.perf_counter() > deadline:
                    print(f"[trial {idx}] timeout waiting for first audio")
                    return None
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                if isinstance(msg, (bytes, bytearray)):
                    first_audio_ts = time.perf_counter()

            # 4. Wait ABORT_DELAY_MS then send abort
            await asyncio.sleep(ABORT_DELAY_MS / 1000.0)
            abort_sent_ts = time.perf_counter()
            await ws.send(json.dumps({"type": "abort"}))

            # 5. Drain — track last audio byte timestamp. Done when idle.
            last_audio_ts = abort_sent_ts
            while True:
                idle_budget = IDLE_DONE_MS / 1000.0
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=idle_budget)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    last_audio_ts = time.perf_counter()
                # text frames (events / acks) are ignored

            abort_to_last_audio_ms = (last_audio_ts - abort_sent_ts) * 1000.0
            return abort_to_last_audio_ms
    except Exception as e:
        print(f"[trial {idx}] exception: {e!r}")
        return None


async def main() -> int:
    print(f"V2V barge-in latency probe  URL={URL}  trials={TRIALS}")
    results: list[float] = []
    for i in range(TRIALS):
        r = await one_trial(i)
        if r is None:
            continue
        print(f"  trial {i+1:2d}: abort_to_last_audio_ms = {r:7.1f}")
        results.append(r)
        # Tiny breather between trials so the server fully drains state.
        await asyncio.sleep(0.5)

    if not results:
        print("NO SUCCESSFUL TRIALS")
        return 1

    results.sort()
    n = len(results)
    p50 = results[n // 2] if n % 2 == 1 else statistics.median(results)
    p95 = results[min(n - 1, int(round(0.95 * (n - 1))))]
    mx = max(results)
    print(
        f"\nSUMMARY  n={n}/{TRIALS}  "
        f"p50={p50:.1f}ms  p95={p95:.1f}ms  max={mx:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
