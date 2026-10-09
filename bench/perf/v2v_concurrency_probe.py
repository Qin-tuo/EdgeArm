#!/usr/bin/env python3
"""Concurrency probe for /v2v/stream ASR path.

Drives N concurrent /v2v/stream sessions (ASR-only, realtime=False) with the
same short WAV, records absolute monotonic timestamps for the ASR-compute
window (asr_endpoint -> asr_final) per session, and reports whether the
windows overlap (true concurrency) vs serialize.

Also supports an over-subscription test: launch N_total clients against a
server with max_slots=N to confirm the (N+1)th gets a 4429-style reject
(WS close) rather than being queued.

Run INSIDE the container (websockets + numpy available):
  python3 /opt/speech/bench/perf/v2v_concurrency_probe.py \
      --url ws://localhost:8000 --wav <wav> --n 2
"""
import argparse
import asyncio
import json
import time
import wave
import sys

import numpy as np
import websockets


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    assert sw == 2, f"expected 16-bit PCM, got sampwidth={sw}"
    pcm = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        pcm = pcm.reshape(-1, 2)[:, 0].copy()
    return pcm.tobytes(), sr


def chunks_of(pcm_bytes, sr, chunk_ms=250):
    frames_per = int(sr * chunk_ms / 1000)
    bytes_per = frames_per * 2
    return [pcm_bytes[i:i + bytes_per] for i in range(0, len(pcm_bytes), bytes_per)]


async def one_session(idx, url, pcm_bytes, sr, language, vad_silence_ms,
                      chunk_ms, t0, result):
    """Run a single /v2v/stream ASR session. realtime=False (no pacing)."""
    chunks = chunks_of(pcm_bytes, sr, chunk_ms)
    silence_ms = vad_silence_ms + chunk_ms
    silence_chunks = int(np.ceil(silence_ms / chunk_ms))
    frames_per = int(sr * chunk_ms / 1000)
    silence = np.zeros(frames_per, dtype=np.int16).tobytes()

    rec = {
        "idx": idx, "text": "", "error": None,
        "t_open": None, "t_endpoint": None, "t_final": None,
    }
    result.append(rec)
    try:
        async with websockets.connect(f"{url}/v2v/stream", max_size=None,
                                       open_timeout=30, close_timeout=10) as ws:
            rec["t_open"] = time.monotonic() - t0
            await ws.send(json.dumps({
                "type": "config",
                "asr_language": language,
                "vad": "silero",
                "vad_silence_ms": vad_silence_ms,
                "sample_rate": sr,
            }))
            # pump audio (no realtime pacing)
            for c in chunks:
                await ws.send(c)
            # trailing silence for VAD endpoint
            for _ in range(silence_chunks):
                await ws.send(silence)

            async def reader():
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    if isinstance(raw, bytes):
                        continue
                    data = json.loads(raw)
                    typ = data.get("type")
                    if typ == "asr_endpoint":
                        rec["t_endpoint"] = time.monotonic() - t0
                    elif typ == "asr_final":
                        rec["t_final"] = time.monotonic() - t0
                        t = data.get("text", "")
                        if isinstance(t, dict):
                            t = t.get("text", "")
                        rec["text"] = t
                        return
            await reader()
    except websockets.exceptions.ConnectionClosed as e:
        rec["error"] = f"closed code={e.code} reason={e.reason!r}"
    except Exception as e:  # noqa
        rec["error"] = f"{type(e).__name__}: {e}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--n", type=int, default=2, help="concurrent sessions")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--vad-silence-ms", type=int, default=400)
    ap.add_argument("--chunk-ms", type=int, default=250)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pcm, sr = load_wav(args.wav)
    dur = len(pcm) / 2 / sr
    print(f"[probe{(' '+args.label) if args.label else ''}] wav={args.wav} "
          f"dur={dur:.2f}s sr={sr} n={args.n}")

    t0 = time.monotonic()
    result = []
    tasks = [
        asyncio.create_task(one_session(
            i, args.url, pcm, sr, args.language,
            args.vad_silence_ms, args.chunk_ms, t0, result))
        for i in range(args.n)
    ]
    await asyncio.gather(*tasks)
    result.sort(key=lambda r: r["idx"])

    print("\n=== per-session results (ms relative to test start) ===")
    windows = []
    for r in result:
        def ms(v):
            return f"{v*1000:8.1f}" if v is not None else "    None"
        print(f"  s{r['idx']}: open={ms(r['t_open'])} "
              f"endpoint={ms(r['t_endpoint'])} final={ms(r['t_final'])} "
              f"err={r['error']} text={r['text']!r}")
        if r["t_endpoint"] is not None and r["t_final"] is not None:
            windows.append((r["idx"], r["t_endpoint"], r["t_final"]))

    # overlap analysis on ASR-compute windows (endpoint -> final)
    if len(windows) >= 2:
        print("\n=== ASR-compute window overlap (endpoint -> final) ===")
        for (ia, sa, ea), (ib, sb, eb) in [
            (windows[i], windows[j])
            for i in range(len(windows)) for j in range(i + 1, len(windows))
        ]:
            overlap = max(0.0, min(ea, eb) - max(sa, sb))
            wa = ea - sa
            wb = eb - sb
            print(f"  s{ia}[{sa*1000:.1f}->{ea*1000:.1f}] ({wa*1000:.1f}ms) "
                  f"vs s{ib}[{sb*1000:.1f}->{eb*1000:.1f}] ({wb*1000:.1f}ms): "
                  f"overlap={overlap*1000:.1f}ms "
                  f"({'OVERLAP(concurrent)' if overlap > 0 else 'DISJOINT(serial)'})")
        starts = sorted(w[1] for w in windows)
        ends = sorted(w[2] for w in windows)
        span = max(ends) - min(starts)
        sum_w = sum(w[2] - w[1] for w in windows)
        print(f"\n  total wall span (first endpoint -> last final): {span*1000:.1f}ms")
        print(f"  sum of individual windows:                       {sum_w*1000:.1f}ms")
        print(f"  ratio span/sum = {span/sum_w:.2f}  "
              f"(near 1.0/max-window => concurrent; near 1.0*sum => serial)")


if __name__ == "__main__":
    asyncio.run(main())
