#!/usr/bin/env python3
"""Streaming WS check for the opt-in punctuation + speaker-embedding features.

Connects to /asr/stream, streams a WAV as PCM16, and prints every final
payload. Run it twice (flags off vs on) to confirm:
  - default (no flags): final.text has NO injected fields (regression guard);
  - ?punctuate=true&speaker_embedding=true: final.text is punctuated AND the
    payload carries {speaker_embedding, embedding_model, dim, normalized}.

Usage:
  python3 punct_speaker_ws_check.py --url ws://localhost:8000/asr/stream \
      --wav zh_short_01.wav --language zh --punct --speaker

Dependencies: websockets (already in OVS images).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets


def load_pcm16(path: str):
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "expect PCM16 wav"
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    if ch > 1:
        # naive downmix to mono int16
        import numpy as np
        a = np.frombuffer(frames, dtype="<i2").reshape(-1, ch).mean(axis=1)
        frames = a.astype("<i2").tobytes()
    return frames, sr


async def run(url: str, wav: str, language: str, punct: bool, speaker: bool, chunk_ms: int):
    pcm, sr = load_pcm16(wav)
    qs = [f"language={language}", f"sample_rate={sr}"]
    if punct:
        qs.append("punctuate=true")
    if speaker:
        qs.append("speaker_embedding=true")
    full_url = url + ("&" if "?" in url else "?") + "&".join(qs)
    print(f"[connect] {full_url}")
    bytes_per_chunk = int(sr * 2 * chunk_ms / 1000)
    finals = []
    async with websockets.connect(full_url, max_size=None) as ws:
        async def sender():
            for i in range(0, len(pcm), bytes_per_chunk):
                await ws.send(pcm[i:i + bytes_per_chunk])
                await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(b"")  # EOF

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(msg, bytes):
                    continue
                obj = json.loads(msg)
                if obj.get("type") == "final" or obj.get("is_final"):
                    finals.append(obj)
                    emb = obj.get("speaker_embedding")
                    shown = dict(obj)
                    if emb:
                        shown["speaker_embedding"] = emb[:24] + f"...(len={len(emb)})"
                    print("[final]", json.dumps(shown, ensure_ascii=False))
                    if obj.get("type") == "final" and not obj.get("endpoint") == "vad":
                        # EOS/forced final → done
                        break
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            pass
        finally:
            send_task.cancel()
    # Summary / assertions
    print("\n=== SUMMARY ===")
    print(f"finals: {len(finals)}")
    for f in finals:
        has_emb = "speaker_embedding" in f
        print(f"  text={f.get('text')!r} | has_embedding={has_emb}"
              + (f" dim={f.get('dim')} model={f.get('embedding_model')} norm={f.get('normalized')}" if has_emb else ""))
    return finals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/asr/stream")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--punct", action="store_true")
    ap.add_argument("--speaker", action="store_true")
    ap.add_argument("--chunk-ms", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.wav, args.language, args.punct, args.speaker, args.chunk_ms))
