#!/usr/bin/env python3
"""Benchmark Paraformer TRT streaming ASR over /asr/stream."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
import wave
from pathlib import Path
from urllib import request

import numpy as np
import websockets


SAMPLE_RATE = 16000
CHUNK_MS = 400


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        with wave.open(str(path), "rb") as wav:
            sr = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
        if width != 2:
            raise RuntimeError(f"{path}: unsupported PCM sample width {width}")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    new_len = int(round(len(audio) * dst_sr / src_sr))
    if new_len <= 0:
        return np.array([], dtype=np.float32)
    x_old = np.linspace(0, len(audio) - 1, num=len(audio))
    x_new = np.linspace(0, len(audio) - 1, num=new_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


async def recv_until_idle(ws, send_t0: float, timeout_s: float) -> list[dict]:
    messages: list[dict] = []
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        except asyncio.TimeoutError:
            break
        latency_ms = (time.perf_counter() - send_t0) * 1000.0
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            msg = {"raw": raw}
        msg["_latency_ms"] = latency_ms
        messages.append(msg)
        if msg.get("is_final") or msg.get("type") == "final":
            break
    return messages


async def bench_one(ws_url: str, wav_path: Path, real_time: bool) -> dict:
    audio, sr = read_wav(wav_path)
    audio = resample_linear(audio, sr, SAMPLE_RATE)
    pcm = to_pcm16(audio)
    chunk_samples = SAMPLE_RATE * CHUNK_MS // 1000

    chunk_rows: list[dict] = []
    partials: list[dict] = []
    final_text = ""
    first_partial_ms = None
    stream_t0 = time.perf_counter()

    async with websockets.connect(ws_url, max_size=8 * 1024 * 1024) as ws:
        for idx, start in enumerate(range(0, len(pcm), chunk_samples)):
            chunk = pcm[start:start + chunk_samples]
            send_t0 = time.perf_counter()
            await ws.send(chunk.tobytes())
            msgs = await recv_until_idle(ws, send_t0, timeout_s=0.03)
            if real_time:
                elapsed = time.perf_counter() - send_t0
                await asyncio.sleep(max(0.0, CHUNK_MS / 1000.0 - elapsed))

            if not msgs:
                chunk_rows.append({
                    "wav": wav_path.name,
                    "chunk": idx,
                    "latency_ms": "",
                    "type": "",
                    "text": "",
                })
                continue

            for msg in msgs:
                text = msg.get("text", "")
                latency_ms = msg["_latency_ms"]
                msg_type = msg.get("type", "")
                partials.append(msg)
                if text and first_partial_ms is None:
                    first_partial_ms = (time.perf_counter() - stream_t0) * 1000.0
                if msg.get("is_final") or msg_type == "final":
                    final_text = text
                chunk_rows.append({
                    "wav": wav_path.name,
                    "chunk": idx,
                    "latency_ms": f"{latency_ms:.2f}",
                    "type": msg_type,
                    "text": text,
                })

        send_t0 = time.perf_counter()
        await ws.send(b"")
        final_msgs = await recv_until_idle(ws, send_t0, timeout_s=10.0)
        for msg in final_msgs:
            text = msg.get("text", "")
            if msg.get("is_final") or msg.get("type") == "final":
                final_text = text
            chunk_rows.append({
                "wav": wav_path.name,
                "chunk": "final",
                "latency_ms": f"{msg['_latency_ms']:.2f}",
                "type": msg.get("type", ""),
                "text": text,
            })

    latencies = [
        float(r["latency_ms"])
        for r in chunk_rows
        if r["chunk"] != "final" and r["latency_ms"] != ""
    ]
    return {
        "wav": str(wav_path),
        "duration_s": len(audio) / SAMPLE_RATE,
        "chunk_count": int(math.ceil(len(pcm) / chunk_samples)),
        "chunk_p50_ms": percentile(latencies, 50),
        "chunk_p95_ms": percentile(latencies, 95),
        "first_partial_ms": first_partial_ms,
        "partial_count": len([p for p in partials if p.get("type") == "partial"]),
        "final_text": final_text,
        "rows": chunk_rows,
    }


def offline_asr(http_url: str, wav_path: Path) -> str:
    boundary = "----paraformer-bench-boundary"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{wav_path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
    )
    body.extend(wav_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request(
        http_url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("text", "")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://orin-nx:18000/asr/stream")
    parser.add_argument("--offline-url", default="")
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("wavs", nargs="+", type=Path)
    args = parser.parse_args()

    results = []
    all_rows = []
    for wav_path in args.wavs:
        result = await bench_one(args.ws_url, wav_path, real_time=not args.no_real_time)
        if args.offline_url:
            result["offline_text"] = offline_asr(args.offline_url, wav_path)
        results.append({k: v for k, v in result.items() if k != "rows"})
        all_rows.extend(result["rows"])
        print(json.dumps(results[-1], ensure_ascii=False))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if args.csv_out:
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav", "chunk", "latency_ms", "type", "text"])
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
