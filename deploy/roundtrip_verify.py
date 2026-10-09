#!/usr/bin/env python3
"""TTS -> ASR round-trip verifier for deployed OpenVoiceStream services."""

from __future__ import annotations

import argparse
import json
import mimetypes
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize(text: str) -> str:
    drop = set(" \t\r\n。，、！？；：,.!?;:\"'()[]{}<>《》“”‘’")
    return "".join(ch.lower() for ch in text if ch not in drop)


def lcs_similarity(a: str, b: str) -> float:
    a = normalize(a)
    b = normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / max(len(a), len(b))


def request_json(url: str, timeout: float) -> dict:
    with OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return resp.read()


def pcm_stream_to_wav(payload: bytes) -> bytes:
    if len(payload) < 4:
        raise RuntimeError(f"TTS stream returned too little data: {len(payload)} bytes")
    sample_rate = struct.unpack("<I", payload[:4])[0]
    pcm = payload[4:]
    out = BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return out.getvalue()


def wav_duration_s(wav_bytes: bytes) -> float:
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        rate = reader.getframerate()
        frames = reader.getnframes()
    return frames / rate if rate > 0 else 0.0


def post_multipart_file(url: str, field: str, path: Path, timeout: float) -> dict:
    boundary = f"openvoicestream-{time.time_ns()}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + path.read_bytes() + footer
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8621")
    parser.add_argument("--text", default="你好，今天天气真不错。")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--min-sim", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--keep-audio", type=Path)
    parser.add_argument("--streaming", action="store_true", help="use /tts/stream and wrap raw PCM as WAV")
    parser.add_argument("--min-audio-sec", type=float, default=0.0)
    parser.add_argument("--expect-asr-segmented", action="store_true")
    parser.add_argument("--max-failed-segments", type=int, default=0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    try:
        health = request_json(f"{base}/health", args.timeout_sec)
        if not health.get("tts"):
            raise RuntimeError(f"TTS is not ready: {health}")
        if not health.get("asr"):
            raise RuntimeError(f"ASR is not ready: {health}")

        if args.streaming:
            stream_payload = post_json(f"{base}/tts/stream", {"text": args.text}, args.timeout_sec)
            wav = pcm_stream_to_wav(stream_payload)
        else:
            wav = post_json(f"{base}/tts", {"text": args.text}, args.timeout_sec)
        if len(wav) < 1000:
            raise RuntimeError(f"TTS returned too little audio: {len(wav)} bytes")
        duration_s = wav_duration_s(wav)
        if duration_s < args.min_audio_sec:
            raise RuntimeError(
                f"TTS audio duration {duration_s:.2f}s < required {args.min_audio_sec:.2f}s"
            )

        if args.keep_audio:
            args.keep_audio.write_bytes(wav)
            wav_path = args.keep_audio
            cleanup = False
        else:
            tmp = NamedTemporaryFile(prefix="openvoicestream-roundtrip-", suffix=".wav", delete=False)
            tmp.write(wav)
            tmp.close()
            wav_path = Path(tmp.name)
            cleanup = True

        try:
            asr = post_multipart_file(f"{base}/asr?language={args.language}", "file", wav_path, args.timeout_sec)
        finally:
            if cleanup:
                wav_path.unlink(missing_ok=True)

        text = str(asr.get("text", "")).strip()
        sim = lcs_similarity(args.text, text)
        result = {
            "url": base,
            "tts_bytes": len(wav),
            "expected": args.text,
            "asr_text": text,
            "similarity": round(sim, 4),
            "audio_duration_s": round(duration_s, 3),
            "asr_segmented": bool(asr.get("segmented")),
            "asr_segment_count": asr.get("segment_count"),
            "asr_failed_segments": asr.get("failed_segments"),
            "health": health,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not text:
            raise RuntimeError("ASR returned empty text")
        if sim < args.min_sim:
            raise RuntimeError(f"similarity {sim:.4f} < required {args.min_sim:.4f}")
        if args.expect_asr_segmented and not asr.get("segmented"):
            raise RuntimeError(f"ASR did not report segmented=true: {asr}")
        failed_segments = int(asr.get("failed_segments", 0) or 0)
        if failed_segments > args.max_failed_segments:
            raise RuntimeError(
                f"ASR failed_segments {failed_segments} > allowed {args.max_failed_segments}"
            )
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"roundtrip verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
