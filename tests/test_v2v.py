#!/usr/bin/env python3
"""Qwen3 TTS + Qwen3 ASR V2V round-trip test.

Uses separate subprocesses to avoid OOM on 8GB Jetson Orin Nano.
Requires: LD_LIBRARY_PATH with ORT CUDA, CUDA 12, nvidia libs.

Usage:
    python3 tests/test_v2v.py
"""

import subprocess, sys, os, time, json

ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = ":".join([
    os.path.expanduser("~/ort-from-container/lib"),
    "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
    "/usr/lib/aarch64-linux-gnu/nvidia",
    "/usr/lib/aarch64-linux-gnu",
    ENV.get("LD_LIBRARY_PATH", ""),
])
ENV["QWEN3_MODEL_BASE"] = os.path.expanduser("~/voice_test/models/qwen3-tts")
ENV["QWEN3_ASR_MODEL_BASE"] = os.path.expanduser("~/voice_test/models/qwen3-asr-v2")
ENV["PYTHONPATH"] = os.path.expanduser("~/voice_test/app_overlay")

TTS_SCRIPT = os.path.expanduser("~/tts_proc.py")
ASR_SCRIPT = os.path.expanduser("~/asr_proc.py")

# ── Subprocess scripts (pushed to device separately) ─────────────────────
# tts_proc.py: loads Qwen3TRTBackend, synthesizes text, saves WAV, prints JSON meta
# asr_proc.py: loads Qwen3ASRBackend, transcribes WAV, prints JSON result

TESTS = [
    ("zh_short", "你好，今天天气不错。"),
    ("zh_med",  "欢迎使用语音合成系统。"),
    ("zh_long", "今天天气真不错，我们一起去公园散步吧。"),
]

def main():
    print("=" * 60)
    print("Qwen3 TTS + Qwen3 ASR V2V Round-Trip")
    print("=" * 60)

    results = []
    for label, text in TESTS:
        print()
        print("[%s] %s" % (label, text))
        wav_path = "/tmp/v2v_qwen3.wav"

        # TTS subprocess
        print("  TTS...", end=" ", flush=True)
        r = subprocess.run(
            [sys.executable, TTS_SCRIPT, text, wav_path],
            env=ENV, capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print("FAILED (exit=%d)\nSTDERR: %s" % (r.returncode, r.stderr[-500:]))
            break
        lines = [l for l in r.stdout.strip().split("\n") if l.strip().startswith("{")]
        meta = json.loads(lines[-1]) if lines else {}
        print("%.1fs audio (%.1fs wall)" % (meta.get("duration_s", 0), meta.get("wall_s", 0)))

        # ASR subprocess
        print("  ASR...", end=" ", flush=True)
        r = subprocess.run(
            [sys.executable, ASR_SCRIPT, wav_path],
            env=ENV, capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print("FAILED (exit=%d)\nSTDERR: %s" % (r.returncode, r.stderr[-500:]))
            break
        lines = [l for l in r.stdout.strip().split("\n") if l.strip().startswith("{")]
        asr_out = json.loads(lines[-1]) if lines else {"text": "", "wall_s": 0}
        match = text[:6] in asr_out["text"]
        print("%r (%.1fs) %s" % (asr_out["text"], asr_out["wall_s"], "MATCH" if match else "DIFF"))
        results.append({"input": text, "output": asr_out["text"], "match": match})

    print()
    matches = sum(1 for r in results if r["match"])
    print("Summary: %d/%d matches" % (matches, len(results)))
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
