#!/usr/bin/env python3
"""kokoro_trt N=2 production stability gate.

Mirrors the accepted qwen3_trt N=2 gate (commit range ff92458..8d6d1b2):
- N=1 baseline TTFA
- pre-stress MD5
- 30+ sustained N=2 dual-client bursts
- post-stress MD5
- CUDA error log scan
- ratio gate (N=2 combined TTFA p50 / N=1 TTFA p50 <= 1.5)

The script does NOT import TensorRT or CUDA. It talks to /tts/stream and
/health over HTTP and reads service logs through `docker logs --since`
or a plain file path.

Fallback when this gate fails: set `OVS_TTS_STREAM_MAX_WORKERS_KOKORO=1`
(backend-specific) to force single-slot Kokoro without affecting other
TTS backends. If the deployed image predates that env var, fall back to
the global `OVS_TTS_STREAM_MAX_WORKERS=1`.

Mac dry-run: `python bench/perf/stability_kokoro_n2.py --mock` writes a
synthetic PASS report without hitting any server. Use this to validate
the CLI surface; never use --mock output as production evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stability_tts_n2_common import main_entry  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_entry(
        backend_label="kokoro_trt",
        expected_backend_substr=("kokoro",),
        fallback_env_var="OVS_TTS_STREAM_MAX_WORKERS_KOKORO",
        # Kokoro is English-focused; spec §Edge Cases line 348-349 picks en.
        default_lang="en",
    ))
