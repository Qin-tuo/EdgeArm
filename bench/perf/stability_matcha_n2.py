#!/usr/bin/env python3
"""matcha_trt N=2 production stability gate.

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

Fallback when this gate fails: set `OVS_TTS_STREAM_MAX_WORKERS_MATCHA=1`
(backend-specific) to force single-slot Matcha without affecting other
TTS backends. If the deployed image predates that env var, fall back to
the global `OVS_TTS_STREAM_MAX_WORKERS=1`.

Matcha advertises MULTI_LANGUAGE — both zh and en prompts are used by
default. Matcha token truncation kicks in at ~80 tokens
(app/backends/jetson/matcha_trt.py:625-630); the MD5 prompt should stay
short enough to avoid accidental truncation noise. Use `--prompt-id
zh_short_01` (or en_short_01) for the byte-comparison capture.

Mac dry-run: `python bench/perf/stability_matcha_n2.py --mock` writes a
synthetic PASS report without hitting any server.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stability_tts_n2_common import main_entry  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_entry(
        backend_label="matcha_trt",
        expected_backend_substr=("matcha",),
        fallback_env_var="OVS_TTS_STREAM_MAX_WORKERS_MATCHA",
        # MULTI_LANGUAGE capability: do not filter by lang by default.
        default_lang=None,
    ))
