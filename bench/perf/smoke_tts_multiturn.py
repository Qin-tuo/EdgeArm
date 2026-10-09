"""Smoke test: multi-turn TTS lifecycle on a single WS.

This test exposes the **multi-turn TTS sticky-flush bug** in
``app/main.py``'s ``/v2v/stream`` handler:

- ``state["tts_flush"]`` was set to ``True`` on the first
  ``CLIENT_TTS_FLUSH`` and never reset.
- ``tts_out_task`` exited on ``tts_flush and queue empty`` after round 1
  and emitted ``SERVER_TTS_DONE``, then returned.
- With both work tasks returned, ``asyncio.gather`` unblocked and the WS
  closed — even in ``multi_utterance=True`` mode.

Symptom on the wire: client sends text+flush for round 2, receives no
audio, then the WS dies. OVS upstream's workaround is to close+reopen
WS every turn, which is heavy-handed and contradicts the documented
``multi_utterance=True`` contract.

This test runs 3 rounds of {text → tts_flush → drain → wait for
``tts_done``} on ONE WebSocket and asserts:

1. Each round receives a non-trivial number of PCM bytes (more than the
   4-byte SR header).
2. Each per-turn ``tts_done`` is non-terminal:
   ``session_complete`` is ``False`` (after fix) or absent
   (legacy lenient path). The test rejects ``session_complete=True``
   before round 3 because that would imply the server is closing the
   session early.
3. The WS stays alive between rounds.

After round 3 the test sends ``asr_eos`` to close the session, then
waits for either a session-final ``tts_done`` (with
``session_complete=True`` after fix) or WS close — both are accepted.

This test is NOT auto-run. The operator invokes it manually after
deploying the fix to a live device. Default endpoint matches the
operator's ``/tmp/wstest4.py`` pattern: ``orin-nx:8621``.

Usage::

    python bench/perf/smoke_tts_multiturn.py
    python bench/perf/smoke_tts_multiturn.py --host orin-nx:8621
    python bench/perf/smoke_tts_multiturn.py --host localhost:8621 --verbose

Requirements: ``pip install websocket-client`` (no audio fixtures
needed — the bug is TTS-side; we drive synthesis via ``{type:text}``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import websocket  # websocket-client


DEFAULT_HOST = "orin-nx:8621"
DEFAULT_TEXTS = [
    "你好，这是第一轮测试。",
    "现在进入第二轮，看看 TTS 是否还活着。",
    "第三轮，验证多轮会话完整收尾。",
]


def _recv_until_tts_done(ws: websocket.WebSocket, *, timeout: float, verbose: bool):
    """Drain frames until a tts_done JSON arrives or timeout fires.

    Returns (pcm_bytes_total, tts_done_payload_or_None,
             other_jsons_list, ws_closed_bool).
    """
    pcm_total = 0
    tts_done_payload: dict[str, Any] | None = None
    others: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    ws_closed = False

    while time.monotonic() < deadline and tts_done_payload is None:
        ws.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            msg = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except (websocket.WebSocketConnectionClosedException, ConnectionError, OSError):
            ws_closed = True
            break
        if msg is None or msg == "":
            ws_closed = True
            break
        if isinstance(msg, (bytes, bytearray)):
            pcm_total += len(msg)
            if verbose:
                print(f"  [pcm] +{len(msg)}B (total {pcm_total}B)")
            continue
        try:
            data = json.loads(msg)
        except (ValueError, TypeError):
            continue
        if verbose:
            print(f"  [json] {data}")
        t = data.get("type")
        if t == "tts_done":
            tts_done_payload = data
        else:
            others.append(data)

    return pcm_total, tts_done_payload, others, ws_closed


def run_smoke(host: str, *, timeout: float = 30.0, verbose: bool = False) -> int:
    """Returns exit code (0 = pass, non-zero = fail)."""
    ws_url = f"ws://{host}/v2v/stream"
    print(f"[smoke] connecting to {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=timeout)
    failures: list[str] = []
    try:
        cfg = {
            "type": "config",
            "asr_language": "Chinese",
            "tts_language": "zh",
            "sample_rate": 16000,
            "vad": "none",            # text-only path; no audio needed
            "multi_utterance": True,
        }
        ws.send(json.dumps(cfg))

        for i, text in enumerate(DEFAULT_TEXTS, start=1):
            print(f"\n[smoke] === round {i}/{len(DEFAULT_TEXTS)} ===")
            ws.send(json.dumps({"type": "text", "text": text}))
            ws.send(json.dumps({"type": "tts_flush"}))

            pcm, done, others, closed = _recv_until_tts_done(
                ws, timeout=timeout, verbose=verbose,
            )
            print(f"[smoke] round {i}: pcm={pcm}B, tts_done={done}, "
                  f"other_jsons={len(others)}, ws_closed={closed}")

            if closed:
                failures.append(f"round {i}: WS closed mid-round")
                break
            if done is None:
                failures.append(f"round {i}: never received tts_done within {timeout}s")
                break
            if pcm <= 4:
                # 4 == just the sample-rate header, no actual audio
                failures.append(f"round {i}: got only {pcm}B PCM (expected > 4)")
            sc = done.get("session_complete")
            if i < len(DEFAULT_TEXTS) and sc is True:
                failures.append(
                    f"round {i}: tts_done.session_complete=True before final round"
                )
            # Lenient: accept False (post-fix) or absent (legacy single-utt path).

        # Close out the session.
        print("\n[smoke] sending asr_eos to close session")
        try:
            ws.send(json.dumps({"type": "asr_eos"}))
        except Exception as e:
            print(f"[smoke] asr_eos send failed (may be ok if WS already closing): {e}")

        # Drain anything trailing (final tts_done w/ session_complete=true,
        # or simply server-side close). Both are acceptable.
        final_pcm, final_done, _, final_closed = _recv_until_tts_done(
            ws, timeout=5.0, verbose=verbose,
        )
        print(f"[smoke] session close: pcm={final_pcm}B, "
              f"tts_done={final_done}, ws_closed={final_closed}")
    finally:
        try: ws.close()
        except Exception: pass

    if failures:
        print("\n[smoke] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[smoke] PASS — 3-round multi-utterance TTS lifecycle is healthy")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"host:port of /v2v/stream endpoint (default {DEFAULT_HOST})")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="per-round drain timeout in seconds")
    p.add_argument("--verbose", action="store_true",
                   help="print every wire frame")
    args = p.parse_args()
    sys.exit(run_smoke(args.host, timeout=args.timeout, verbose=args.verbose))


if __name__ == "__main__":
    # NOTE: not auto-run by CI. Operator invokes manually after device deploy.
    # This exists to expose the multi-turn TTS sticky-flush bug against a
    # live WS endpoint.
    main()
