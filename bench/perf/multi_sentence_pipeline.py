"""TTS multi-sentence single-user benchmark for pipeline parallelism gate.

Sends a multi-sentence request to /tts/stream and measures:
 - TTFA (first PCM byte past SR header)
 - per-sentence "audible duration" via chunk arrival timing
 - total wall clock
 - audio MD5 (full body PCM)

The pipeline-parallelism change should:
 - keep TTFA (sentence 1's TTFA) unchanged
 - keep audio MD5 IDENTICAL to baseline (same audio output, just reordered timing)
 - reduce total wall clock by ~10-30% (prefetch overlap of next-sentence prefill)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import requests

TEXT_MULTI = (
    "我们都非常震惊。这位母亲表示。"
    "今天天气真不错，适合出门散步。"
    "人工智能正在改变我们的生活方式。"
    "请问您需要什么帮助吗？"
    "感谢您的关注和支持。"
)
TEXT_SINGLE = "我们都非常震惊。这位母亲表示。"


def run_once(host: str, text: str, verbose: bool = False) -> dict:
    url = f"http://{host}/tts/stream"
    t0 = time.perf_counter()
    r = requests.post(url, json={"text": text}, stream=True, timeout=120)
    r.raise_for_status()
    ttfa_ms = None
    body = bytearray()
    chunk_timings = []
    for chunk in r.iter_content(chunk_size=4096):
        if not chunk:
            continue
        body.extend(chunk)
        now_ms = (time.perf_counter() - t0) * 1000
        chunk_timings.append((now_ms, len(chunk)))
        # TTFA: first time body has > 4 bytes (i.e. past SR header)
        if ttfa_ms is None and len(body) > 4:
            ttfa_ms = now_ms
    total_ms = (time.perf_counter() - t0) * 1000
    pcm = bytes(body[4:])  # strip 4-byte SR header
    return {
        "ttfa_ms": ttfa_ms,
        "total_ms": total_ms,
        "pcm_bytes": len(pcm),
        "audio_md5": hashlib.md5(pcm).hexdigest(),
        "chunks": chunk_timings if verbose else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="100.82.225.102:8621")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--text", choices=["multi", "single"], default="multi")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = TEXT_MULTI if args.text == "multi" else TEXT_SINGLE
    nsent = text.count("。") + text.count("？") + text.count("！")
    print(f"=== TTS pipeline bench: {args.text} ({nsent} sentences) × {args.runs} runs @ {args.host} ===")
    print(f"text len={len(text)} chars")

    results = []
    for i in range(1, args.runs + 1):
        r = run_once(args.host, text, args.verbose)
        results.append(r)
        print(f"run {i}: ttfa={r['ttfa_ms']:.1f}ms  total={r['total_ms']:.1f}ms  "
              f"pcm={r['pcm_bytes']}B  md5={r['audio_md5']}")
        if args.verbose and r["chunks"]:
            for t, sz in r["chunks"][:8]:
                print(f"   chunk @ {t:.1f}ms  size={sz}")
            print(f"   ... ({len(r['chunks'])} chunks total)")

    # Summary
    ttfas = [r["ttfa_ms"] for r in results if r["ttfa_ms"]]
    totals = [r["total_ms"] for r in results]
    md5s = set(r["audio_md5"] for r in results)
    print("\n=== Summary ===")
    print(f"  ttfa  min/median/max: {min(ttfas):.1f} / {sorted(ttfas)[len(ttfas)//2]:.1f} / {max(ttfas):.1f} ms")
    print(f"  total min/median/max: {min(totals):.1f} / {sorted(totals)[len(totals)//2]:.1f} / {max(totals):.1f} ms")
    print(f"  audio md5(s): {md5s}")
    if len(md5s) == 1:
        print(f"  GATE PASS: all {args.runs} runs produced identical audio")
    else:
        print(f"  GATE FAIL: audio MD5 varies across runs (non-deterministic?)")
        sys.exit(1)


if __name__ == "__main__":
    main()
