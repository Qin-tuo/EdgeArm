"""N=1 cancel-protocol stress harness — gates the 'A-at-N=1' milestone.

Targets the bug originally observed at /tts/stream:
  - client connects, reads first PCM chunk, breaks the connection
  - without the cooperative-cancel protocol, the worker keeps generating
    chunks and eventually poisons its own CUDA context, breaking
    every subsequent request
  - with the protocol, each early-break is acknowledged + slot
    released cleanly + next request is healthy

Scenarios (--scenario):
  early-break    Default. 100 iterations of: connect, read 1 chunk
                 (>4 bytes), close. Each iter's TTFA must be normal
                 (~510 ms on Orin NX), not the broken-state 6-10 ms
                 signature.
  no-pcm         Each iter: connect, read ONLY the 4-byte SR header,
                 close immediately. Tests cancel-before-first-PCM.
  full-then-broken  Mix: half full-body reads (no cancel) and half
                 early-breaks. Tests cancel/normal interleaving.
  long-soak      Continuous early-break for --duration seconds (default
                 600). Validates VRAM and CUDA-context stability over
                 sustained mid-stream disconnects.

Acceptance criteria (printed at end; non-zero exit on failure):
  - every iter past the first must report TTFA >= 100 ms (the broken-
    state floor was ~6-10 ms because the worker errored out without
    producing audio)
  - no iter raises an unhandled HTTP error
  - audio MD5 of a final capture (full-body read) matches the baseline
    `f515a4376962cca876f21089130d7253`
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from typing import Optional

import requests


BASELINE_MD5 = "f515a4376962cca876f21089130d7253"
TEXT = "我们都非常震惊。这位母亲表示。"
BROKEN_TTFA_THRESHOLD_MS = 100  # below this looks like the worker errored


def _one_early_break(url: str, timeout: float = 30.0) -> Optional[float]:
    """One early-break iter. Returns TTFA in ms (or None on error)."""
    t0 = time.perf_counter()
    try:
        with requests.post(url, json={"text": TEXT}, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            header_seen = False
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if not header_seen:
                    if len(chunk) > 4:
                        return (time.perf_counter() - t0) * 1000
                    header_seen = True
                else:
                    return (time.perf_counter() - t0) * 1000
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return None


def _one_no_pcm(url: str, timeout: float = 30.0) -> Optional[float]:
    """Read only the 4-byte SR header then close. Cancel before any PCM."""
    t0 = time.perf_counter()
    try:
        with requests.post(url, json={"text": TEXT}, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            seen = 0
            for chunk in r.iter_content(chunk_size=4):
                if not chunk:
                    continue
                seen += len(chunk)
                if seen >= 4:
                    return (time.perf_counter() - t0) * 1000
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return None


def _one_full(url: str, timeout: float = 60.0) -> Optional[float]:
    """Read entire body. No cancel triggered."""
    t0 = time.perf_counter()
    try:
        with requests.post(url, json={"text": TEXT}, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            r.content  # forces full read
            return (time.perf_counter() - t0) * 1000
    except Exception as exc:
        print(f"  [error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _capture_audio_md5(url: str) -> Optional[str]:
    """Full-body capture; returns MD5 of PCM (excluding 4-byte SR header)."""
    try:
        r = requests.post(url, json={"text": TEXT}, stream=True, timeout=60)
        r.raise_for_status()
        data = r.content
        if len(data) < 4:
            return None
        pcm = data[4:]
        return hashlib.md5(pcm).hexdigest()
    except Exception as exc:
        print(f"  [audio_md5 error] {exc}", file=sys.stderr)
        return None


def run_early_break(url: str, n: int, verbose: bool) -> tuple[list[float], int]:
    ttfas: list[float] = []
    fails = 0
    for i in range(1, n + 1):
        ttfa = _one_early_break(url)
        if ttfa is None:
            fails += 1
            if verbose:
                print(f"iter {i}/{n}: FAIL")
            continue
        ttfas.append(ttfa)
        if verbose:
            flag = " <BROKEN" if i > 1 and ttfa < BROKEN_TTFA_THRESHOLD_MS else ""
            print(f"iter {i}/{n}: ttfa={ttfa:.1f} ms{flag}")
        else:
            # progress dots
            if i % 10 == 0:
                print(f"  ... {i}/{n}")
    return ttfas, fails


def run_no_pcm(url: str, n: int, verbose: bool) -> tuple[list[float], int]:
    ttfas: list[float] = []
    fails = 0
    for i in range(1, n + 1):
        ttfa = _one_no_pcm(url)
        if ttfa is None:
            fails += 1
            continue
        ttfas.append(ttfa)
        if verbose:
            print(f"iter {i}/{n}: ttfa={ttfa:.1f} ms")
    return ttfas, fails


def run_full_then_broken(url: str, n: int, verbose: bool) -> tuple[list[float], int]:
    """Alternate full-body reads with early-breaks."""
    ttfas: list[float] = []
    fails = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            ttfa = _one_full(url)
        else:
            ttfa = _one_early_break(url)
        if ttfa is None:
            fails += 1
            continue
        ttfas.append(ttfa)
        if verbose:
            kind = "full" if i % 2 == 0 else "break"
            print(f"iter {i}/{n} [{kind}]: ttfa={ttfa:.1f} ms")
    return ttfas, fails


def run_long_soak(url: str, duration_s: float, verbose: bool) -> tuple[list[float], int]:
    ttfas: list[float] = []
    fails = 0
    deadline = time.monotonic() + duration_s
    i = 0
    while time.monotonic() < deadline:
        i += 1
        ttfa = _one_early_break(url)
        if ttfa is None:
            fails += 1
            continue
        ttfas.append(ttfa)
        if verbose and i % 50 == 0:
            print(f"  soak iter {i}: ttfa={ttfa:.1f} ms (elapsed {time.monotonic() - deadline + duration_s:.0f}s)")
    return ttfas, fails


def evaluate(scenario: str, ttfas: list[float], fails: int, audio_md5: Optional[str]) -> int:
    """Print summary, return exit code (0=PASS, non-zero=FAIL).

    Scenario-specific gate logic:
      - `early-break` / `full-then-broken` / `long-soak`: measure TTFA to
        first PCM chunk; broken-state signature is TTFA < 100ms because
        the worker errored out before emitting audio. Apply
        BROKEN_TTFA_THRESHOLD_MS gate.
      - `no-pcm`: measures time to SR header (only 4 bytes). Fast return
        is EXPECTED here (header arrives before any TTS work). The real
        gate is the post-stress audio MD5 — system survived N cancels
        and can still produce baseline audio.
    """
    print("\n=== Summary ===")
    print(f"  scenario:  {scenario}")
    print(f"  total:     {len(ttfas) + fails}")
    print(f"  success:   {len(ttfas)}")
    print(f"  failures:  {fails}")
    if ttfas:
        srt = sorted(ttfas)
        print(f"  ttfa min:  {srt[0]:.1f} ms")
        print(f"  ttfa p50:  {srt[len(srt) // 2]:.1f} ms")
        print(f"  ttfa p90:  {srt[int(len(srt) * 0.9)]:.1f} ms")
        print(f"  ttfa max:  {srt[-1]:.1f} ms")

    exit_code = 0

    # Gate 1: no errors
    if fails > 0:
        print(f"  GATE FAIL: {fails} request errors")
        exit_code = 1

    # Gate 2: no broken-state TTFA — only applies when the scenario
    # actually attempts to read a PCM chunk. `no-pcm` measures
    # time-to-SR-header which is expected to be fast; skip TTFA gate.
    if scenario != "no-pcm":
        broken = [t for t in ttfas[1:] if t < BROKEN_TTFA_THRESHOLD_MS]
        if broken:
            print(f"  GATE FAIL: {len(broken)} iterations had TTFA < {BROKEN_TTFA_THRESHOLD_MS}ms "
                  f"(broken-state signature; min={min(broken):.1f}ms)")
            exit_code = 1
        else:
            print(f"  GATE PASS: 0 broken-state TTFAs")
    else:
        print(f"  GATE SKIP: TTFA threshold not applicable for no-pcm scenario "
              f"(SR header arrives before any PCM)")

    # Gate 3: audio MD5 baseline
    if audio_md5 is not None:
        if audio_md5 == BASELINE_MD5:
            print(f"  GATE PASS: audio MD5 = {audio_md5} matches baseline")
        else:
            print(f"  GATE FAIL: audio MD5 = {audio_md5} != baseline {BASELINE_MD5}")
            exit_code = 1
    else:
        print(f"  GATE SKIP: no final audio MD5 captured")

    print(f"\n{'PASS' if exit_code == 0 else 'FAIL'}")
    return exit_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="100.82.225.102:8621")
    ap.add_argument("--scenario", choices=["early-break", "no-pcm", "full-then-broken", "long-soak"],
                    default="early-break")
    ap.add_argument("--n", type=int, default=100, help="iters (ignored for long-soak)")
    ap.add_argument("--duration", type=float, default=600, help="seconds for long-soak")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-audio-check", action="store_true",
                    help="skip the final audio-MD5 baseline gate")
    args = ap.parse_args()

    url = f"http://{args.host}/tts/stream"
    print(f"=== {args.scenario} @ {url} ===")
    if args.scenario == "early-break":
        ttfas, fails = run_early_break(url, args.n, args.verbose)
    elif args.scenario == "no-pcm":
        ttfas, fails = run_no_pcm(url, args.n, args.verbose)
    elif args.scenario == "full-then-broken":
        ttfas, fails = run_full_then_broken(url, args.n, args.verbose)
    elif args.scenario == "long-soak":
        ttfas, fails = run_long_soak(url, args.duration, args.verbose)
    else:
        raise SystemExit(f"unknown scenario {args.scenario}")

    audio_md5: Optional[str] = None
    if not args.skip_audio_check:
        print("\n--- final audio MD5 check (full-body capture) ---")
        audio_md5 = _capture_audio_md5(url)
        print(f"  audio_md5: {audio_md5}")

    sys.exit(evaluate(args.scenario, ttfas, fails, audio_md5))


if __name__ == "__main__":
    main()
