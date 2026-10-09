"""Long-running stability test.

Drives ASR / TTS continuously for `--duration` minutes; records per-iteration
RTF and container RSS; reports drift (first 5 min mean vs last 5 min mean).
Surface memory leaks, thermal throttling, slow GC pile-up.
"""
from __future__ import annotations
import statistics as st
import time
from pathlib import Path

from client import ASRClient, TTSClient
from runners import load_corpus, load_prompts
from memory import MemorySampler


def run_stability(asr: ASRClient, tts: TTSClient,
                  corpus: list[dict], prompts: list[dict],
                  duration_s: float, mode: str = "v2v",
                  memory_sampler: MemorySampler | None = None) -> dict:
    """mode: 'asr' | 'tts' | 'v2v' (alternating)."""
    t_start = time.monotonic()
    deadline = t_start + duration_s
    rows: list[dict] = []
    i = 0
    while time.monotonic() < deadline:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        try:
            if mode in ("asr", "v2v"):
                entry = corpus[i % len(corpus)]
                lang = "Chinese" if entry["lang"] == "zh" else "English"
                r = asr.transcribe_streaming(entry["bytes"], lang)
                rows.append({"i": i, "elapsed_min": elapsed_min, "kind": "asr",
                             "id": entry["id"], "rtf": r.rtf,
                             "eos_to_final_ms": r.eos_to_final_ms,
                             "audio_dur_s": r.audio_dur_s})
            if mode in ("tts", "v2v"):
                p = prompts[i % len(prompts)]
                rt = tts.synthesize(p["text"], p["lang"])
                rows.append({"i": i, "elapsed_min": elapsed_min, "kind": "tts",
                             "id": p["id"], "rtf": rt.rtf,
                             "tfd_ms": rt.tfd_ms, "total_ms": rt.total_ms,
                             "audio_dur_s": rt.audio_dur_s})
        except Exception as e:
            rows.append({"i": i, "elapsed_min": elapsed_min, "error": str(e)})
        if i % 10 == 0:
            n_ok = sum(1 for r in rows if "error" not in r)
            n_err = len(rows) - n_ok
            mem_str = ""
            if memory_sampler and memory_sampler.peak_mib > 0:
                mem_str = f"  RSS now={memory_sampler.samples[-1]:.0f}MiB peak={memory_sampler.peak_mib:.0f}MiB"
            print(f"  t+{elapsed_min:5.1f}min  i={i:4d}  ok={n_ok} err={n_err}{mem_str}")
        i += 1

    # Drift analysis: first 5 min vs last 5 min RTF medians
    early = [r["rtf"] for r in rows if "rtf" in r and r["elapsed_min"] <= 5.0]
    late_cut = max(0, (duration_s / 60.0) - 5.0)
    late  = [r["rtf"] for r in rows if "rtf" in r and r["elapsed_min"] >= late_cut]

    drift = {
        "early_n":  len(early),
        "late_n":   len(late),
        "early_rtf_p50": st.median(early) if early else None,
        "late_rtf_p50":  st.median(late)  if late  else None,
        "rtf_drift_pct": None,
    }
    if early and late and drift["early_rtf_p50"]:
        drift["rtf_drift_pct"] = round(
            (drift["late_rtf_p50"] - drift["early_rtf_p50"]) / drift["early_rtf_p50"] * 100, 2)

    mem_drift = None
    if memory_sampler and len(memory_sampler.samples) >= 20:
        head = memory_sampler.samples[:10]
        tail = memory_sampler.samples[-10:]
        mem_drift = {
            "head_mib_mean": round(sum(head) / len(head), 1),
            "tail_mib_mean": round(sum(tail) / len(tail), 1),
            "rss_growth_mib": round(sum(tail) / len(tail) - sum(head) / len(head), 1),
            "peak_mib": memory_sampler.peak_mib,
        }

    return {
        "duration_min": (time.monotonic() - t_start) / 60.0,
        "total_iters": i,
        "rows": rows,
        "drift": drift,
        "memory_drift": mem_drift,
    }
