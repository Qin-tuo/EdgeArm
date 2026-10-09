#!/usr/bin/env python3
"""In-process Qwen3 RK ASR streaming eos-to-final benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


for p in (
    "/opt/speech",
    "/opt/speech/third_party/rkvoice-stream",
    "/workspace/third_party/rkvoice-stream",
):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _norm_zh(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'（）()：:；;,.!?！？-]", "", text).lower()


def _norm_en(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text).lower()
    return [w for w in text.split() if w]


def _norm_en_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def _norm_chars(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'（）()：:；;,.!?！？\-¿¡`´’‘]", "", text).lower()


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = _norm_zh(ref), _norm_zh(hyp)
        return _edit_distance(r, h) / max(1, len(r))
    if lang in {"ja", "ko"}:
        r, h = _norm_chars(ref), _norm_chars(hyp)
        return _edit_distance(r, h) / max(1, len(r))
    r, h = _norm_en(ref), _norm_en(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def _char_err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = _norm_zh(ref), _norm_zh(hyp)
    elif lang in {"ja", "ko"}:
        r, h = _norm_chars(ref), _norm_chars(hyp)
    else:
        r, h = _norm_en_chars(ref), _norm_en_chars(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def _load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / max(1, len(vals))


def _median(rows: list[dict], key: str) -> float:
    vals = sorted(float(r[key]) for r in rows if r.get(key) is not None)
    return vals[len(vals) // 2] if vals else 0.0


LANGUAGE_HINTS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "es": "Spanish",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
}


def _language_hint(lang: str) -> str:
    return LANGUAGE_HINTS.get(lang, "English")


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_error_rate": _mean(rows, "error_rate"),
        "mean_char_error_rate": _mean(rows, "char_error_rate"),
        "mean_eos_to_final_ms": _mean(rows, "eos_to_final_ms"),
        "median_eos_to_final_ms": _median(rows, "eos_to_final_ms"),
        "mean_finalize_ms": _mean(rows, "finalize_ms"),
        "median_finalize_ms": _median(rows, "finalize_ms"),
        "mean_feed_wall_ms": _mean(rows, "feed_wall_ms"),
    }


def build_engine(args):
    from rkvoice_stream.backends.asr.qwen3 import Qwen3ASREngine

    return Qwen3ASREngine(
        model_dir=args.model_dir,
        platform=args.platform,
        lib_path=args.lib_path,
        decoder_type="rkllm",
        decoder_quant=args.decoder_quant,
        encoder_sizes=[2, 4],
        npu_core_mask=args.npu_core_mask,
        enabled_cpus=args.enabled_cpus,
        max_context_len=args.max_context_len,
        max_new_tokens=args.max_new_tokens,
        embed_flash=1,
        compact_suffix=True,
        final_stop_on_punctuation=_env_bool("ASR_FINAL_STOP_ON_PUNCTUATION", True),
        final_stop_min_chars=int(os.environ.get("ASR_FINAL_STOP_MIN_CHARS", "8")),
        final_stop_min_chunks=int(os.environ.get("ASR_FINAL_STOP_MIN_CHUNKS", "2")),
        decoder_embed_cache_reuse=_env_bool("ASR_DECODER_EMBED_CACHE_REUSE", False),
        decoder_async_mode=_env_bool("ASR_DECODER_ASYNC", False),
        decoder_async_timeout_s=float(os.environ.get("ASR_DECODER_ASYNC_TIMEOUT_S", "30")),
        verbose=True,
    )


def run_one(engine, wav_path: Path, ref: str, lang: str,
            chunk_ms: int, realtime: bool) -> dict:
    from rkvoice_stream.backends.asr.qwen3.streaming import Qwen3TrueStreamingASRStream

    audio = _load_audio(wav_path)
    chunk_n = max(1, int(16000 * chunk_ms / 1000))
    language = _language_hint(lang)
    stream = Qwen3TrueStreamingASRStream(engine, language=language)

    feed_t0 = time.perf_counter()
    for start in range(0, len(audio), chunk_n):
        t0 = time.perf_counter()
        stream.feed_audio(audio[start:start + chunk_n])
        if realtime:
            time.sleep(max(0.0, chunk_ms / 1000 - (time.perf_counter() - t0)))
    feed_wall_ms = (time.perf_counter() - feed_t0) * 1000

    eos_t0 = time.perf_counter()
    result = stream.finish()
    eos_to_final_ms = (time.perf_counter() - eos_t0) * 1000
    text = result.get("text", "")
    return {
        "text": text,
        "ref": ref,
        "error_rate": _err_rate(ref, text, lang),
        "char_error_rate": _char_err_rate(ref, text, lang),
        "feed_wall_ms": feed_wall_ms,
        "eos_to_final_ms": eos_to_final_ms,
        "finalize_ms": result.get("finalize_ms"),
        "stats": result.get("stats", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/tmp/asr_corpus")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--category", default="short")
    parser.add_argument(
        "--lang",
        default="zh",
        help="Language code to run, or 'all' for all languages in the manifest.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Per-language limit; <=0 means no limit.")
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--model-dir", default=os.environ.get("ASR_MODEL_DIR", "/opt/asr/models"))
    parser.add_argument("--lib-path", default=os.environ.get("RKLLM_LIB_PATH", "/opt/asr/lib/librkllmrt.so"))
    parser.add_argument("--platform", default=os.environ.get("ASR_PLATFORM", "rk3576"))
    parser.add_argument("--decoder-quant", default=os.environ.get("ASR_DECODER_QUANT", "w8a8"))
    parser.add_argument("--npu-core-mask", default=os.environ.get("ASR_NPU_CORE_MASK", "NPU_CORE_AUTO"))
    parser.add_argument("--enabled-cpus", type=int, default=int(os.environ.get("ASR_ENABLED_CPUS", "4")))
    parser.add_argument("--max-context-len", type=int, default=int(os.environ.get("ASR_MAX_CONTEXT_LEN", "512")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("ASR_MAX_NEW_TOKENS", "64")))
    parser.add_argument("--perf-mode", default=os.environ.get("RK_PERF_MODE", "unknown"))
    args = parser.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / args.manifest).read_text(encoding="utf-8"))
    lang_filter = None if args.lang == "all" else args.lang
    items = [
        x for x in manifest["files"]
        if x.get("category") == args.category
        and (lang_filter is None or x.get("lang") == lang_filter)
    ]
    if args.limit > 0:
        limited = []
        for _, group in itertools.groupby(sorted(items, key=lambda x: x.get("lang", "")), key=lambda x: x.get("lang", "")):
            limited.extend(list(group)[: args.limit])
        items = limited

    engine = build_engine(args)
    rows = []
    try:
        for item in items:
            ref = item.get("eval_transcript") or item["transcript"]
            lang = item["lang"]
            row = run_one(
                engine,
                corpus / item["filename"],
                ref,
                lang,
                args.chunk_ms,
                args.realtime,
            )
            row.update({
                "id": item["id"],
                "lang": item["lang"],
                "category": item["category"],
                "duration_s": item["duration_s"],
                "embed_cache_reuse": bool(getattr(engine.decoder, "_embed_cache_reuse", False)),
                "async_mode": bool(getattr(engine.decoder, "_async_mode", False)),
            })
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        try:
            engine.close()
        except Exception:
            pass

    print(json.dumps({
        "summary": {
            "platform": args.platform,
            "decoder_quant": args.decoder_quant,
            "lang": args.lang,
            "category": args.category,
            "manifest": args.manifest,
            "chunk_ms": args.chunk_ms,
            "realtime": args.realtime,
            "perf_mode": args.perf_mode,
            "embed_cache_reuse": _env_bool("ASR_DECODER_EMBED_CACHE_REUSE", False),
            "async_mode": _env_bool("ASR_DECODER_ASYNC", False),
            **_summary(rows),
            "by_lang": {
                lang: _summary([r for r in rows if r.get("lang") == lang])
                for lang in sorted({r.get("lang") for r in rows})
            },
        }
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
