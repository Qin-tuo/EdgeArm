#!/usr/bin/env python3
"""Direct smoke bench for Paraformer RKNN.

This bypasses the HTTP server and can exercise either:
- stream: create_stream() -> accept_waveform(chunks) -> finalize()
- offline: transcribe(wav_bytes)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _norm_zh(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'（）()：:；;,.!?！？-]", "", text).lower()


def _norm_en(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text).lower()
    return [w for w in text.split() if w]


def _norm_en_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


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
    r, h = _norm_en(ref), _norm_en(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def _char_err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = _norm_zh(ref), _norm_zh(hyp)
    else:
        r, h = _norm_en_chars(ref), _norm_en_chars(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/work/bench/perf/corpus")
    ap.add_argument("--category", default="short")
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--chunk-ms", type=int, default=250)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument(
        "--call-prepare",
        action="store_true",
        help="Call stream.prepare_finalize() before finalize() and time it separately.",
    )
    ap.add_argument(
        "--mode",
        choices=["stream", "offline", "stream_cif_final_decode"],
        default="stream",
    )
    args = ap.parse_args()

    from rkvoice_stream.backends.asr.paraformer_rknn import (
        CACHE_COUNT,
        CACHE_SHAPE,
        CIF_TAIL_THRESHOLD,
        RIGHT_LOOKAHEAD_LFR,
        ParaformerRKNNBackend,
        cif,
        compute_fbank,
        decode_ids,
        stack_frames,
    )

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    items = [
        x for x in manifest["files"]
        if x["category"] == args.category and x["lang"] == args.lang
    ][: args.limit]

    backend = ParaformerRKNNBackend()
    t0 = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - t0) * 1000

    rows = []
    chunk_samples = None
    for item in items:
        wav_path = corpus / item["filename"]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if args.mode == "stream":
            chunk_samples = max(1, int(sr * args.chunk_ms / 1000))
            stream = backend.create_stream(language="auto")
            feed_ms = 0.0
            prepare_ms = 0.0
            for start in range(0, len(audio), chunk_samples):
                t1 = time.perf_counter()
                stream.accept_waveform(sr, np.asarray(audio[start:start + chunk_samples], dtype=np.float32))
                feed_ms += (time.perf_counter() - t1) * 1000
            if args.call_prepare:
                t_prepare = time.perf_counter()
                stream.prepare_finalize()
                prepare_ms = (time.perf_counter() - t_prepare) * 1000
            t2 = time.perf_counter()
            hyp = stream.finalize()
            finalize_ms = (time.perf_counter() - t2) * 1000
            total_compute_ms = feed_ms + prepare_ms + finalize_ms
        elif args.mode == "stream_cif_final_decode":
            chunk_samples = max(1, int(sr * args.chunk_ms / 1000))
            all_audio = np.zeros(0, dtype=np.float32)
            prev_total_lfr = 0
            cif_processed_lfr = 0
            carry_weight = 0.0
            carry_embed = np.zeros(512, dtype=np.float32)
            all_embeds: list[np.ndarray] = []
            feed_ms = 0.0
            max_bucket = max(backend._encoders)
            for start_sample in range(0, len(audio), chunk_samples):
                all_audio = np.concatenate([
                    all_audio,
                    np.asarray(audio[start_sample:start_sample + chunk_samples], dtype=np.float32),
                ])
                t1 = time.perf_counter()
                feats = stack_frames(compute_fbank(all_audio))
                cur_total_lfr = feats.shape[0]
                if cur_total_lfr <= prev_total_lfr:
                    feed_ms += (time.perf_counter() - t1) * 1000
                    continue
                prev_total_lfr = cur_total_lfr
                enc_input = feats if cur_total_lfr <= max_bucket else feats[-max_bucket:]
                enc, alphas = backend._run_encoder(enc_input)
                if enc is None or alphas is None:
                    feed_ms += (time.perf_counter() - t1) * 1000
                    continue
                window_start_abs = cur_total_lfr - enc.shape[1]
                cif_end_abs = max(window_start_abs, cur_total_lfr - RIGHT_LOOKAHEAD_LFR)
                cif_start_abs = max(cif_processed_lfr, window_start_abs)
                if cif_end_abs > cif_start_abs:
                    seg_start = cif_start_abs - window_start_abs
                    seg_end = cif_end_abs - window_start_abs
                    embeds, carry_weight, carry_embed = cif(
                        enc[0][seg_start:seg_end],
                        alphas[0][seg_start:seg_end],
                        carry_weight=carry_weight,
                        carry_embed=carry_embed,
                    )
                    if len(embeds) > 0:
                        all_embeds.append(embeds)
                    cif_processed_lfr = cif_end_abs
                feed_ms += (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            feats = stack_frames(compute_fbank(all_audio))
            cur_total_lfr = feats.shape[0]
            enc_input = feats if cur_total_lfr <= max_bucket else feats[-max_bucket:]
            enc, alphas = backend._run_encoder(enc_input)
            if enc is not None and alphas is not None:
                window_start_abs = cur_total_lfr - enc.shape[1]
                cif_start_abs = max(cif_processed_lfr, window_start_abs)
                if cur_total_lfr > cif_start_abs:
                    seg_start = cif_start_abs - window_start_abs
                    seg_end = cur_total_lfr - window_start_abs
                    embeds, carry_weight, carry_embed = cif(
                        enc[0][seg_start:seg_end],
                        alphas[0][seg_start:seg_end],
                        carry_weight=carry_weight,
                        carry_embed=carry_embed,
                    )
                    if len(embeds) > 0:
                        all_embeds.append(embeds)
                if carry_weight >= CIF_TAIL_THRESHOLD:
                    all_embeds.append((carry_embed / carry_weight)[np.newaxis, :])
            acoustic = (
                np.concatenate(all_embeds, axis=0)
                if all_embeds
                else np.empty((0, 512), dtype=np.float32)
            )
            cache = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
            token_ids: list[int] = []
            if enc is not None and len(acoustic) > 0:
                sample_ids = backend._run_decoder(enc, enc.shape[1], acoustic, len(acoustic), cache)
                if sample_ids is not None:
                    token_ids.extend(sample_ids.tolist())
            hyp = decode_ids(token_ids, backend._tokens)
            finalize_ms = (time.perf_counter() - t2) * 1000
            total_compute_ms = feed_ms + finalize_ms
            prepare_ms = 0.0
        else:
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            t2 = time.perf_counter()
            hyp = backend.transcribe(buf.getvalue(), language="auto").text
            total_compute_ms = (time.perf_counter() - t2) * 1000
            feed_ms = 0.0
            finalize_ms = total_compute_ms
            prepare_ms = 0.0
        ref = item.get("eval_transcript") or item["transcript"]
        err = _err_rate(ref, hyp, args.lang)
        char_err = _char_err_rate(ref, hyp, args.lang)
        rows.append({
            "id": item["id"],
            "mode": args.mode,
            "duration_s": item["duration_s"],
            "text": hyp,
            "ref": ref,
            "error_rate": err,
            "char_error_rate": char_err,
            "feed_ms": feed_ms,
            "prepare_ms": prepare_ms,
            "finalize_ms": finalize_ms,
            "total_compute_ms": total_compute_ms,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / max(1, len(rows))

    print(json.dumps({
        "summary": {
            "lang": args.lang,
            "mode": args.mode,
            "category": args.category,
            "n": len(rows),
            "chunk_ms": args.chunk_ms,
            "preload_ms": preload_ms,
            "mean_error_rate": mean("error_rate"),
            "mean_char_error_rate": mean("char_error_rate"),
            "mean_feed_ms": mean("feed_ms"),
            "mean_prepare_ms": mean("prepare_ms"),
            "mean_finalize_ms": mean("finalize_ms"),
            "mean_total_compute_ms": mean("total_compute_ms"),
        }
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
