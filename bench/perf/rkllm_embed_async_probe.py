#!/usr/bin/env python3
"""Probe RKLLM v1.2.3 embed cache reuse and async run behavior on RK3588.

Run inside the RK service image with /opt/asr/models mounted.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


for p in (
    "/opt/speech",
    "/opt/speech/third_party/rkvoice-stream",
    "/workspace/third_party/rkvoice-stream",
):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

import rkvoice_stream.backends.asr.qwen3.decoder as decoder_mod
from rkvoice_stream.backends.asr.qwen3.engine import Qwen3ASREngine
from rkvoice_stream.backends.asr.qwen3.decoder import (
    RKLLM_INPUT_EMBED,
    RKLLM_INFER_GENERATE,
    RKLLMEmbedInput,
    RKLLMInferParam,
    RKLLMInput,
    RKLLM_Handle_t,
)


def set_optional_signatures(decoder) -> dict:
    caps = {}
    try:
        decoder.lib.rkllm_run_async.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        decoder.lib.rkllm_run_async.restype = ctypes.c_int
        caps["rkllm_run_async"] = True
    except AttributeError:
        caps["rkllm_run_async"] = False
    try:
        decoder.lib.rkllm_is_running.argtypes = [RKLLM_Handle_t]
        decoder.lib.rkllm_is_running.restype = ctypes.c_int
        caps["rkllm_is_running"] = True
    except AttributeError:
        caps["rkllm_is_running"] = False
    try:
        decoder.lib.rkllm_get_kv_cache_size.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(ctypes.c_int),
        ]
        decoder.lib.rkllm_get_kv_cache_size.restype = ctypes.c_int
        caps["rkllm_get_kv_cache_size"] = True
    except AttributeError:
        caps["rkllm_get_kv_cache_size"] = False
    return caps


def reset_decoder_state(decoder, early_stop_tokens: int = 1) -> None:
    decoder._output_chunks = []
    decoder._perf = {}
    decoder._repeat_buf = []
    decoder._aborted = False
    decoder._abort_reason = ""
    decoder._early_stop_tokens = early_stop_tokens


def clean_text(text: str) -> str:
    for tag in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(tag, "")
    return text


def kv_size(decoder) -> int | None:
    if not hasattr(decoder.lib, "rkllm_get_kv_cache_size"):
        return None
    try:
        cache_sizes = (ctypes.c_int * 1)()
        ret = int(decoder.lib.rkllm_get_kv_cache_size(decoder.handle, cache_sizes))
        if ret != 0:
            return None
        return int(cache_sizes[0])
    except Exception:
        return None


def clear_kv(decoder) -> int:
    return int(decoder.lib.rkllm_clear_kv_cache(decoder.handle, 0, None, None))


def make_inputs(embed_array: np.ndarray, n_tokens: int, keep_history: int):
    embed_array = np.ascontiguousarray(embed_array, dtype=np.float32)
    embed_ptr = embed_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rkllm_input = RKLLMInput()
    rkllm_input.role = b""
    rkllm_input.enable_thinking = ctypes.c_bool(False)
    rkllm_input.input_type = RKLLM_INPUT_EMBED
    rkllm_input.input_data.embed_input = RKLLMEmbedInput(embed_ptr, n_tokens)

    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLM_INFER_GENERATE
    infer_param.lora_params = None
    infer_param.prompt_cache_params = None
    infer_param.keep_history = keep_history
    return embed_array, rkllm_input, infer_param


def run_embed_probe(
    decoder,
    embed_array: np.ndarray,
    n_tokens: int,
    *,
    label: str,
    clear_before: bool,
    keep_history: int,
    async_mode: bool = False,
    timeout_s: float = 20.0,
) -> dict:
    with decoder._lock:
        if clear_before:
            clear_kv(decoder)
        reset_decoder_state(decoder)
        retained, rkllm_input, infer_param = make_inputs(embed_array, n_tokens, keep_history)
        before_kv = kv_size(decoder)
        t0 = time.perf_counter()
        if async_mode:
            submit_t0 = time.perf_counter()
            ret = int(
                decoder.lib.rkllm_run_async(
                    decoder.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None
                )
            )
            submit_ms = (time.perf_counter() - submit_t0) * 1000.0
            polls = 0
            is_running_samples = []
            while ret == 0:
                polls += 1
                state = int(decoder.lib.rkllm_is_running(decoder.handle))
                if len(is_running_samples) < 40:
                    is_running_samples.append(state)
                if decoder._perf:
                    break
                if time.perf_counter() - t0 > timeout_s:
                    decoder.abort()
                    break
                time.sleep(0.01)
            total_ms = (time.perf_counter() - t0) * 1000.0
        else:
            submit_ms = None
            is_running_samples = None
            ret = int(
                decoder.lib.rkllm_run(
                    decoder.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None
                )
            )
            polls = None
            total_ms = (time.perf_counter() - t0) * 1000.0
        after_kv = kv_size(decoder)
        text = clean_text("".join(decoder._output_chunks))
        return {
            "label": label,
            "ret_code": ret,
            "clear_before": clear_before,
            "keep_history": keep_history,
            "async_mode": async_mode,
            "submit_ms": submit_ms,
            "wall_ms": total_ms,
            "polls": polls,
            "is_running_samples": is_running_samples,
            "n_input_tokens": int(n_tokens),
            "n_tokens_generated": len(decoder._output_chunks),
            "aborted": bool(decoder._aborted),
            "abort_reason": decoder._abort_reason,
            "kv_before": before_kv,
            "kv_after": after_kv,
            "perf": dict(decoder._perf),
            "text_prefix": text[:80],
        }


def run_wrapper_probe(decoder, embed_array: np.ndarray, n_tokens: int,
                      label: str) -> dict:
    t0 = time.perf_counter()
    result = decoder.run_embed(embed_array, n_tokens, keep_history=0)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "label": label,
        "wall_ms": wall_ms,
        "n_input_tokens": int(n_tokens),
        "perf": dict(result.get("perf") or {}),
        "ret_code": result.get("ret_code"),
        "n_tokens_generated": result.get("n_tokens_generated"),
        "aborted": result.get("aborted"),
        "abort_reason": result.get("abort_reason"),
        "text_prefix": (result.get("text") or "")[:80],
    }


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[len(xs) // 2]


def main() -> int:
    patched_async_init = os.environ.get("ASR_PATCH_DECODER_IS_ASYNC", "0") == "1"
    if patched_async_init:
        import inspect

        src = inspect.getsource(decoder_mod.RKLLMDecoder)
        marker = "param.is_async = False"
        if marker not in src:
            raise RuntimeError("RKLLMDecoder source marker not found for async patch")
        src = src.replace(marker, "param.is_async = True")
        exec(src, decoder_mod.__dict__)

    model_dir = os.environ.get("ASR_MODEL_DIR", "/opt/asr/models")
    lib_path = os.environ.get("RKLLM_LIB_PATH", "/opt/asr/lib/librkllmrt.so")
    engine = Qwen3ASREngine(
        model_dir=model_dir,
        platform=os.environ.get("ASR_PLATFORM", "rk3588"),
        lib_path=lib_path,
        decoder_type="rkllm",
        decoder_quant=os.environ.get("ASR_DECODER_QUANT", "fp16"),
        encoder_quant=os.environ.get("ASR_ENCODER_QUANT", "fp16"),
        encoder_sizes=[2, 4],
        npu_core_mask=os.environ.get("ASR_NPU_CORE_MASK", "NPU_CORE_AUTO"),
        enabled_cpus=int(os.environ.get("ASR_ENABLED_CPUS", "4")),
        max_context_len=int(os.environ.get("ASR_MAX_CONTEXT_LEN", "512")),
        max_new_tokens=int(os.environ.get("ASR_MAX_NEW_TOKENS", "8")),
        embed_flash=int(os.environ.get("ASR_EMBED_FLASH", "1")),
        compact_suffix=os.environ.get("ASR_COMPACT_SUFFIX", "1") == "1",
        final_stop_on_punctuation=False,
        decoder_embed_cache_reuse=os.environ.get(
            "ASR_DECODER_EMBED_CACHE_REUSE", "0").lower() in (
                "1", "true", "yes", "on"
            ),
        decoder_async_mode=os.environ.get("ASR_DECODER_ASYNC", "0").lower() in (
            "1", "true", "yes", "on"
        ),
        decoder_async_timeout_s=float(
            os.environ.get("ASR_DECODER_ASYNC_TIMEOUT_S", "30")
        ),
        verbose=True,
    )
    decoder = engine.decoder
    caps = set_optional_signatures(decoder)

    # Deterministic low-amplitude pseudo-audio. Text accuracy is irrelevant here;
    # the probe measures decoder prefill/cache/runtime behavior.
    sr = 16000
    t = np.arange(sr * 4, dtype=np.float32) / sr
    audio = (0.02 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    audio_emb = engine.encoder.encode(audio)
    if isinstance(audio_emb, tuple):
        audio_emb = audio_emb[0]
    audio_emb = np.asarray(audio_emb, dtype=np.float32)

    n_audio_1 = max(8, min(audio_emb.shape[0], 28))
    n_audio_2 = max(n_audio_1, min(audio_emb.shape[0], n_audio_1 + 16))
    embed1, n1 = engine.build_embed(audio_emb[:n_audio_1], language="Chinese")
    embed2, n2 = engine.build_embed(audio_emb[:n_audio_2], language="Chinese")

    results: list[dict] = []

    for i in range(3):
        results.append(
            run_embed_probe(
                decoder, embed2, n2,
                label=f"clear_each_{i}",
                clear_before=True,
                keep_history=0,
            )
        )

    clear_kv(decoder)
    results.append(
        run_embed_probe(
            decoder, embed1, n1,
            label="reuse_same_first",
            clear_before=False,
            keep_history=0,
        )
    )
    results.append(
        run_embed_probe(
            decoder, embed1, n1,
            label="reuse_same_second",
            clear_before=False,
            keep_history=0,
        )
    )

    clear_kv(decoder)
    results.append(
        run_embed_probe(
            decoder, embed1, n1,
            label="reuse_extend_first_short",
            clear_before=False,
            keep_history=0,
        )
    )
    results.append(
        run_embed_probe(
            decoder, embed2, n2,
            label="reuse_extend_second_long",
            clear_before=False,
            keep_history=0,
        )
    )

    async_result = None
    if caps.get("rkllm_run_async") and caps.get("rkllm_is_running"):
        async_result = run_embed_probe(
            decoder, embed2, n2,
            label="async_submit",
            clear_before=True,
            keep_history=0,
            async_mode=True,
            timeout_s=float(os.environ.get("ASR_ASYNC_TIMEOUT_S", "5.0")),
        )

    wrapper_results: list[dict] = []
    if hasattr(decoder, "clear_embed_cache"):
        decoder.clear_embed_cache()
    else:
        clear_kv(decoder)
    wrapper_results.append(run_wrapper_probe(
        decoder, embed1, n1, "wrapper_same_first"))
    wrapper_results.append(run_wrapper_probe(
        decoder, embed1, n1, "wrapper_same_second"))
    if hasattr(decoder, "clear_embed_cache"):
        decoder.clear_embed_cache()
    else:
        clear_kv(decoder)
    wrapper_results.append(run_wrapper_probe(
        decoder, embed1, n1, "wrapper_extend_first_short"))
    wrapper_results.append(run_wrapper_probe(
        decoder, embed2, n2, "wrapper_extend_second_long"))

    clear_prefills = [
        r.get("perf", {}).get("prefill_time_ms", 0.0)
        for r in results
        if r["label"].startswith("clear_each_")
    ]
    clear_walls = [r["wall_ms"] for r in results if r["label"].startswith("clear_each_")]
    summary = {
        "capabilities": caps,
        "patched_decoder_is_async": patched_async_init,
        "engine_embed_cache_reuse": bool(getattr(decoder, "_embed_cache_reuse", False)),
        "engine_async_mode": bool(getattr(decoder, "_async_mode", False)),
        "model_dir": model_dir,
        "lib_path": lib_path,
        "audio_emb_shape": list(audio_emb.shape),
        "embed1_tokens": int(n1),
        "embed2_tokens": int(n2),
        "clear_each_median_prefill_ms": median(clear_prefills),
        "clear_each_median_wall_ms": median(clear_walls),
        "results": results,
        "async_result": async_result,
        "wrapper_results": wrapper_results,
    }
    print("===RKLLM_PROBE_JSON_BEGIN===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===RKLLM_PROBE_JSON_END===")
    try:
        engine.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
