#!/usr/bin/env bash
# Run RK Qwen3-ASR with a long-dictation true-streaming profile.
#
# This profile lets VAD split long audio into multiple finalized segments, then
# accumulates them into one transcript.  It is for transcription quality checks,
# not low-latency V2V turn-taking.

set -euo pipefail

PLATFORM="${1:-rk3588}"
DECODER_QUANT="${2:-w8a8}"
CATEGORY="${3:-long}"
LANG="${4:-en}"
LIMIT="${LIMIT:-2}"
MANIFEST="${MANIFEST:-manifest.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
IMAGE="${IMAGE:-openvoicestream:rk-slim-fresh-20260603}"
CORPUS="${CORPUS:-/tmp/asr_corpus}"
PERF_DIR="${PERF_DIR:-/tmp/seeed-perf}"
PATCH_DIR="${PATCH_DIR:-/tmp/slv-warmup-patch}"

case "$PLATFORM" in
  rk3588|rk3576) ;;
  *) echo "unsupported platform: $PLATFORM" >&2; exit 2 ;;
esac

docker run --rm --privileged --network host \
  -v /dev:/dev \
  -v rk-asr-models:/opt/asr/models \
  -v "$CORPUS":"$CORPUS":ro \
  -v "$PERF_DIR":"$PERF_DIR":ro \
  -v "$PATCH_DIR/qwen3_rk.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3_rk.py:ro \
  -v "$PATCH_DIR/engine.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/engine.py:ro \
  -v "$PATCH_DIR/decoder.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/decoder.py:ro \
  -v "$PATCH_DIR/config.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/config.py:ro \
  -v "$PATCH_DIR/streaming.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/streaming.py:ro \
  -v "$PATCH_DIR/stream.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/stream.py:ro \
  -v "$PATCH_DIR/chunk_confirm.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/chunk_confirm.py:ro \
  -e PYTHONPATH=/opt/speech/third_party/rkvoice-stream:/opt/speech \
  -e ASR_MODEL_DIR=/opt/asr/models \
  -e RKLLM_LIB_PATH=/opt/asr/lib/librkllmrt.so \
  -e ASR_PLATFORM="$PLATFORM" \
  -e ASR_DECODER_QUANT="$DECODER_QUANT" \
  -e ASR_DECODER_EMBED_CACHE_REUSE=1 \
  -e ASR_DECODER_ASYNC=1 \
  -e ASR_FINAL_STOP_ON_PUNCTUATION=0 \
  -e ASR_NPU_CORE_MASK=NPU_CORE_AUTO \
  -e RK_PERF_MODE=highperf \
  -e QWEN3_ASR_STREAM_PARTIAL=0 \
  -e QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT=1 \
  -e QWEN3_ASR_ACCUMULATE_SEGMENTS=1 \
  -e QWEN3_ASR_TRUE_ROLL_SEC=30 \
  -e QWEN3_ASR_VAD_FINAL_ASYNC=0 \
  -e QWEN3_ASR_SEGMENT_TEXT_OVERLAP_TOKENS=8 \
  "$IMAGE" \
  /opt/venv/bin/python "$PERF_DIR/qwen3_asr_stream_eos_bench.py" \
    --corpus "$CORPUS" \
    --manifest "$MANIFEST" \
    --category "$CATEGORY" \
    --lang "$LANG" \
    --limit "$LIMIT" \
    --chunk-ms 250 \
    --realtime \
    --platform "$PLATFORM" \
    --decoder-quant "$DECODER_QUANT" \
    --perf-mode highperf \
    --max-context-len 512 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --enabled-cpus 4
