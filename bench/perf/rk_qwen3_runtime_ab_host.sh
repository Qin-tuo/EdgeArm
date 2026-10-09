#!/usr/bin/env bash
# Run RK Qwen3-ASR runtime knob A/B on-device via Docker.

set -euo pipefail

PLATFORM="${1:?usage: $0 <rk3576|rk3588> [decoder_quant]}"
DECODER_QUANT="${2:-w8a8}"
IMAGE="${IMAGE:-openvoicestream:rk-slim-fresh-20260603}"
CORPUS="${CORPUS:-/tmp/asr_corpus}"
PERF_DIR="${PERF_DIR:-/tmp/seeed-perf}"
PATCH_DIR="${PATCH_DIR:-/tmp/slv-warmup-patch}"

case "$PLATFORM" in
  rk3576|rk3588) ;;
  *) echo "unsupported platform: $PLATFORM" >&2; exit 2 ;;
esac

run_variant() {
  local label="$1"
  local ctx="$2"
  local cpus="$3"

  echo "VARIANT=$label ctx=$ctx cpus=$cpus"
  docker run --rm --privileged --network host \
    -v /dev:/dev \
    -v rk-asr-models:/opt/asr/models \
    -v "$CORPUS":"$CORPUS":ro \
    -v "$PERF_DIR":"$PERF_DIR":ro \
    -v "$PATCH_DIR/qwen3_rk.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3_rk.py:ro \
    -v "$PATCH_DIR/engine.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/engine.py:ro \
    -v "$PATCH_DIR/decoder.py":/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/decoder.py:ro \
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
    -e ASR_NPU_CORE_MASK=NPU_CORE_AUTO \
    -e RK_PERF_MODE=highperf \
    "$IMAGE" \
    /opt/venv/bin/python "$PERF_DIR/qwen3_asr_stream_eos_bench.py" \
      --corpus "$CORPUS" \
      --category short \
      --lang zh \
      --limit 5 \
      --chunk-ms 250 \
      --realtime \
      --platform "$PLATFORM" \
      --decoder-quant "$DECODER_QUANT" \
      --perf-mode highperf \
      --max-context-len "$ctx" \
      --enabled-cpus "$cpus" \
    | grep '"summary"'
}

run_variant baseline 512 4
run_variant ctx256 256 4
run_variant cpus2 512 2
