#!/usr/bin/env bash
# Run RK3588 Qwen3-ASR W8A8/FP16 quality-boundary checks on-device via Docker.

set -euo pipefail

PLATFORM="${1:-rk3588}"
IMAGE="${IMAGE:-openvoicestream:rk-slim-fresh-20260603}"
CORPUS="${CORPUS:-/tmp/asr_corpus}"
PERF_DIR="${PERF_DIR:-/tmp/seeed-perf}"
PATCH_DIR="${PATCH_DIR:-/tmp/slv-warmup-patch}"
OUT_DIR="${OUT_DIR:-/tmp/qwen3-quality-boundary}"

case "$PLATFORM" in
  rk3588|rk3576) ;;
  *) echo "unsupported platform: $PLATFORM" >&2; exit 2 ;;
esac

mkdir -p "$OUT_DIR"

run_case() {
  local quant="$1"
  local manifest="$2"
  local category="$3"
  local lang="$4"
  local label="${PLATFORM}_${quant}_${manifest%.json}_${category}_${lang}"
  local out="${OUT_DIR}/${label}.jsonl"

  echo "CASE=$label"
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
    -e ASR_DECODER_QUANT="$quant" \
    -e ASR_DECODER_EMBED_CACHE_REUSE=1 \
    -e ASR_DECODER_ASYNC=1 \
    -e ASR_NPU_CORE_MASK=NPU_CORE_AUTO \
    -e RK_PERF_MODE=highperf \
    "$IMAGE" \
    /opt/venv/bin/python "$PERF_DIR/qwen3_asr_stream_eos_bench.py" \
      --corpus "$CORPUS" \
      --manifest "$manifest" \
      --category "$category" \
      --lang "$lang" \
      --limit 0 \
      --chunk-ms 250 \
      --realtime \
      --platform "$PLATFORM" \
      --decoder-quant "$quant" \
      --perf-mode highperf \
      --max-context-len 512 \
      --enabled-cpus 4 \
    | tee "$out" \
    | grep '"summary"'
}

for quant in w8a8 fp16; do
  run_case "$quant" manifest.json short all
  run_case "$quant" manifest.json long all
  run_case "$quant" multilingual_manifest.json short all
  run_case "$quant" multilingual_manifest.json long all
done

echo "OUT_DIR=$OUT_DIR"
