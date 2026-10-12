#!/usr/bin/env bash
# Set RK1828 execution modes before starting any LLM worker.
set -euo pipefail
WORK_MODE="${EDGEARM_RK1828_WORK_MODE:-efficient}"
case "$WORK_MODE" in
  efficient) WORK_VALUE=0; WORK_LABEL=EFFICIENT ;;
  normal) WORK_VALUE=1; WORK_LABEL=NORMAL ;;
  performance) WORK_VALUE=2; WORK_LABEL=PERFORMANCE ;;
  *) echo 'EDGEARM_RK1828_WORK_MODE must be efficient, normal, or performance.' >&2; exit 1 ;;
esac
PREFILL_MODE="${EDGEARM_RK1828_PREFILL_MODE:-performance}"
case "$PREFILL_MODE" in
  performance) PREFILL_VALUE=1; PREFILL_LABEL=PERFORMANCE ;;
  efficient) PREFILL_VALUE=0; PREFILL_LABEL=EFFICIENT ;;
  *) echo 'EDGEARM_RK1828_PREFILL_MODE must be performance or efficient.' >&2; exit 1 ;;
esac
if [[ "$EUID" != 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi
if pgrep -f '(^|/)(rknn_qwen[^ /]*|rknn_llm[^ /]*|rkllm3-server)( |$)' >/dev/null; then
  echo 'An LLM worker is running; stop it before changing RK1828 modes.' >&2
  exit 1
fi
systemctl is-active --quiet rk182x.service
export NSMI_LOG_LEVEL="${NSMI_LOG_LEVEL:-DEBUG}"
echo "Setting Work Mode=$WORK_LABEL ($WORK_VALUE), Prefill Mode=$PREFILL_LABEL ($PREFILL_VALUE)."
timeout 15s /usr/bin/rknn-smi set -t work_mode -d 0 -c 0 -s "$WORK_VALUE"
timeout 15s /usr/bin/rknn-smi set -t prefill_mode -d 0 -c 0 -s "$PREFILL_VALUE"
echo 'Read back Work Mode:'
timeout 15s /usr/bin/rknn-smi info -t work_mode -d 0 -c 0
echo 'Read back Prefill Mode:'
timeout 15s /usr/bin/rknn-smi info -t prefill_mode -d 0 -c 0
