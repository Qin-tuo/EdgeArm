#!/usr/bin/env bash
# Capture host SDK logs for one explicit mode combination. No reset/restart.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$EUID" == 0 ]] || { echo 'Run with sudo: RK1828 model access requires root.' >&2; exit 1; }
[[ $# == 3 && -d "$3" ]] || {
  echo "Usage: sudo $0 <efficient|normal|performance> <efficient|performance> <case-directory>" >&2; exit 1;
}
WORK="$1"; PREFILL="$2"; CASE="$(realpath "$3")"
case "$WORK" in efficient|normal|performance) ;; *) echo 'Invalid work mode.' >&2; exit 1;; esac
case "$PREFILL" in efficient|performance) ;; *) echo 'Invalid prefill mode.' >&2; exit 1;; esac
[[ "$CASE" == "$ROOT/.local/reports/inference-check/vendor-logs/"* ]] || { echo 'Invalid case directory.' >&2; exit 1; }
[[ -f "$CASE/case.json" && -f "$CASE/rk182x-console.log" ]] || { echo 'Prepare case and console recording first.' >&2; exit 1; }
[[ ! -e "$CASE/host.exit-code" && ! -e "$CASE/rk3588-host.log" ]] || { echo 'Case already started; use a new directory.' >&2; exit 1; }
pgrep -x rknn-console >/dev/null || { echo 'Manufacturer console is not running.' >&2; exit 1; }
"$ROOT/.venv/bin/python" - "$CASE" "$WORK" "$PREFILL" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
case = json.loads((p / 'case.json').read_text())
if (case['work_mode'], case['prefill_mode']) != tuple(sys.argv[2:]):
    raise SystemExit('Requested modes do not match the prepared case.')
log = (p / 'rk182x-console.log').read_text(errors='replace')
for command in ('uart_irq_set_owner subsoc', 'rknn_set_log_level 3', 'rknn_server_set_log_level 2'):
    if command not in log:
        raise SystemExit('Enter manufacturer console command first: ' + command)
PY
export RKNN_LOG_LEVEL=3
export EDGEARM_RK1828_WORK_MODE="$WORK" EDGEARM_RK1828_PREFILL_MODE="$PREFILL"
export RK1828_ALLOW_TOOLS=0 RK1828_KV_REUSE=0
export EDGEARM_NATIVE_LLM_RUNTIME="$ROOT/.local/runtime/rk1828-v1.1.0"
START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
set +e
{
  date --iso-8601=seconds
  printf 'RKNN_LOG_LEVEL=%s work=%s prefill=%s\n' "$RKNN_LOG_LEVEL" "$WORK" "$PREFILL"
  printf 'Console commands must already be entered before this application starts.\n'
  "$ROOT/scripts/start-native-llm.sh"
} 2>&1 | tee "$CASE/rk3588-host.log"
RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$RC" > "$CASE/host.exit-code"
date --iso-8601=seconds > "$CASE/host-ended.txt"
journalctl -b -u rk182x.service --since "$START_TIME" --no-pager -o short-precise > "$CASE/proxy-journal.log" || true
journalctl -k -b --since "$START_TIME" --no-pager -o short-precise > "$CASE/kernel-journal.log" || true
dmesg --ctime > "$CASE/dmesg.log" 2>&1 || true
chmod a+r "$CASE/rk3588-host.log" "$CASE/host.exit-code" "$CASE/host-ended.txt"
exit "$RC"
