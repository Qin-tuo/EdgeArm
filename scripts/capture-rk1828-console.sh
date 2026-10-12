#!/usr/bin/env bash
# Run the manufacturer's interactive console inside a flushed terminal log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$EUID" == 0 ]] || { echo 'Run with sudo: RK1828 console access requires root.' >&2; exit 1; }
[[ $# == 1 && -d "$1" ]] || { echo "Usage: sudo $0 <existing-case-directory>" >&2; exit 1; }
CASE="$(realpath "$1")"
[[ "$CASE" == "$ROOT/.local/reports/inference-check/vendor-logs/"* ]] || {
  echo 'Case directory must be under .local/reports/inference-check/vendor-logs/.' >&2; exit 1;
}
printf 'Console recording: %s/rk182x-console.log\n' "$CASE"
printf 'Enter these three manufacturer commands, in order:\n'
printf 'uart_irq_set_owner subsoc\nrknn_set_log_level 3\nrknn_server_set_log_level 2\n'
printf 'Keep this terminal open during inference; console exit is Ctrl+X.\n'
exec /usr/bin/script -q -a -e -f -c '/usr/bin/rknn-console rk1820' "$CASE/rk182x-console.log"
