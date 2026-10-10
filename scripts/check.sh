#!/usr/bin/env bash
# Software checks: bundled CPU VAD only; no device open, external models or daemon startup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.local/bin:$ROOT/.venv/bin:$PATH"
export TMPDIR="$ROOT/.local/tmp"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/agent:$ROOT"
mkdir -p "$TMPDIR" .local/reports
.venv/bin/python -m pip check
dockerd --validate --config-file deploy/docker-daemon.json
scripts/compose.sh config --format json > .local/reports/so101-compose.json
.venv/bin/python scripts/audit.py
.venv/bin/python scripts/run-tests.py agent/tests server/tests tests \
  --junitxml=.local/reports/so101-tests.xml | tee .local/reports/so101-tests.log
