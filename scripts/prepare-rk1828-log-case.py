#!/usr/bin/env python3
"""Prepare a fresh manufacturer log case and local terminal launchers."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_mode', choices=('efficient', 'normal', 'performance'))
    parser.add_argument('prefill_mode', choices=('efficient', 'performance'))
    args = parser.parse_args()
    active = subprocess.run(
        ['pgrep', '-af', r'(^|/)(rknn-console|rknn_qwen[^ /]*|rknn_llm[^ /]*|rkllm3-server)( |$)'],
        capture_output=True, text=True)
    if active.returncode not in (0, 1):
        raise SystemExit('Could not check existing console/model processes.')
    if active.stdout.strip():
        raise SystemExit('Stop existing console/model terminals first:\n' + active.stdout)
    model = ROOT / '.local/data/llm/artifacts.local.json'
    runtime = ROOT / '.local/runtime/rk1828-v1.1.0/MANIFEST.json'
    for source in (model, runtime):
        if not source.is_file():
            raise SystemExit('Required local manifest missing: ' + str(source))
    stamp = datetime.now().astimezone()
    case_id = stamp.strftime('%Y%m%d-%H%M%S-%f') + f'-{args.work_mode}-{args.prefill_mode}'
    case = ROOT / '.local/reports/inference-check/vendor-logs' / case_id
    case.mkdir(parents=True, exist_ok=False)
    metadata = {
        'case_id': case_id, 'work_mode': args.work_mode,
        'work_mode_value': ('efficient', 'normal', 'performance').index(args.work_mode),
        'prefill_mode': args.prefill_mode,
        'prefill_mode_value': ('efficient', 'performance').index(args.prefill_mode),
        'RKNN_LOG_LEVEL': 3,
        'console_commands': ['uart_irq_set_owner subsoc', 'rknn_set_log_level 3',
                             'rknn_server_set_log_level 2'],
        'model': 'Qwen3-4B', 'runtime': 'matched V1.1.0 project worker',
        'root': str(ROOT), 'created_local': stamp.isoformat(),
        'status': 'prepared_not_run',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    (case / 'case.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(model, case / 'model-manifest.json')
    shutil.copyfile(runtime, case / 'runtime-manifest.json')
    launchers = ROOT / '.local/diagnostics'
    launchers.mkdir(parents=True, exist_ok=True)
    commands = {
        'start-manufacturer-console.sh': [str(ROOT / 'scripts/capture-rk1828-console.sh'), str(case)],
        'start-manufacturer-llm.sh': [str(ROOT / 'scripts/capture-rk1828-llm.sh'), args.work_mode,
                                     args.prefill_mode, str(case)],
    }
    for name, command in commands.items():
        target = launchers / name
        target.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n')
        target.chmod(0o755)
    (launchers / 'manufacturer-active-case.txt').write_text(str(case) + '\n')
    print(case)
    print('Prepared only; no mode change, service restart, reset, or inference.')


if __name__ == '__main__':
    main()
