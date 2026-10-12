#!/usr/bin/env python3
"""Package a stopped manufacturer log case, preserving the raw logs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / '.local/reports/inference-check'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory', type=Path)
    args = parser.parse_args()
    case = args.case_directory.resolve()
    case.relative_to(REPORTS / 'vendor-logs')
    for name in ('case.json', 'requests.json', 'rk182x-console.log', 'rk3588-host.log',
                 'host.exit-code', 'kernel-journal.log', 'proxy-journal.log', 'dmesg.log'):
        if not (case / name).is_file():
            raise SystemExit('Capture not finalized; missing ' + name)
    ep = (case / 'rk182x-console.log').read_bytes()
    ep_text = ep.decode(errors='replace')
    if 'Script done on' not in ep_text:
        raise SystemExit('Exit the console with Ctrl+X before packaging.')
    for ack in ('RKNN log level set to 3', 'RKNN Server log level set to 2'):
        if ack not in ep_text:
            raise SystemExit('Missing console acknowledgement: ' + ack)
    host = (case / 'rk3588-host.log').read_text(errors='replace')
    if 'RKNN_LOG_LEVEL=3' not in host:
        raise SystemExit('Host log level was not recorded.')
    report = json.loads((case / 'requests.json').read_text())
    if report['status'] not in ('failed', 'sample_passed'):
        raise SystemExit('Request replay has not finished.')
    meta = json.loads((case / 'case.json').read_text())
    modes = meta['work_mode'] + '-' + meta['prefill_mode']
    date = datetime.fromisoformat(report['started_utc']).astimezone().strftime('%Y%m%d')
    archive = REPORTS / f'rk1828-manufacturer-logs-{modes}-{date}.zip'
    if archive.exists():
        raise SystemExit('Archive already exists; preserve it and use a fresh case.')
    meta.update(status=report['status'], capture_started_utc=report['started_utc'],
                capture_ended_utc=report['ended_utc'])
    rows = report['cases']
    last = rows[-1] if rows else None
    if last:
        write_json(case / 'last-request.json', last['request'])
        request_id = last['turn'] - 1
        matches = re.findall(r'req#(\d+) done prefill_tokens=(\d+).*?decode_tokens=(\d+)', host)
        metrics = next((m for m in reversed(matches) if int(m[0]) == request_id), None)
        if metrics:
            meta['last_worker_metrics'] = {'request_id': request_id,
                'prefill_tokens': int(metrics[1]), 'decode_tokens': int(metrics[2])}
        if report['status'] == 'failed':
            meta.update(failure_turn=last['turn'], http_status=last.get('http_status'),
                        error=report.get('error'))
            write_json(case / 'failed-request.json', last['request'])
        start = host.find(f'[server] req#{request_id} max_new=')
        (case / 'rk3588-failure-excerpt.txt').write_text(host[max(0, start-500):])
    cut = ep.find(b'Warning: PCIe receive length (4294967295)')
    (case / 'rk182x-failure-excerpt.txt').write_bytes(
        ep[max(0, cut-18000):cut+800] if cut >= 0 else ep[-18000:])
    write_json(case / 'case.json', meta)
    processes = subprocess.check_output(['ps', '-eo', 'pid,user,args'], text=True)
    selected = []
    for line in processes.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and re.match(
            r'(?:/[^ ]*/)?(?:rknn3_transfer_proxy[^ ]*|rknn-console|rknn_qwen[^ ]*)\b', fields[2]):
            selected.append(line)
    write_json(case / 'post-stop-state.json', {'collected_local': datetime.now().astimezone().isoformat(),
                                             'processes': selected})
    (case / 'README.txt').write_text(
        f'RK1828 manufacturer logs: {modes}\n'
        f'Actual replay UTC: {report["started_utc"]} — {report["ended_utc"]}\n'
        f'Result: {report["status"]}; successful requests: '
        f'{sum(bool(r.get("transport_passed")) for r in rows)}\n'
        f'Last worker metrics: {meta.get("last_worker_metrics")}\n'
        f'Error: {report.get("error")}\n\n'
        'Console: uart_irq_set_owner subsoc; rknn_set_log_level 3; '
        'rknn_server_set_log_level 2. Acknowledgements verified.\n'
        'Host: RKNN_LOG_LEVEL=3, full application stdout/stderr; '
        'worker stderr has Python shim timestamps.\n'
        'Paired V1.1.0 worker/libraries, Qwen3-4B, core mask ff, context 2048, '
        'max_new 512, tools/KV reuse disabled. Actual mode readback is in host log.\n'
        'Raw .log files include initialization, inference, failure and console termination. '
        'Excerpts do not replace the originals; non-text console bytes are preserved.\n'
        'Requests/responses, system logs, model/runtime manifests and capture source included. '
        'No model weights included. No automatic inference retry/reset/restart.\n'
        'A communication failure alone does not identify the root cause or a token threshold. '
        'Debug logging affects speed; timings are not performance measurements.\n')
    write_json(case / 'SHA256.json', {p.name: {'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in case.iterdir() if p.is_file() and p.name != 'SHA256.json'})
    source = ['scripts/capture-rk1828-console.sh', 'scripts/capture-rk1828-llm.sh',
              'scripts/prepare-rk1828-log-case.py', 'scripts/reproduce-rk1828-modes.py',
              'scripts/package-rk1828-logs.py', 'scripts/setup-rk1828-mode.sh',
              'scripts/start-native-llm.sh', 'services/rk1828-llm/patches/Qwen3-cpp-main.cc',
              'services/rk1828-llm/rk1828_llm_server.py']
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(case.iterdir()):
            if p.is_file():
                z.write(p, modes + '/' + p.name)
        for name in source:
            z.write(ROOT / name, 'source/' + name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        checks = json.loads(z.read(modes + '/SHA256.json'))
        assert all(hashlib.sha256(z.read(modes + '/' + n)).hexdigest() == v['sha256']
                   for n, v in checks.items())
    print(archive)
    print('PASS archive integrity and raw file hashes; no device access performed.')


if __name__ == '__main__':
    main()
