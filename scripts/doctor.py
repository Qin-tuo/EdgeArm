#!/usr/bin/env python3
"""Read-only inventory and runtime readiness check for the selected host."""
import argparse
import datetime
import glob
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(args, env=None):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15, env=env)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def read(path):
    try:
        return Path(path).read_text().strip().strip("\0")
    except OSError:
        return "unavailable"


def inspect():
    checks = []

    def add(name, status, detail):
        checks.append(dict(name=name, status=status, detail=detail))

    model = read('/proc/device-tree/model')
    add('board', 'ok' if 'RK3588' in model and platform.machine() == 'aarch64' else 'blocked', model)
    mem = re.search(r'MemTotal:\s+(\d+)', read('/proc/meminfo'))
    add('memory', 'info', {'total_kib': int(mem[1]) if mem else None, 'selected_class_gb': 16})
    add('kernel', 'info', platform.release())
    add('os', 'info', read('/etc/os-release'))
    for name, path in [('root_storage', '/'), ('project_storage', str(ROOT))]:
        usage = shutil.disk_usage(path)
        status = 'ok' if usage.free >= 1024**3 else ('warning' if path == '/' else 'blocked')
        add(name, status, {'path': path, 'free_bytes': usage.free, 'total_bytes': usage.total})
    devices = sorted(glob.glob('/dev/pcie-rkep-*'))
    add('rk1828_device', 'ok' if devices else 'blocked', devices)
    libs = [Path('/usr/lib/librknn3_api.so'), Path('/usr/lib/librknn3_api_rkcp.so')]
    versions = []
    for lib in libs:
        if lib.exists():
            versions += [x.decode() for x in re.findall(rb'librknn3_api version: ([0-9.]+)', lib.read_bytes())]
    add('rknn3_libraries', 'ok' if all(p.exists() for p in libs) else 'blocked',
        {'files': [str(p) for p in libs], 'versions': sorted(set(versions))})
    rc, service = command(['systemctl', 'is-active', 'rk182x.service'])
    add('rk1828_service', 'ok' if rc == 0 and service == 'active' else 'blocked', service)
    # Full argv is needed: the vendor may use version-suffixed proxy binaries.
    _, processes = command(['ps', '-eo', 'args'])
    proxy = [s for s in processes.splitlines() if re.search(r'(^|/)rknn3_transfer_proxy[^ /]*( |$)', s)]
    add('rk1828_proxy', 'ok' if proxy else 'blocked', proxy)
    docker = ROOT / '.local/bin/docker'
    env = dict(os.environ, DOCKER_CONFIG=str(ROOT / '.local/docker-config'))
    rc, version = command([str(docker), 'compose', 'version'], env)
    add('docker_compose', 'ok' if rc == 0 else 'blocked', version)
    rc, info = command([str(docker), 'info', '--format', '{{.DockerRootDir}}'], env)
    if rc == 0:
        path = Path(info)
        same_disk = path.exists() and path.stat().st_dev == ROOT.stat().st_dev
        add('docker_daemon', 'ok' if same_disk else 'blocked', info)
    else:
        add('docker_daemon', 'blocked', info)
    cards = read('/proc/asound/cards')
    add('audio', 'unverified', cards)
    add('peripherals', 'unverified', 'Microphone/AEC, speaker, D435i and five-axis servo connection require physical validation.')
    return {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'hostname': platform.node(), 'checks': checks,
            'host_prerequisites_ready': not any(c['status'] == 'blocked' for c in checks),
            'hardware_acceptance_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = inspect()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in report['checks']:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
    raise SystemExit(0 if report['host_prerequisites_ready'] else 1)
