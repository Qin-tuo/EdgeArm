#!/usr/bin/env python3
"""Read-only inventory and runtime readiness check for the selected host."""
import argparse
import datetime
import glob
import gzip
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def kernel_checks(config):
    """Required features for the standard daemon and resource limits."""
    enabled = {line.split('=', 1)[0] for line in config.splitlines()
               if line.endswith(('=y', '=m'))}
    groups = {
        'docker_kernel_runtime': ['CONFIG_NAMESPACES', 'CONFIG_NET_NS', 'CONFIG_PID_NS',
                                 'CONFIG_IPC_NS', 'CONFIG_UTS_NS', 'CONFIG_POSIX_MQUEUE',
                                 'CONFIG_CGROUPS', 'CONFIG_BPF_SYSCALL', 'CONFIG_CGROUP_BPF',
                                 'CONFIG_SECCOMP', 'CONFIG_SECCOMP_FILTER'],
        'docker_kernel_memory_limits': ['CONFIG_MEMCG'],
        'docker_kernel_pids_limits': ['CONFIG_CGROUP_PIDS'],
        'docker_kernel_overlay2': ['CONFIG_OVERLAY_FS'],
        'docker_kernel_bridge': ['CONFIG_VETH', 'CONFIG_BRIDGE', 'CONFIG_BRIDGE_NETFILTER',
                                 'CONFIG_NF_CONNTRACK', 'CONFIG_NF_NAT',
                                 'CONFIG_IP_NF_IPTABLES', 'CONFIG_IP_NF_FILTER',
                                 'CONFIG_IP_NF_MANGLE', 'CONFIG_NETFILTER_XT_MATCH_ADDRTYPE',
                                 'CONFIG_NETFILTER_XT_MATCH_CONNTRACK',
                                 'CONFIG_NETFILTER_XT_TARGET_MASQUERADE'],
    }
    return [{'name': name, 'status': 'blocked' if any(key not in enabled for key in keys) else 'ok',
             'detail': {'missing': [key for key in keys if key not in enabled],
                        'required': keys}}
            for name, keys in groups.items()]


def running_kernel_config():
    try:
        return '/proc/config.gz', gzip.decompress(Path('/proc/config.gz').read_bytes()).decode()
    except (OSError, EOFError):
        path = Path('/boot') / f'config-{platform.release()}'
        try:
            return str(path), path.read_text()
        except OSError:
            return None, None


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
    source, config = running_kernel_config()
    if config is None:
        add('docker_kernel_config', 'unverified', 'Running kernel config is not readable.')
    else:
        for check in kernel_checks(config):
            check['detail']['config_source'] = source
            add(check['name'], check['status'], check['detail'])
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
    proxy = [s for s in processes.splitlines() if s.split()
             and Path(s.split()[0]).name.startswith('rknn3_transfer_proxy')]
    parent = any(Path(s.split()[0]).name == 'rknn3_transfer_proxy' for s in proxy)
    child = any('-s' in s.split() for s in proxy)
    add('rk1828_proxy', 'ok' if parent and child else 'blocked',
        {'parent_present': parent, 'device_proxy_present': child, 'processes': proxy})
    docker = ROOT / '.local/bin/docker'
    env = dict(os.environ, DOCKER_CONFIG=str(ROOT / '.local/docker-config'))
    rc, version = command([str(docker), 'compose', 'version'], env)
    add('docker_compose', 'ok' if rc == 0 else 'blocked', version)
    rc, buildx = command([str(docker), 'buildx', 'version'], env)
    add('docker_buildx', 'ok' if rc == 0 else 'blocked', buildx)
    rc, info = command([str(docker), 'info', '--format', '{{json .}}'], env)
    if rc == 0:
        try:
            details = json.loads(info)
            path = Path(details['DockerRootDir'])
            same_disk = path.exists() and path.stat().st_dev == ROOT.stat().st_dev
            add('docker_daemon', 'ok' if same_disk else 'blocked',
                {'data_root': str(path), 'storage_driver': details.get('Driver'),
                 'cgroup_version': details.get('CgroupVersion')})
            limits = {key: details.get(key) for key in ('MemoryLimit', 'SwapLimit')}
            add('docker_resource_limits', 'ok' if all(limits.values()) else 'blocked', limits)
        except (OSError, ValueError, KeyError) as error:
            add('docker_daemon', 'blocked', str(error))
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
