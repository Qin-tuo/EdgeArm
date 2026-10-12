"""A running daemon must not conceal BSP features missing for containers."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'kernel_doctor', Path(__file__).resolve().parents[2] / 'scripts/doctor.py')
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def test_incomplete_bsp_blocks_all_standard_container_requirements():
    checks = {c['name']: c for c in doctor.kernel_checks(
        'CONFIG_NAMESPACES=y\nCONFIG_CGROUPS=y\n# CONFIG_BPF_SYSCALL is not set\n')}
    assert 'CONFIG_BPF_SYSCALL' in checks['docker_kernel_runtime']['detail']['missing']
    assert checks['docker_kernel_memory_limits']['detail']['missing'] == ['CONFIG_MEMCG']
    assert checks['docker_kernel_overlay2']['detail']['missing'] == ['CONFIG_OVERLAY_FS']
    assert all(c['status'] == 'blocked' for c in checks.values())


def test_supported_modules_and_builtins_pass_config_check():
    groups = doctor.kernel_checks('')
    symbols = {key for c in groups for key in c['detail']['required']}
    config = '\n'.join(f'{key}=m' if key == 'CONFIG_OVERLAY_FS' else f'{key}=y' for key in symbols)
    assert all(c['status'] == 'ok' for c in doctor.kernel_checks(config))
