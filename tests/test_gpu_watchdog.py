"""Unit tests for server.core.gpu_watchdog (Week 1 stub)."""

from server.core import gpu_watchdog


def test_is_ok_returns_true():
    assert gpu_watchdog.is_ok() is True


def test_import_does_not_pull_hardware_libs():
    # Re-import should not raise even on a host without CUDA / RKNN.
    import importlib
    importlib.reload(gpu_watchdog)
    assert gpu_watchdog.is_ok() is True
