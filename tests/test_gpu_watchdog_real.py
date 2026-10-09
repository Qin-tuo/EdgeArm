"""Week 2 GPU watchdog unit tests.

Covers:
- platform auto-detection branches via env hints and shutil.which monkeypatching,
- hysteresis (3 failures to flip ok→false, 5 successes to recover),
- ``_parse_interval`` validation (default / clamp / invalid),
- ``status()`` schema,
- absence of heavy GPU library imports at module load time.
"""

from __future__ import annotations

import sys

import pytest

from server.core import gpu_watchdog as gw


def setup_function(_func):
    gw._reset_for_tests()


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------

def test_module_import_does_not_pull_pycuda_or_pynvml():
    # Re-import is OK; just assert the heavy libs are NOT in sys.modules
    # after importing this module fresh.
    assert "pycuda" not in sys.modules
    assert "pynvml" not in sys.modules


def test_is_ok_returns_bool():
    assert gw.is_ok() is True
    assert isinstance(gw.is_ok(), bool)


def test_status_schema_has_required_keys():
    s = gw.status()
    for key in (
        "ok",
        "platform",
        "reason",
        "last_checked_at",
        "consecutive_failures",
        "consecutive_successes",
        "last_duration_s",
        "checks_total",
        "failures_total",
    ):
        assert key in s, f"missing key {key} in status() output"


# ---------------------------------------------------------------------------
# Interval parser
# ---------------------------------------------------------------------------

def test_parse_interval_default(monkeypatch):
    monkeypatch.delenv("OVS_GPU_WATCHDOG_INTERVAL_S", raising=False)
    assert gw._parse_interval() == 5.0


def test_parse_interval_invalid_uses_default(monkeypatch):
    monkeypatch.setenv("OVS_GPU_WATCHDOG_INTERVAL_S", "not-a-number")
    assert gw._parse_interval() == 5.0


def test_parse_interval_below_min_clamps(monkeypatch):
    monkeypatch.setenv("OVS_GPU_WATCHDOG_INTERVAL_S", "0.1")
    assert gw._parse_interval() == 1.0


def test_parse_interval_custom_value(monkeypatch):
    monkeypatch.setenv("OVS_GPU_WATCHDOG_INTERVAL_S", "10")
    assert gw._parse_interval() == 10.0


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def test_detect_platform_rk_via_env(monkeypatch):
    monkeypatch.setenv("LANGUAGE_MODE", "rk")
    assert gw._detect_platform() == gw._PLATFORM_RK


def test_detect_platform_rk_via_rk_platform_env(monkeypatch):
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.setenv("RK_PLATFORM", "rk3576")
    assert gw._detect_platform() == gw._PLATFORM_RK


def test_detect_platform_cpu_only_default(monkeypatch):
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("RK_PLATFORM", raising=False)
    monkeypatch.delenv("ASR_PLATFORM", raising=False)
    # Force every shutil.which lookup to return None.
    monkeypatch.setattr(gw.shutil, "which", lambda _x: None)
    # And pretend /etc/nv_tegra_release doesn't exist.

    class _FakePath:
        def __init__(self, p): self._p = p
        def exists(self): return False
        def iterdir(self): return []  # pragma: no cover
        def read_text(self, errors="strict"): return ""  # pragma: no cover

    monkeypatch.setattr(gw, "Path", _FakePath)
    assert gw._detect_platform() == gw._PLATFORM_CPU


def test_detect_platform_desktop_cuda_via_nvidia_smi(monkeypatch):
    monkeypatch.delenv("LANGUAGE_MODE", raising=False)
    monkeypatch.delenv("RK_PLATFORM", raising=False)
    monkeypatch.delenv("ASR_PLATFORM", raising=False)

    class _FakePath:
        def __init__(self, p): self._p = p
        def exists(self): return False
        def iterdir(self): return []  # pragma: no cover

    monkeypatch.setattr(gw, "Path", _FakePath)
    monkeypatch.setattr(
        gw.shutil,
        "which",
        lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None,
    )
    assert gw._detect_platform() == gw._PLATFORM_DESKTOP_CUDA


# ---------------------------------------------------------------------------
# CPU-only probe
# ---------------------------------------------------------------------------

def test_check_cpu_only_returns_ok():
    ok, reason = gw._check_cpu_only()
    assert ok is True
    assert reason == "cpu_only"


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

def test_three_consecutive_failures_flip_ok_to_false():
    # Seed platform so failure metric label is stable.
    gw._status.platform = gw._PLATFORM_CPU
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    # Two failures: still ok (in grace window)
    assert gw.is_ok() is True
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    # Third failure crosses the threshold.
    assert gw.is_ok() is False
    assert gw.status()["consecutive_failures"] >= 3


def test_four_successes_after_failure_still_not_ok():
    gw._status.platform = gw._PLATFORM_CPU
    for _ in range(3):
        gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    assert gw.is_ok() is False
    for _ in range(4):
        gw._apply_check_result(True, gw._REASON_OK, 0.001)
    assert gw.is_ok() is False, "must require 5 successes to recover"


def test_five_successes_after_failure_recover():
    gw._status.platform = gw._PLATFORM_CPU
    for _ in range(3):
        gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    assert gw.is_ok() is False
    for _ in range(5):
        gw._apply_check_result(True, gw._REASON_OK, 0.001)
    assert gw.is_ok() is True
    assert gw.status()["reason"] == gw._REASON_OK


def test_intermittent_failure_does_not_flip():
    gw._status.platform = gw._PLATFORM_CPU
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._apply_check_result(True, gw._REASON_OK, 0.001)
    # A single success resets consecutive_failures so we don't flip.
    assert gw.is_ok() is True
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    assert gw.is_ok() is True
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    assert gw.is_ok() is False


# ---------------------------------------------------------------------------
# Status fields
# ---------------------------------------------------------------------------

def test_status_updates_after_check_application():
    gw._status.platform = gw._PLATFORM_CPU
    gw._apply_check_result(True, gw._REASON_OK, 0.002)
    s = gw.status()
    assert s["checks_total"] == 1
    assert s["last_duration_s"] == 0.002
    assert s["consecutive_successes"] == 1
    assert s["consecutive_failures"] == 0


def test_failures_total_increments():
    gw._status.platform = gw._PLATFORM_CPU
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    s = gw.status()
    assert s["failures_total"] == 2


# ---------------------------------------------------------------------------
# Probe dispatch (synthetic platform)
# ---------------------------------------------------------------------------

def test_check_once_sync_cpu_only():
    ok, reason, duration = gw._check_once_sync(gw._PLATFORM_CPU)
    assert ok is True
    assert duration >= 0
    assert reason == "cpu_only"


def test_check_once_sync_handles_unknown_platform():
    ok, reason, duration = gw._check_once_sync("nonexistent")
    assert ok is True  # falls back to cpu_only
    assert duration >= 0


# ---------------------------------------------------------------------------
# Start/stop lifecycle
# ---------------------------------------------------------------------------

def test_start_and_stop_lifecycle():
    import asyncio

    async def _drive():
        await gw.start()
        await gw.start()  # idempotent — should not raise
        # Brief await so the task actually schedules an iteration.
        await asyncio.sleep(0.02)
        await gw.stop()
        # Second stop is safe.
        await gw.stop()

    asyncio.run(_drive())


def test_reset_for_tests_clears_state():
    gw._status.platform = gw._PLATFORM_CPU
    gw._apply_check_result(False, gw._REASON_FAILED, 0.001)
    gw._reset_for_tests()
    s = gw.status()
    assert s["checks_total"] == 0
    assert s["ok"] is True
    assert s["consecutive_failures"] == 0
