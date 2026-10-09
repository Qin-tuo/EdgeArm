"""GPU/NPU health watchdog (Week 2 production hardening).

Runs a single asyncio background task that probes the local accelerator
periodically (default 5 s). The probe is platform-auto-detected from
host artefacts and env hints — never blocks the request path. Readiness
(/readyz) and /metrics read ONLY cached state; live probing happens in
the background loop.

Hysteresis: 3 consecutive raw-probe failures flip the cached state to
NOT-OK; 5 consecutive successes recover. Single transient failures do
not bounce orchestrators.

Heavy GPU libraries (pycuda/pynvml/etc.) are imported lazily inside
probe functions and treated as optional; their absence never fails
startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INTERVAL_S = 5.0
_MIN_INTERVAL_S = 1.0
_FAIL_THRESHOLD = 3      # N: consecutive failures to flip ok→false
_RECOVER_THRESHOLD = 5   # M: consecutive successes to flip false→true
_PROBE_TIMEOUT_S = 2.0

_PLATFORM_JETSON = "jetson"
_PLATFORM_RK = "rk"
_PLATFORM_HAILO = "hailo"
_PLATFORM_DESKTOP_CUDA = "desktop_cuda"
_PLATFORM_CPU = "cpu_only"

# Fixed reason tokens (avoid label cardinality blowup from raw exception text)
_REASON_OK = "ok"
_REASON_TIMEOUT = "probe_timeout"
_REASON_UNAVAILABLE = "probe_unavailable"
_REASON_FAILED = "probe_failed"
_REASON_TASK_CRASHED = "watchdog_task_crashed"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WatchdogStatus:
    ok: bool = True
    platform: str = "unknown"
    reason: str = _REASON_OK
    last_checked_at: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_duration_s: Optional[float] = None
    checks_total: int = 0
    failures_total: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "platform": self.platform,
            "reason": self.reason,
            "last_checked_at": self.last_checked_at,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_duration_s": self.last_duration_s,
            "checks_total": self.checks_total,
            "failures_total": self.failures_total,
        }


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_status = WatchdogStatus()
_task: Optional[asyncio.Task] = None
_stopping = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_interval() -> float:
    raw = os.environ.get("OVS_GPU_WATCHDOG_INTERVAL_S")
    if raw is None or not str(raw).strip():
        return _DEFAULT_INTERVAL_S
    try:
        v = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "gpu_watchdog: invalid OVS_GPU_WATCHDOG_INTERVAL_S=%r; using default %.1fs",
            raw, _DEFAULT_INTERVAL_S,
        )
        return _DEFAULT_INTERVAL_S
    if v < _MIN_INTERVAL_S:
        return _MIN_INTERVAL_S
    return v


def _run_command(cmd: list[str], timeout: float = _PROBE_TIMEOUT_S) -> Tuple[bool, str]:
    """Run a shell probe with a short timeout. Return (ok, reason_token)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, _REASON_TIMEOUT
    except FileNotFoundError:
        return False, _REASON_UNAVAILABLE
    except Exception:
        logger.exception("gpu_watchdog: probe command crashed: %s", cmd)
        return False, _REASON_FAILED
    if proc.returncode != 0:
        return False, _REASON_FAILED
    return True, _REASON_OK


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _detect_platform() -> str:
    """Detect host platform from filesystem + env hints. Pure function.

    Order:
      1. RK if /sys/class/devfreq present or env hint says so.
      2. Jetson if /etc/nv_tegra_release present or tegrastats on PATH.
      3. Hailo if hailortcli on PATH.
      4. Desktop CUDA if nvidia-smi on PATH.
      5. CPU-only fallback.
    """
    mode = (os.environ.get("LANGUAGE_MODE") or "").strip().lower()
    rk_platform = (os.environ.get("RK_PLATFORM") or "").strip().lower()
    asr_platform = (os.environ.get("ASR_PLATFORM") or "").strip().lower()
    if mode == "rk" or rk_platform.startswith("rk") or asr_platform.startswith("rk"):
        return _PLATFORM_RK
    if Path("/sys/class/devfreq").exists():
        # devfreq exists on many ARM SoCs; treat as RK only when there's
        # an actual NPU sysfs hint or env. We already covered env hints
        # above; without those, fall through.
        if Path("/sys/kernel/debug/rknpu").exists():
            return _PLATFORM_RK

    if Path("/etc/nv_tegra_release").exists():
        return _PLATFORM_JETSON
    if shutil.which("tegrastats") is not None:
        return _PLATFORM_JETSON

    if shutil.which("hailortcli") is not None:
        return _PLATFORM_HAILO

    if shutil.which("nvidia-smi") is not None:
        return _PLATFORM_DESKTOP_CUDA

    return _PLATFORM_CPU


# ---------------------------------------------------------------------------
# Platform probes
# ---------------------------------------------------------------------------

def _check_jetson() -> Tuple[bool, str]:
    # Prefer the cheap path: read /etc/nv_tegra_release once + tegrastats
    # snapshot via --interval 1 (start, capture one line, kill). Tegrastats
    # without timeout hangs forever; rely on subprocess timeout.
    if Path("/etc/nv_tegra_release").exists():
        # Just confirming the file is readable is enough as a baseline
        # signal — paired with an nvidia-smi or tegrastats probe below.
        pass

    if shutil.which("tegrastats") is not None:
        ok, reason = _run_command(["tegrastats", "--interval", "1000"], timeout=_PROBE_TIMEOUT_S)
        # tegrastats normally never exits; subprocess timeout raised is
        # expected and treated as failure only if there's no other signal.
        if reason == _REASON_TIMEOUT:
            # Timeout while reading is acceptable — process exists, GPU
            # alive. We treat that as OK so the probe stays lightweight.
            return True, _REASON_OK
        if ok:
            return True, _REASON_OK
    if shutil.which("nvidia-smi") is not None:
        return _run_command(["nvidia-smi", "-L"])
    # Optional pycuda light op (last resort; never required).
    try:
        import pycuda.driver as _cuda  # type: ignore
        _cuda.init()
        if _cuda.Device.count() > 0:
            return True, _REASON_OK
        return False, _REASON_UNAVAILABLE
    except Exception:
        return False, _REASON_UNAVAILABLE


def _check_rk() -> Tuple[bool, str]:
    # First try /sys/kernel/debug/rknpu/load (requires permissions).
    rknpu_load = Path("/sys/kernel/debug/rknpu/load")
    if rknpu_load.exists():
        try:
            content = rknpu_load.read_text(errors="ignore").strip()
            if content:
                return True, _REASON_OK
        except (PermissionError, OSError):
            pass  # fall through to devfreq

    # Fallback: any readable cur_freq under /sys/class/devfreq.
    devfreq = Path("/sys/class/devfreq")
    if devfreq.exists():
        for entry in devfreq.iterdir():
            cur = entry / "cur_freq"
            if cur.exists():
                try:
                    raw = cur.read_text(errors="ignore").strip()
                    if raw and raw.lstrip("-").isdigit():
                        return True, _REASON_OK
                except (PermissionError, OSError):
                    continue
    return False, _REASON_UNAVAILABLE


def _check_hailo() -> Tuple[bool, str]:
    if shutil.which("hailortcli") is None:
        return False, _REASON_UNAVAILABLE
    ok, reason = _run_command(["hailortcli", "fw-control", "identify"])
    if ok:
        return True, _REASON_OK
    return False, reason


def _check_desktop_cuda() -> Tuple[bool, str]:
    # Prefer NVML when available; fall back to nvidia-smi.
    try:
        import pynvml  # type: ignore
        try:
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count > 0:
                    # Touch device 0 to ensure handles work.
                    pynvml.nvmlDeviceGetHandleByIndex(0)
                    return True, _REASON_OK
                return False, _REASON_UNAVAILABLE
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass  # NVML init failed; fall through to nvidia-smi.
    except Exception:
        pass

    if shutil.which("nvidia-smi") is not None:
        return _run_command(["nvidia-smi", "-L"])
    return False, _REASON_UNAVAILABLE


def _check_cpu_only() -> Tuple[bool, str]:
    return True, "cpu_only"


_PROBES: dict[str, Callable[[], Tuple[bool, str]]] = {
    _PLATFORM_JETSON: _check_jetson,
    _PLATFORM_RK: _check_rk,
    _PLATFORM_HAILO: _check_hailo,
    _PLATFORM_DESKTOP_CUDA: _check_desktop_cuda,
    _PLATFORM_CPU: _check_cpu_only,
}


# ---------------------------------------------------------------------------
# Single-check entrypoint (sync, runs in default executor)
# ---------------------------------------------------------------------------

def _check_once_sync(platform: str) -> Tuple[bool, str, float]:
    """Run one probe. Returns ``(ok, reason, duration_s)``."""
    fn = _PROBES.get(platform, _check_cpu_only)
    t0 = time.perf_counter()
    try:
        ok, reason = fn()
    except Exception:
        logger.exception("gpu_watchdog: probe raised for platform=%s", platform)
        ok, reason = False, _REASON_FAILED
    return ok, reason, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Hysteresis state update
# ---------------------------------------------------------------------------

def _apply_check_result(ok: bool, reason: str, duration: float) -> None:
    global _status
    _status.last_checked_at = time.time()
    _status.last_duration_s = duration
    _status.checks_total += 1

    # Update metrics on raw probe result, BEFORE applying hysteresis.
    try:
        from server.core import metrics as _m
        _m.observe_gpu_watchdog_check_duration(duration)
        if not ok:
            _m.record_gpu_watchdog_failure(_status.platform, reason)
    except Exception:
        pass

    if ok:
        _status.consecutive_successes += 1
        _status.consecutive_failures = 0
        # Recovery transition: only when currently NOT-OK.
        if not _status.ok and _status.consecutive_successes >= _RECOVER_THRESHOLD:
            _status.ok = True
            _status.reason = _REASON_OK
    else:
        _status.failures_total += 1
        _status.consecutive_failures += 1
        _status.consecutive_successes = 0
        if _status.ok and _status.consecutive_failures >= _FAIL_THRESHOLD:
            _status.ok = False
            _status.reason = reason
        # Always update latest failure reason for diagnostics even when
        # still inside hysteresis grace window.
        if not _status.ok:
            _status.reason = reason

    # Always publish cached state to the gauge.
    try:
        from server.core import metrics as _m2
        _m2.set_gpu_watchdog_ok(_status.ok)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_ok() -> bool:
    """Cached health read. Never runs hardware probes."""
    return bool(_status.ok)


def status() -> dict:
    """JSON-serialisable snapshot of cached watchdog state."""
    return _status.as_dict()


async def _run_loop() -> None:
    """Background task: probe → update cache → sleep."""
    global _stopping
    interval = _parse_interval()
    loop = asyncio.get_event_loop()
    # First, identify platform once and seed reason.
    try:
        _status.platform = _detect_platform()
        _status.reason = _REASON_OK
    except Exception:
        logger.exception("gpu_watchdog: platform detection raised")
        _status.platform = _PLATFORM_CPU

    try:
        from server.core import metrics as _m
        _m.set_gpu_watchdog_ok(_status.ok)
    except Exception:
        pass

    while not _stopping:
        try:
            ok, reason, duration = await loop.run_in_executor(
                None, _check_once_sync, _status.platform,
            )
            _apply_check_result(ok, reason, duration)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("gpu_watchdog: loop iteration crashed")
            # Don't kill the loop on transient error; treat as failure.
            _apply_check_result(False, _REASON_FAILED, 0.0)

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


async def start() -> None:
    """Launch the background watchdog task. Idempotent."""
    global _task, _stopping
    if _task is not None and not _task.done():
        return
    _stopping = False
    loop = asyncio.get_event_loop()
    _task = loop.create_task(_run_with_crash_guard())


async def _run_with_crash_guard() -> None:
    """Wrap _run_loop so a crash surfaces as ``ok=False`` instead of a
    silent dead task."""
    try:
        await _run_loop()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("gpu_watchdog: background task crashed")
        _status.ok = False
        _status.reason = _REASON_TASK_CRASHED
        try:
            from server.core import metrics as _m
            _m.set_gpu_watchdog_ok(False)
        except Exception:
            pass


async def stop() -> None:
    """Cancel and await the background task."""
    global _task, _stopping
    _stopping = True
    if _task is None:
        return
    if not _task.done():
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
    _task = None


def _reset_for_tests() -> None:
    """Test-only hook: clear cached state and stop any background task."""
    global _status, _task, _stopping
    _status = WatchdogStatus()
    _stopping = True
    _task = None
