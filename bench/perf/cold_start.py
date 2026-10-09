"""Cold-start time measurement.

Two modes:
  - 'container': docker restart <name>, poll /health until 200 (typical fleet scenario)
  - 'process':   already-running service, just measure from a marker timestamp

Output: boot_ms = wall time from start signal to first /health 200.
"""
from __future__ import annotations
import subprocess, time
from typing import Optional
import requests


def measure_container_boot(container: str, base_url: str,
                           health_path: str = "/health",
                           timeout_s: float = 300.0,
                           poll_interval_s: float = 0.25) -> dict:
    """Restart container, poll /health, return boot timing."""
    t0 = time.monotonic()
    subprocess.run(["docker", "restart", container], check=True,
                   capture_output=True, timeout=30)
    t_restart_done = time.monotonic()

    t_first_200: Optional[float] = None
    t_first_connect: Optional[float] = None
    last_err = None
    while time.monotonic() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url.rstrip('/')}{health_path}", timeout=2)
            if t_first_connect is None:
                t_first_connect = time.monotonic()
            if r.status_code == 200:
                t_first_200 = time.monotonic()
                break
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = type(e).__name__
        time.sleep(poll_interval_s)

    if t_first_200 is None:
        return {
            "container": container,
            "boot_ms": None,
            "error": f"health never returned 200 within {timeout_s}s (last: {last_err})",
        }

    return {
        "container": container,
        "docker_restart_ms": (t_restart_done - t0) * 1000,
        "first_connect_ms": (t_first_connect - t0) * 1000 if t_first_connect else None,
        "boot_ms":          (t_first_200 - t0) * 1000,
        "boot_after_restart_ms": (t_first_200 - t_restart_done) * 1000,
    }
