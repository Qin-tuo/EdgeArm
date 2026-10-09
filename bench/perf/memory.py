"""Background memory sampler for `docker stats <container>`.

Polls every 500ms while a runner is active, returns peak MiB.
Optional — if no container name given, results just omit the memory field.
"""
from __future__ import annotations
import shutil, subprocess, threading, time


class MemorySampler:
    def __init__(self, container: str | None, interval_s: float = 0.5):
        self.container = container
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: float = 0.0
        self.samples: list[float] = []
        self._available = bool(container) and shutil.which("docker") is not None

    def __enter__(self):
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                # format: "1.234GiB / 7.5GiB" or "456.7MiB / 7.5GiB"
                usage = out.split("/")[0].strip()
                mib = self._parse_mib(usage)
                if mib > 0:
                    self.samples.append(mib)
                    self.peak_mib = max(self.peak_mib, mib)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    @staticmethod
    def _parse_mib(s: str) -> float:
        s = s.strip()
        if s.endswith("GiB"):
            return float(s[:-3]) * 1024
        if s.endswith("MiB"):
            return float(s[:-3])
        if s.endswith("KiB"):
            return float(s[:-3]) / 1024
        return 0.0

    def summary(self) -> dict:
        if not self._available or not self.samples:
            return {}
        return {
            "peak_mib": round(self.peak_mib, 1),
            "mean_mib": round(sum(self.samples) / len(self.samples), 1),
            "samples": len(self.samples),
        }
