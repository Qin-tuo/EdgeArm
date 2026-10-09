"""Check experiment grouping and asset manifests through their public CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    repository = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="rvs-tools-") as temporary:
        root = Path(temporary)
        records = []
        for task_id, status, duration, simulated in [
            (1, "completed", 10, True), (2, "completed", 30, True),
            (3, "cancelled", 7, True), (4, "completed", 100, False),
        ]:
            records.append({
                "schema_version": 1, "run_id": "test", "task_id": task_id,
                "stage": "task", "backend": "runtime", "model": "none",
                "simulated": simulated, "phase": "end", "status": status, "elapsed_ms": duration,
            })
        metrics = root / "metrics.jsonl"
        metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
        command = [sys.executable, str(repository / "tools/benchmarks/summarize.py"), str(metrics)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        groups = json.loads(result.stdout)["groups"]
        assert len(groups) == 2, "simulation and execution metrics were mixed"
        group = next(group for group in groups if group["simulated"])
        assert group["samples"] == 3 and group["successful_samples"] == 2
        assert group["status_counts"]["cancelled"] == 1 and group["p50_ms"] == 20
        assert group["p95_ms"] == 29, "percentile method changed"
        with metrics.open("a") as stream:
            stream.write(json.dumps(records[0]) + "\n")
        assert subprocess.run(command, capture_output=True, timeout=5).returncode != 0, "duplicate tasks inflated samples"

        artifact = root / "weights.test"
        artifact.write_bytes(b"independent test artifact\0")
        manifest = root / "manifest.json"
        command = [
            sys.executable, str(repository / "tools/assets/register.py"),
            "--id", "test", "--kind", "artifact", "--source", "independent-test",
            "--license", "test-only", "--output", str(manifest), str(artifact),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=5)
        data = json.loads(manifest.read_text())
        assert data["permission_verified"] is False
        assert data["files"][0]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert subprocess.run(command, capture_output=True, timeout=5).returncode != 0, "frozen manifest was overwritten"
    print("metrics grouping and asset registration checks passed")


if __name__ == "__main__":
    main()

