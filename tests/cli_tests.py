"""Black-box checks for configuration, output contracts and simulated artifacts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave


def run(binary, config, output, *extra):
    return subprocess.run(
        [str(binary), "--config", str(config), "--output-root", str(output), *extra],
        text=True, capture_output=True, timeout=8,
    )


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    binary, repository = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="rvs-tests-") as temporary:
        root = Path(temporary)
        text = '你好 "quoted"\nnext line'
        echo = run(binary, repository / "configs/echo.ini", root / "echo", "--input", text, "--repeat", "2")
        check(echo.returncode == 0, echo.stderr)
        rows = [json.loads(line) for line in echo.stdout.splitlines()]
        check(len(rows) == 3 and rows[0]["output"] == "回显：" + text, "echo contract failed")
        check(rows[0]["simulated"] is False, "builtin echo mislabeled")
        log = Path(rows[-1]["run_directory"]) / "metrics.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines()]
        check(sum(e["stage"] == "task" for e in events) == 2, "terminal events missing")
        check(all(e["elapsed_ms"] >= 0 for e in events), "negative duration")

        voice = run(binary, repository / "configs/voice-mock.ini", root / "voice", "--input", text)
        check(voice.returncode == 0, voice.stderr)
        rows = [json.loads(line) for line in voice.stdout.splitlines()]
        check(rows[0]["simulated"] is True, "mock mislabeled")
        check(rows[0]["output"] == "模拟回答：" + text, "voice packet stages failed")
        with wave.open(rows[0]["artifact"], "rb") as audio:
            check(audio.getnchannels() == 1 and audio.getsampwidth() == 2, "invalid WAV encoding")
            check(audio.getframerate() == 16000 and audio.getnframes() > 0, "invalid WAV dimensions")
            check(set(audio.readframes(audio.getnframes())) == {0}, "mock WAV should contain silence")
        events = [json.loads(line) for line in (Path(rows[-1]["run_directory"]) / "metrics.jsonl").read_text().splitlines()]
        check(all(e["simulated"] for e in events), "mock metrics mislabeled")
        check({e["stage"] for e in events if e["phase"] == "first_output"} == {"llm", "tts"}, "first output events missing")

        unsupported = run(binary, repository / "configs/voice-cpu.example.ini", root / "unsupported")
        check(unsupported.returncode == 2 and "unsupported" in unsupported.stderr, "real backend silently fell back")
        check(not (root / "unsupported").exists(), "invalid backend created run artifacts")

        invalid = root / "invalid.ini"
        invalid.write_text("[runtime]\nqueue_capacity=0\n[pipeline]\nstages=echo\n")
        failure = run(binary, invalid, root / "invalid-run")
        check(failure.returncode == 2 and "invalid_input" in failure.stderr, "invalid configuration accepted")
        duplicate = root / "duplicate.ini"
        duplicate.write_text("[pipeline]\nstages=echo\nstages=echo\n")
        check("duplicate" in run(binary, duplicate, root / "duplicate-run").stderr, "duplicate key accepted")

        slow = root / "slow.ini"
        slow.write_text("[runtime]\ntask_timeout_ms=5000\n[pipeline]\nstages=echo\n[echo]\ndelay_ms=100\n")
        cancelled = run(binary, slow, root / "cancelled", "--cancel-after-ms", "1")
        check(cancelled.returncode == 1, "cancelled CLI task reported success")
        rows = [json.loads(line) for line in cancelled.stdout.splitlines()]
        check(rows[0]["status"] == "cancelled" and "output" not in rows[0], "cancelled output leaked")
        events = [json.loads(line) for line in (Path(rows[-1]["run_directory"]) / "metrics.jsonl").read_text().splitlines()]
        check(any(e["stage"] == "task" and e["status"] == "cancelled" for e in events), "cancelled metrics missing")
    print("CLI, metrics, WAV, backend and configuration behavior checks passed")


if __name__ == "__main__":
    main()
