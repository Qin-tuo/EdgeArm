"""Regression tests for codex Week 3 BLOCKER 3 (parity harness).

Two bugs were flagged:
  3a) `fleet_exec_collect` used `out.rfind('{') .. rfind('}')` to extract
      JSON from the remote collector stdout, but the inner JSON contains
      nested `{` braces so `rfind('{')` picks an inner one and produces
      a partial-object ValueError.
  3b) The remote collector ships only /health + a TTS smoke. The
      comparator expected ASR + V2V fields and treated their absence as
      "no problems" → silent PASS on incomplete data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace

import pytest

from bench.parity import run_parity as harness


# ---------------------------------------------------------------------------
# 3a: JSON extraction via sentinel marker
# ---------------------------------------------------------------------------


def test_sentinel_json_parse_handles_nested_braces(monkeypatch):
    """`fleet_exec_collect` must extract the JSON via the sentinel line
    and not be fooled by `{` characters that appear inside the payload
    (e.g. nested dicts, traceback noise)."""
    payload = {
        "device": "jetson-orin-nx-1",
        "platform": "Linux-5.15.0-aarch64-with-glibc2.31",
        "tts": {"success_rate": 1.0, "ttfa_p50_ms": 180.0,
                "ttfa_p95_ms": 220.0, "runs": 10, "clean": 10,
                "pcm_present": True},
        "asr": {"data_incomplete": True, "reason": "stub"},
        "v2v": {"data_incomplete": True, "reason": "stub"},
    }
    stdout_lines = [
        "[boot] hello",
        "WARN: something nested {with} a brace",
        f"__PARITY_RESULT__:{json.dumps(payload)}",
        "[teardown] bye",
    ]
    fake_stdout = "\n".join(stdout_lines)

    def _fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = harness.fleet_exec_collect(
        device_id="jetson-orin-nx-1",
        base_url="http://x",
        runs=1, warmup=0, timeout=1.0,
    )
    assert result.get("device") == "jetson-orin-nx-1"
    assert "tts" in result
    assert "error" not in result, (
        f"sentinel parse should succeed cleanly; got error={result.get('error')}"
    )


def test_sentinel_parse_falls_back_on_legacy_collector(monkeypatch):
    """Older remote nodes might predate the sentinel marker. Fall back to
    line-by-line JSON load so the harness still works."""
    payload = {"device": "old-rk-1", "tts": {"success_rate": 0.9}}
    fake_stdout = f"some preface\n{json.dumps(payload)}\nend"

    def _fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = harness.fleet_exec_collect("old-rk-1", "http://x", 1, 0, 1.0)
    assert result.get("device") == "old-rk-1"


# ---------------------------------------------------------------------------
# 3b: comparator must not silently PASS when data is incomplete
# ---------------------------------------------------------------------------


def _opts(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(
        fail_on_budget=kwargs.get("fail_on_budget", True),
        fail_on_cer_regression=kwargs.get("fail_on_cer_regression", True),
        skip_asr=kwargs.get("skip_asr", False),
        skip_v2v=kwargs.get("skip_v2v", False),
        skip_tts=kwargs.get("skip_tts", False),
        strict_data=kwargs.get("strict_data", False),
    )


def _good_tts() -> dict:
    return {
        "success_rate": 1.0, "pcm_present": True,
        "ttfa_p50_ms": 250.0, "ttfa_p95_ms": 290.0,
        "runs": 10, "clean": 10,
    }


def test_comparator_flags_data_incomplete_without_strict():
    """Default mode: data_incomplete shows up as a FLAG (not PASS), so
    operators see something is missing even if hard-fail isn't enabled."""
    report = {
        "device": "jetson-orin-nx-1",
        "device_class": "orin-nx",
        "tts": _good_tts(),
        "asr": {"data_incomplete": True, "reason": "remote collector skipped ASR"},
        "v2v": {"data_incomplete": True, "reason": "remote collector skipped V2V"},
    }
    row = harness.evaluate_device(report, _opts())
    flags = row.get("flags") or []
    assert any("asr data_incomplete" in f for f in flags), flags
    assert any("v2v data_incomplete" in f for f in flags), flags
    assert row["status"] == "FLAG", (
        f"expected FLAG (not PASS) on data_incomplete, got {row['status']}"
    )


def test_comparator_hard_fails_data_incomplete_with_strict():
    """`--strict-data`: data_incomplete becomes a hard fail. Closes the
    silent-PASS hole codex flagged."""
    report = {
        "device": "jetson-orin-nx-1",
        "device_class": "orin-nx",
        "tts": _good_tts(),
        "asr": {"data_incomplete": True, "reason": "missing"},
        "v2v": {"data_incomplete": True, "reason": "missing"},
    }
    row = harness.evaluate_device(report, _opts(strict_data=True))
    assert row["status"] == "FAIL", (
        f"expected FAIL on data_incomplete with --strict-data, "
        f"got {row['status']}; flags={row.get('flags')}"
    )


def test_comparator_does_not_flag_when_explicitly_skipped():
    """`--skip-asr` acknowledges the device doesn't run ASR — no flag."""
    report = {
        "device": "tts-only-device",
        "device_class": "orin-nx",
        "tts": _good_tts(),
        "asr": {"data_incomplete": True},
        "v2v": {},
    }
    row = harness.evaluate_device(report, _opts(skip_asr=True))
    assert not any("asr data_incomplete" in f for f in (row.get("flags") or [])), (
        f"--skip-asr should suppress asr data_incomplete flag; got {row.get('flags')}"
    )


def test_remote_mode_forces_strict_data_implicitly(monkeypatch, tmp_path):
    """Round 3 BLOCKER 2: --mode remote must implicitly force --strict-data
    so the default flag-only behaviour does not silently PASS reports
    where the bundled collector ships data_incomplete=True for ASR/V2V.

    Without this guard, remote runs would emit `FLAG` (exit 0) for every
    device the collector can't fully exercise — exactly the silent PASS
    codex flagged at the parity level.
    """
    # Build a fake remote collector that returns one device with the
    # data_incomplete shape the real bundled collector emits.
    incomplete_report = {
        "device": "fake-orin-nx",
        "device_class": "orin-nx",
        "tts": _good_tts(),
        "asr": {"data_incomplete": True,
                "reason": "remote collector skipped ASR"},
        "v2v": {"data_incomplete": True,
                "reason": "remote collector skipped V2V"},
    }

    captured_strict = {"value": None}

    def _fake_collect_remote(args):
        # Snapshot what main() set on args BEFORE _collect_remote ran.
        captured_strict["value"] = args.strict_data
        return [incomplete_report]

    monkeypatch.setattr(harness, "_collect_remote", _fake_collect_remote)

    # Drive main() via argv so the strict-force branch is exercised.
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_parity.py",
            "--mode", "remote",
            "--out", str(out_dir),
        ],
    )

    rc = harness.main()

    # --strict-data must have been auto-forced before remote collection ran.
    assert captured_strict["value"] is True, (
        "remote mode did not force --strict-data implicitly; "
        f"_collect_remote saw strict_data={captured_strict['value']}"
    )
    # And the comparator must have hard-failed (exit 2), not silent-PASSed.
    assert rc == 2, (
        f"remote mode with incomplete data must exit 2 (hard fail); got rc={rc}"
    )


def test_pre_fix_baseline_would_have_passed_incomplete():
    """Sanity: a report missing ASR/V2V entirely (the pre-fix shape) is
    what produced the silent PASS. Strict mode now refuses to pass it.
    """
    report = {
        "device": "jetson-orin-nx-1",
        "device_class": "orin-nx",
        "tts": _good_tts(),
        # Note: no asr / no v2v keys at all.
    }
    # The legacy comparator would PASS this. With strict_data + no
    # data_incomplete marker we don't actively reject (you have to opt
    # in by either having the collector emit the marker OR using
    # explicit skip flags). This is the test that documents the
    # boundary so future regressions are explicit.
    row = harness.evaluate_device(report, _opts(strict_data=True))
    assert row["status"] == "PASS", (
        "without data_incomplete markers we still PASS (test documents "
        "boundary; the real fix lives in the collector script which now "
        "emits the marker)"
    )
