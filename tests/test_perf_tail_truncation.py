from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runners():
    path = Path(__file__).resolve().parents[1] / "bench" / "perf" / "runners.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("perf_runners", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detect_tail_truncation_for_chinese_prefix_drop():
    runners = _load_runners()

    truncated, missing = runners.detect_tail_truncation(
        "传统上，王位继承人在完成学业后会直接入伍。",
        "传统上，王位继承人。",
        "zh",
    )

    assert truncated is True
    assert missing >= 4


def test_detect_tail_truncation_does_not_flag_substitution():
    runners = _load_runners()

    truncated, missing = runners.detect_tail_truncation(
        "传统上，王位继承人在完成学业后会直接入伍。",
        "传统上，王位继承后会直接入伍。",
        "zh",
    )

    assert truncated is False
    assert missing == 0


def test_coverage_rate_flags_non_prefix_early_stop():
    runners = _load_runners()

    coverage = runners.compute_coverage_rate(
        "而且太平洋海啸预警中心（AlsothePacificTsunamiWarningCenter）也表示并未发现海啸迹象。",
        "这还是迹象。",
        "zh",
    )

    assert coverage < 0.5


def test_coverage_rate_tolerates_substitution_better_than_early_stop():
    runners = _load_runners()

    coverage = runners.compute_coverage_rate(
        "传统上，王位继承人在完成学业后会直接入伍。",
        "传统上，往往继承人在完成学业后会直接入。",
        "zh",
    )

    assert coverage >= 0.75


def test_metric_transcript_keeps_strict_error_auditable():
    runners = _load_runners()
    entry = {
        "transcript": "而且太平洋海啸预警中心（AlsothePacificTsunamiWarningCenter）也表示并未发现海啸迹象。",
        "eval_transcript": "而且太平洋海啸预警中心也表示并未发现海啸迹象。",
    }
    hyp = "而且太平洋海啸预警中心也表示，并未发现海啸迹象。"

    strict = runners.compute_error_rate(runners.strict_transcript(entry), hyp, "zh")
    headline = runners.compute_error_rate(runners.metric_transcript(entry), hyp, "zh")

    assert strict > 0.5
    assert headline < 0.05
