from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "bench"
        / "perf"
        / "compare_v2v_stream_results.py"
    )
    spec = importlib.util.spec_from_file_location("compare_v2v_stream_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(path: Path, group: str, error_rate: float, total_ms: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    group: {
                        "n": 3,
                        "error_rate": {"mean": error_rate},
                        "total_latency_ms": {"mean": total_ms},
                        "endpoint_latency_ms": {"mean": 1.0},
                        "asr_finalize_ms": {"mean": total_ms - 1.0},
                        "tfd_ms": {"mean": 500.0},
                    }
                },
                "raw": [{"label": "steady"}, {"label": "steady", "error": "x"}],
            }
        )
    )
    return path


def _multi_result(path: Path, groups: dict[str, tuple[float, float]]) -> Path:
    payload = {"summary": {}, "raw": [{"label": "steady"}]}
    for group, (error_rate, total_ms) in groups.items():
        payload["summary"][group] = {
            "n": 3,
            "error_rate": {"mean": error_rate},
            "total_latency_ms": {"mean": total_ms},
            "endpoint_latency_ms": {"mean": 1.0},
            "asr_finalize_ms": {"mean": total_ms - 1.0},
            "tfd_ms": {"mean": 500.0},
        }
    path.write_text(json.dumps(payload))
    return path


def _v2v_result_with_raw(
    path: Path,
    *,
    group: str,
    error_rate: float,
    total_ms: float,
    raw_text: str,
) -> Path:
    category, lang = group.split("/")
    payload = {
        "summary": {
            group: {
                "n": 1,
                "error_rate": {"mean": error_rate},
                "total_latency_ms": {"mean": total_ms},
                "endpoint_latency_ms": {"mean": 1.0},
                "asr_finalize_ms": {"mean": total_ms - 1.0},
                "tfd_ms": {"mean": 500.0},
            }
        },
        "raw": [
            {
                "label": "steady",
                "id": "zh_short_03",
                "lang": lang,
                "category": category,
                "text": raw_text,
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def _v2v_result_with_raw_no_summary_error(
    path: Path,
    *,
    group: str,
    raw_text: str,
) -> Path:
    category, lang = group.split("/")
    payload = {
        "summary": {
            group: {
                "n": 1,
                "total_latency_ms": {"mean": 500.0},
                "endpoint_latency_ms": {"mean": 1.0},
                "asr_finalize_ms": {"mean": 499.0},
                "tfd_ms": {"mean": 300.0},
            }
        },
        "raw": [
            {
                "label": "steady",
                "id": "zh_short_02",
                "lang": lang,
                "category": category,
                "text": raw_text,
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def test_load_rows_from_labeled_result(tmp_path: Path):
    module = _load_module()
    path = _result(tmp_path / "rk.json", "short/en", 0.2, 800)

    rows = module.load_rows([f"rk3576={path}"])

    assert len(rows) == 1
    assert rows[0].label == "rk3576"
    assert rows[0].group == "short/en"
    assert rows[0].errors == 1
    assert rows[0].error_rate_mean == 0.2


def test_raw_backfill_uses_eval_transcript_when_available(tmp_path: Path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_load_reference_map",
        lambda: {
            "zh_short_02": {
                "lang": "zh",
                "transcript": "而且太平洋海啸预警中心（AlsothePacificTsunamiWarningCenter）也表示并未发现海啸迹象。",
                "eval_transcript": "而且太平洋海啸预警中心也表示并未发现海啸迹象。",
            }
        },
    )
    path = _v2v_result_with_raw_no_summary_error(
        tmp_path / "raw.json",
        group="short/zh",
        raw_text="而且太平洋海啸预警中心也表示，并未发现海啸迹象。",
    )

    rows = module.load_rows([f"rk={path}"])

    assert rows[0].error_rate_mean is not None
    assert rows[0].error_rate_mean < 0.05


def test_render_markdown_sorts_by_quality_then_latency(tmp_path: Path):
    module = _load_module()
    slow = _result(tmp_path / "slow.json", "short/en", 0.1, 900)
    fast = _result(tmp_path / "fast.json", "short/en", 0.1, 300)
    worse = _result(tmp_path / "worse.json", "short/en", 0.2, 100)

    rows = module.load_rows([f"slow={slow}", f"fast={fast}", f"worse={worse}"])
    md = module.render_markdown(rows)

    assert md.index("fast") < md.index("slow")
    assert md.index("slow") < md.index("worse")


def test_render_best_picks_lowest_error_first(tmp_path: Path):
    module = _load_module()
    low_latency = _result(tmp_path / "low_latency.json", "short/en", 0.2, 100)
    best_quality = _result(tmp_path / "best_quality.json", "short/en", 0.1, 900)

    rows = module.load_rows(
        [f"low_latency={low_latency}", f"best_quality={best_quality}"]
    )
    best = module.render_best(rows)

    assert "`best_quality`" in best


def test_gate_fails_low_latency_quality_regression(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/en", 0.1, 900)
    candidate = _result(tmp_path / "candidate.json", "short/en", 0.2, 300)
    rows = module.load_rows([f"fp16={baseline}", f"w8a8={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="fp16",
        candidate_label="w8a8",
        group="short/en",
    )

    assert result.passed is False
    assert "error_rate" in result.reasons[0]


def test_gate_passes_same_quality_lower_latency(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/zh", 0.22, 900)
    candidate = _result(tmp_path / "candidate.json", "short/zh", 0.22, 500)
    rows = module.load_rows([f"fp16={baseline}", f"w8a8={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="fp16",
        candidate_label="w8a8",
        group="short/zh",
    )

    assert result.passed is True
    assert result.reasons == ()


def test_gate_fails_when_candidate_has_more_error_rows(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/zh", 0.3, 900)
    candidate = _result(tmp_path / "candidate.json", "short/zh", 0.1, 500)
    payload = json.loads(candidate.read_text())
    payload["raw"].append({
        "label": "steady",
        "id": "zh_short_02",
        "lang": "zh",
        "category": "short",
        "error": "timeout waiting for asr_final",
    })
    candidate.write_text(json.dumps(payload))
    rows = module.load_rows([f"base={baseline}", f"cand={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="base",
        candidate_label="cand",
        group="short/zh",
    )

    assert result.passed is False
    assert any(reason.startswith("error_rows") for reason in result.reasons)


def test_gate_reports_missing_group(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/en", 0.1, 900)
    candidate = _result(tmp_path / "candidate.json", "short/en", 0.1, 500)
    rows = module.load_rows([f"fp16={baseline}", f"w8a8={candidate}"])

    try:
        module.evaluate_gate(
            rows,
            baseline_label="fp16",
            candidate_label="w8a8",
            group="short/zh",
        )
    except ValueError as exc:
        assert "available groups" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_gate_can_fail_tfd_regression(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/en", 0.1, 500)
    candidate = _result(tmp_path / "candidate.json", "short/en", 0.1, 400)
    payload = json.loads(candidate.read_text())
    payload["summary"]["short/en"]["tfd_ms"]["mean"] = 1800.0
    candidate.write_text(json.dumps(payload))
    rows = module.load_rows([f"fp16={baseline}", f"w8a8={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="fp16",
        candidate_label="w8a8",
        group="short/en",
        max_tfd_ratio=2.0,
    )

    assert result.passed is False
    assert any(reason.startswith("tfd_ms") for reason in result.reasons)


def test_gate_can_forbid_tail_truncation_from_raw_manifest(tmp_path: Path):
    module = _load_module()
    baseline = _v2v_result_with_raw(
        tmp_path / "baseline.json",
        group="short/zh",
        error_rate=0.1,
        total_ms=1000,
        raw_text="传统上，王位继承人在完成学业后会直接入。",
    )
    candidate = _v2v_result_with_raw(
        tmp_path / "candidate.json",
        group="short/zh",
        error_rate=0.1,
        total_ms=500,
        raw_text="传统上，王位继承人。",
    )
    rows = module.load_rows([f"baseline={baseline}", f"vad={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="baseline",
        candidate_label="vad",
        group="short/zh",
        max_tail_truncation_rate=0.0,
    )

    assert result.passed is False
    assert any(reason.startswith("tail_truncation_rate") for reason in result.reasons)


def test_gate_can_cap_absolute_tfd(tmp_path: Path):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/en", 0.1, 500)
    candidate = _result(tmp_path / "candidate.json", "short/en", 0.1, 400)
    rows = module.load_rows([f"fp16={baseline}", f"w8a8={candidate}"])

    result = module.evaluate_gate(
        rows,
        baseline_label="fp16",
        candidate_label="w8a8",
        group="short/en",
        max_tfd_ms=600.0,
    )

    assert result.passed is True
    assert result.max_tfd_ms == 600.0


def test_main_returns_failure_for_failed_gate(tmp_path: Path, capsys):
    module = _load_module()
    baseline = _result(tmp_path / "baseline.json", "short/en", 0.1, 900)
    candidate = _result(tmp_path / "candidate.json", "short/en", 0.2, 300)

    code = module.main(
        [
            "--gate",
            "--baseline-label",
            "fp16",
            "--candidate-label",
            "w8a8",
            "--group",
            "short/en",
            f"fp16={baseline}",
            f"w8a8={candidate}",
        ]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL" in out


def test_main_can_gate_multiple_groups(tmp_path: Path, capsys):
    module = _load_module()
    baseline = _multi_result(
        tmp_path / "baseline.json",
        {"short/en": (0.1, 900), "short/zh": (0.2, 900)},
    )
    candidate = _multi_result(
        tmp_path / "candidate.json",
        {"short/en": (0.1, 700), "short/zh": (0.3, 700)},
    )

    code = module.main(
        [
            "--gate",
            "--baseline-label",
            "fp16",
            "--candidate-label",
            "w8a8",
            "--groups",
            "short/en,short/zh",
            f"fp16={baseline}",
            f"w8a8={candidate}",
        ]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "PASS: `w8a8` vs `fp16` on `short/en`" in out
    assert "FAIL: `w8a8` vs `fp16` on `short/zh`" in out
    assert "Gate summary: 1/2 passed" in out
