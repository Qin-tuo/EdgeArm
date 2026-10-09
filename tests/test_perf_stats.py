from __future__ import annotations

import json
from pathlib import Path

from bench.perf.stats import render_markdown, save_results, summarize


def test_summarize_includes_error_rate_and_drops_errors():
    records = [
        {"label": "warmup", "category": "short", "error_rate": 0.9},
        {"label": "steady", "category": "short", "error_rate": 0.2},
        {"label": "steady", "category": "short", "error_rate": 0.4},
        {"label": "steady", "category": "short", "error": "timeout"},
    ]

    summary = summarize(records, metrics=("error_rate",))

    assert summary["short"]["n"] == 2
    assert summary["short"]["error_rate"]["mean"] == 0.3
    assert summary["short"]["error_rate"]["n"] == 2


def test_render_markdown_reports_errors_and_percent_error_rate():
    raw = [
        {
            "label": "steady",
            "id": "en_short_01",
            "lang": "en",
            "category": "short",
            "error_rate": 0.198,
            "total_latency_ms": 820.0,
            "asr_text": "hello",
        },
        {
            "label": "steady",
            "id": "zh_short_01",
            "lang": "zh",
            "category": "short",
            "error": "session rejected",
        },
    ]
    summary = {
        "short": {
            "n": 1,
            "error_rate": {
                "mean": 0.198,
                "p50": 0.198,
                "p95": 0.198,
                "min": 0.198,
                "max": 0.198,
                "n": 1,
            },
        }
    }

    md = render_markdown("v2v_stream_remote", summary, raw, None, {"eos": "client"})

    assert "- Errors: **1**" in md
    assert "| CER/WER | 19.80% | 19.80% | 19.80% | 19.80% | 19.80% | 1 |" in md
    assert "| en_short_01 | en | short | 820 | 19.8% | hello |" in md
    assert "## Errors (first 20)" in md
    assert "| zh_short_01 | zh | short | session rejected |" in md


def test_save_results_uses_unique_filenames(tmp_path: Path):
    raw = [{"label": "steady", "category": "short", "rtf": 0.2}]
    summary = summarize(raw, metrics=("rtf",))

    first_json, first_md = save_results(tmp_path, "v2v_stream_remote", raw, summary, None, {})
    second_json, second_md = save_results(tmp_path, "v2v_stream_remote", raw, summary, None, {})

    assert first_json != second_json
    assert first_md != second_md
    for path in (first_json, first_md, second_json, second_md):
        assert path.exists()
    assert json.loads(first_json.read_text())["summary"] == summary
