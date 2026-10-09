#!/usr/bin/env python3
"""Compare /v2v/stream result JSON files across devices or RKLLM variants."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


METRICS = (
    "error_rate",
    "tail_truncation_rate",
    "total_latency_ms",
    "endpoint_latency_ms",
    "asr_finalize_ms",
    "tfd_ms",
)

_PUNCT_TABLE = str.maketrans(
    "",
    "",
    "，。！？、；：\"\"''（）《》【】「」『』"
    + "·～—-…,.!?:;\"'()<>[]{}/",
)


@dataclass(frozen=True)
class CompareRow:
    label: str
    group: str
    n: int
    errors: int
    error_rate_mean: float | None
    tail_truncation_rate_mean: float | None
    total_latency_ms_mean: float | None
    endpoint_latency_ms_mean: float | None
    asr_finalize_ms_mean: float | None
    tfd_ms_mean: float | None
    path: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    baseline: CompareRow
    candidate: CompareRow
    max_error_rate: float
    max_tail_truncation_rate: float | None
    max_total_latency_ms: float
    max_tfd_ms: float | None
    reasons: tuple[str, ...]


def _mean_metric(summary_group: dict, metric: str) -> float | None:
    value = summary_group.get(metric)
    if not isinstance(value, dict):
        return None
    mean = value.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "corpus" / "manifest.json"


def _load_reference_map(manifest_path: Path | None = None) -> dict[str, dict]:
    path = manifest_path or _default_manifest_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    refs = {}
    for entry in payload.get("files", []):
        sample_id = entry.get("id")
        if sample_id:
            refs[sample_id] = entry
    return refs


def _metric_reference(ref_entry: dict) -> str:
    return ref_entry.get("eval_transcript") or ref_entry.get("transcript", "")


def _normalize_for_tail(text: str, lang: str) -> str:
    text = (text or "").translate(_PUNCT_TABLE).lower().strip()
    text = " ".join(text.split())
    if lang == "zh":
        text = text.replace(" ", "")
    return text


def _tail_truncation(reference: str, hypothesis: str, lang: str) -> tuple[bool, int]:
    ref = _normalize_for_tail(reference, lang)
    hyp = _normalize_for_tail(hypothesis, lang)
    if not ref or not hyp:
        return False, 0
    if lang == "zh":
        ref_units = list(ref)
        hyp_units = list(hyp)
        min_missing = 4
        min_hyp = 4
    else:
        ref_units = ref.split()
        hyp_units = hyp.split()
        min_missing = 2
        min_hyp = 2
    if len(hyp_units) < min_hyp or len(hyp_units) >= len(ref_units):
        return False, 0
    if ref_units[:len(hyp_units)] != hyp_units:
        return False, 0
    missing = len(ref_units) - len(hyp_units)
    return missing >= min_missing, missing


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def _error_rate(reference: str, hypothesis: str, lang: str) -> float | None:
    ref = _normalize_for_tail(reference, lang)
    hyp = _normalize_for_tail(hypothesis, lang)
    if not ref or not hyp:
        return None
    ref_units = list(ref) if lang == "zh" else ref.split()
    hyp_units = list(hyp) if lang == "zh" else hyp.split()
    if not ref_units:
        return None
    return _levenshtein(ref_units, hyp_units) / len(ref_units)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_for_raw(row: dict) -> str | None:
    category = row.get("category")
    lang = row.get("lang")
    if category and lang:
        return f"{category}/{lang}"
    return None


def _raw_derived_metrics(raw: list[dict]) -> dict[str, dict[str, float]]:
    refs = _load_reference_map()
    by_group: dict[str, dict[str, list[float]]] = {}
    for row in raw:
        if "error" in row:
            continue
        group = _group_for_raw(row)
        if not group:
            continue
        bucket = by_group.setdefault(group, {"error_rate": [], "tail_truncation_rate": []})
        if isinstance(row.get("error_rate"), (int, float)):
            bucket["error_rate"].append(float(row["error_rate"]))
        sample_id = row.get("id")
        ref_entry = refs.get(sample_id, {})
        reference = _metric_reference(ref_entry)
        hypothesis = row.get("text") or row.get("asr_text") or ""
        lang = row.get("lang") or ref_entry.get("lang") or ""
        if not isinstance(row.get("error_rate"), (int, float)) and reference and hypothesis:
            computed_error_rate = _error_rate(reference, hypothesis, lang)
            if computed_error_rate is not None:
                bucket["error_rate"].append(computed_error_rate)
        if isinstance(row.get("tail_truncation_rate"), (int, float)):
            tail_value = float(row["tail_truncation_rate"])
        elif isinstance(row.get("tail_truncated"), bool):
            tail_value = 1.0 if row["tail_truncated"] else 0.0
        elif reference and hypothesis:
            tail_value = 1.0 if _tail_truncation(reference, hypothesis, lang)[0] else 0.0
        else:
            tail_value = 0.0
        bucket["tail_truncation_rate"].append(tail_value)
    out: dict[str, dict[str, float]] = {}
    for group, metrics in by_group.items():
        out[group] = {}
        for name, values in metrics.items():
            value = _mean(values)
            if value is not None:
                out[group][name] = value
    return out


def _load_result(label: str, path: Path) -> list[CompareRow]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary") or {}
    raw = payload.get("raw") or []
    derived = _raw_derived_metrics(raw)
    error_count = sum(1 for row in raw if "error" in row)
    rows: list[CompareRow] = []
    for group, stats in sorted(summary.items()):
        if not isinstance(stats, dict):
            continue
        rows.append(
            CompareRow(
                label=label,
                group=group,
                n=int(stats.get("n") or 0),
                errors=error_count,
                error_rate_mean=(
                    _mean_metric(stats, "error_rate")
                    if _mean_metric(stats, "error_rate") is not None
                    else derived.get(group, {}).get("error_rate")
                ),
                tail_truncation_rate_mean=(
                    _mean_metric(stats, "tail_truncation_rate")
                    if _mean_metric(stats, "tail_truncation_rate") is not None
                    else derived.get(group, {}).get("tail_truncation_rate")
                ),
                total_latency_ms_mean=_mean_metric(stats, "total_latency_ms"),
                endpoint_latency_ms_mean=_mean_metric(stats, "endpoint_latency_ms"),
                asr_finalize_ms_mean=_mean_metric(stats, "asr_finalize_ms"),
                tfd_ms_mean=_mean_metric(stats, "tfd_ms"),
                path=str(path),
            )
        )
    return rows


def _parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty label in {value!r}")
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def load_rows(result_args: Iterable[str]) -> list[CompareRow]:
    rows: list[CompareRow] = []
    for value in result_args:
        label, path = _parse_result_arg(value)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(_load_result(label, path))
    return rows


def _fmt_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.0f}"


def _fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.1f}%"


def render_markdown(rows: list[CompareRow]) -> str:
    lines = [
        "| Group | Label | N | ErrRows | CER/WER | Tail trunc | Total ms | Endpoint ms | ASR finalize ms | TFD ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            r.group,
            float("inf") if r.error_rate_mean is None else r.error_rate_mean,
            float("inf") if r.total_latency_ms_mean is None else r.total_latency_ms_mean,
            r.label,
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.group,
                    row.label,
                    str(row.n),
                    str(row.errors),
                    _fmt_pct(row.error_rate_mean),
                    _fmt_pct(row.tail_truncation_rate_mean),
                    _fmt_ms(row.total_latency_ms_mean),
                    _fmt_ms(row.endpoint_latency_ms_mean),
                    _fmt_ms(row.asr_finalize_ms_mean),
                    _fmt_ms(row.tfd_ms_mean),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_best(rows: list[CompareRow]) -> str:
    by_group: dict[str, list[CompareRow]] = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)
    lines: list[str] = []
    for group, items in sorted(by_group.items()):
        ranked = sorted(
            items,
            key=lambda r: (
                float("inf") if r.error_rate_mean is None else r.error_rate_mean,
                float("inf") if r.total_latency_ms_mean is None else r.total_latency_ms_mean,
                r.errors,
            ),
        )
        best = ranked[0]
        lines.append(
            f"- {group}: best by quality is `{best.label}` "
            f"({_fmt_pct(best.error_rate_mean)}, total {_fmt_ms(best.total_latency_ms_mean)} ms)"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _single_row(rows: list[CompareRow], label: str, group: str) -> CompareRow:
    matches = [row for row in rows if row.label == label and row.group == group]
    if not matches:
        groups = sorted({row.group for row in rows if row.label == label})
        if groups:
            raise ValueError(
                f"no row for label={label!r} group={group!r}; available groups: {groups}"
            )
        labels = sorted({row.label for row in rows})
        raise ValueError(f"no rows for label={label!r}; available labels: {labels}")
    if len(matches) > 1:
        raise ValueError(f"multiple rows for label={label!r} group={group!r}")
    return matches[0]


def evaluate_gate(
    rows: list[CompareRow],
    *,
    baseline_label: str,
    candidate_label: str,
    group: str,
    max_error_rate_delta: float = 0.0,
    max_total_latency_ratio: float = 1.0,
    min_total_latency_improvement_ms: float = 0.0,
    max_tfd_ratio: float | None = None,
    max_tfd_ms: float | None = None,
    max_tail_truncation_rate: float | None = None,
) -> GateResult:
    baseline = _single_row(rows, baseline_label, group)
    candidate = _single_row(rows, candidate_label, group)
    missing = [
        name
        for name, value in (
            ("baseline error_rate", baseline.error_rate_mean),
            ("candidate error_rate", candidate.error_rate_mean),
            ("baseline total_latency_ms", baseline.total_latency_ms_mean),
            ("candidate total_latency_ms", candidate.total_latency_ms_mean),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"missing metrics for gate: {', '.join(missing)}")

    assert baseline.error_rate_mean is not None
    assert candidate.error_rate_mean is not None
    assert baseline.total_latency_ms_mean is not None
    assert candidate.total_latency_ms_mean is not None

    max_error_rate = baseline.error_rate_mean + max_error_rate_delta
    max_total_latency_ms = baseline.total_latency_ms_mean * max_total_latency_ratio
    reasons: list[str] = []
    if candidate.errors > baseline.errors:
        reasons.append(
            "error_rows "
            f"{candidate.errors} exceeds baseline {baseline.errors}"
        )
    if candidate.error_rate_mean > max_error_rate:
        reasons.append(
            "error_rate "
            f"{_fmt_pct(candidate.error_rate_mean)} exceeds allowed "
            f"{_fmt_pct(max_error_rate)}"
        )
    if max_tail_truncation_rate is not None:
        if candidate.tail_truncation_rate_mean is None:
            reasons.append("candidate tail_truncation_rate is missing")
        elif candidate.tail_truncation_rate_mean > max_tail_truncation_rate:
            reasons.append(
                "tail_truncation_rate "
                f"{_fmt_pct(candidate.tail_truncation_rate_mean)} exceeds allowed "
                f"{_fmt_pct(max_tail_truncation_rate)}"
            )
    if candidate.total_latency_ms_mean > max_total_latency_ms:
        reasons.append(
            "total_latency "
            f"{_fmt_ms(candidate.total_latency_ms_mean)} ms exceeds allowed "
            f"{_fmt_ms(max_total_latency_ms)} ms"
        )
    improvement = baseline.total_latency_ms_mean - candidate.total_latency_ms_mean
    if improvement < min_total_latency_improvement_ms:
        reasons.append(
            "total_latency improvement "
            f"{_fmt_ms(improvement)} ms is below required "
            f"{_fmt_ms(min_total_latency_improvement_ms)} ms"
        )
    allowed_tfd_ms = None
    if max_tfd_ratio is not None or max_tfd_ms is not None:
        if candidate.tfd_ms_mean is None:
            reasons.append("candidate tfd_ms is missing")
        if baseline.tfd_ms_mean is None and max_tfd_ratio is not None:
            reasons.append("baseline tfd_ms is missing")
        if max_tfd_ratio is not None and baseline.tfd_ms_mean is not None:
            allowed_tfd_ms = baseline.tfd_ms_mean * max_tfd_ratio
        if max_tfd_ms is not None:
            allowed_tfd_ms = (
                max_tfd_ms
                if allowed_tfd_ms is None
                else min(allowed_tfd_ms, max_tfd_ms)
            )
        if (
            allowed_tfd_ms is not None
            and candidate.tfd_ms_mean is not None
            and candidate.tfd_ms_mean > allowed_tfd_ms
        ):
            reasons.append(
                "tfd_ms "
                f"{_fmt_ms(candidate.tfd_ms_mean)} ms exceeds allowed "
                f"{_fmt_ms(allowed_tfd_ms)} ms"
            )
    return GateResult(
        passed=not reasons,
        baseline=baseline,
        candidate=candidate,
        max_error_rate=max_error_rate,
        max_tail_truncation_rate=max_tail_truncation_rate,
        max_total_latency_ms=max_total_latency_ms,
        max_tfd_ms=allowed_tfd_ms,
        reasons=tuple(reasons),
    )


def render_gate(result: GateResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"{status}: `{result.candidate.label}` vs `{result.baseline.label}` "
        f"on `{result.candidate.group}`",
        (
            "- error_rows: "
            f"{result.candidate.errors} vs baseline {result.baseline.errors}"
        ),
        (
            "- error_rate: "
            f"{_fmt_pct(result.candidate.error_rate_mean)} vs baseline "
            f"{_fmt_pct(result.baseline.error_rate_mean)} "
            f"(allowed <= {_fmt_pct(result.max_error_rate)})"
        ),
        (
            "- tail_truncation_rate: "
            f"{_fmt_pct(result.candidate.tail_truncation_rate_mean)}"
            + (
                " (no limit)"
                if result.max_tail_truncation_rate is None
                else f" (allowed <= {_fmt_pct(result.max_tail_truncation_rate)})"
            )
        ),
        (
            "- total_latency_ms: "
            f"{_fmt_ms(result.candidate.total_latency_ms_mean)} vs baseline "
            f"{_fmt_ms(result.baseline.total_latency_ms_mean)} "
            f"(allowed <= {_fmt_ms(result.max_total_latency_ms)} ms)"
        ),
    ]
    if result.max_tfd_ms is not None:
        lines.append(
            "- tfd_ms: "
            f"{_fmt_ms(result.candidate.tfd_ms_mean)} vs baseline "
            f"{_fmt_ms(result.baseline.tfd_ms_mean)} "
            f"(allowed <= {_fmt_ms(result.max_tfd_ms)} ms)"
        )
    if result.reasons:
        lines.append("- reasons: " + "; ".join(result.reasons))
    return "\n".join(lines) + "\n"


def render_gates(results: list[GateResult]) -> str:
    lines: list[str] = []
    for result in results:
        if lines:
            lines.append("")
        lines.append(render_gate(result).rstrip())
    if results:
        passed = sum(1 for result in results if result.passed)
        lines.append("")
        lines.append(f"Gate summary: {passed}/{len(results)} passed")
    return "\n".join(lines) + ("\n" if lines else "")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare /v2v/stream benchmark result JSON files"
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Result JSON path, optionally label=path",
    )
    parser.add_argument("--json", action="store_true", help="Emit rows as JSON")
    parser.add_argument("--best", action="store_true", help="Append best-by-quality summary")
    parser.add_argument("--gate", action="store_true", help="Evaluate candidate against baseline")
    parser.add_argument("--baseline-label", help="Baseline label for --gate")
    parser.add_argument("--candidate-label", help="Candidate label for --gate")
    parser.add_argument("--group", help="Summary group for --gate, for example short/en")
    parser.add_argument(
        "--groups",
        help="Comma-separated summary groups for --gate, for example short/en,short/zh",
    )
    parser.add_argument(
        "--max-error-rate-delta",
        type=float,
        default=0.0,
        help="Allowed absolute CER/WER regression for --gate, default 0.0",
    )
    parser.add_argument(
        "--max-total-latency-ratio",
        type=float,
        default=1.0,
        help="Allowed candidate/baseline total latency ratio for --gate, default 1.0",
    )
    parser.add_argument(
        "--min-total-latency-improvement-ms",
        type=float,
        default=0.0,
        help="Required total latency improvement for --gate, default 0",
    )
    parser.add_argument(
        "--max-tfd-ratio",
        type=float,
        help="Optional allowed candidate/baseline TFD ratio for streaming gate",
    )
    parser.add_argument(
        "--max-tfd-ms",
        type=float,
        help="Optional absolute allowed TFD in milliseconds for streaming gate",
    )
    parser.add_argument(
        "--max-tail-truncation-rate",
        type=float,
        help="Optional allowed mean tail truncation rate for endpoint-policy gates",
    )
    parser.add_argument(
        "--forbid-tail-truncation",
        action="store_true",
        help="Shortcut for --max-tail-truncation-rate 0.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate_groups = (
        [g.strip() for g in args.groups.split(",") if g.strip()]
        if args.groups
        else ([args.group] if args.group else [])
    )
    if args.gate and not (args.baseline_label and args.candidate_label and gate_groups):
        print(
            "[ERROR] --gate requires --baseline-label, --candidate-label, and --group/--groups",
            file=sys.stderr,
        )
        return 2
    try:
        rows = load_rows(args.results)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False))
    else:
        print(render_markdown(rows), end="")
        if args.best:
            print()
            print(render_best(rows), end="")
    if args.gate:
        max_tail_truncation_rate = args.max_tail_truncation_rate
        if args.forbid_tail_truncation:
            max_tail_truncation_rate = 0.0
        try:
            gates = [
                evaluate_gate(
                    rows,
                    baseline_label=args.baseline_label,
                    candidate_label=args.candidate_label,
                    group=group,
                    max_error_rate_delta=args.max_error_rate_delta,
                    max_total_latency_ratio=args.max_total_latency_ratio,
                    min_total_latency_improvement_ms=args.min_total_latency_improvement_ms,
                    max_tfd_ratio=args.max_tfd_ratio,
                    max_tfd_ms=args.max_tfd_ms,
                    max_tail_truncation_rate=max_tail_truncation_rate,
                )
                for group in gate_groups
            ]
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        if not args.json:
            if not args.best:
                print()
            print(render_gates(gates), end="")
        return 0 if all(gate.passed for gate in gates) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
