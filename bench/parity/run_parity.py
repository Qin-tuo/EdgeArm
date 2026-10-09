#!/usr/bin/env python3
"""Multi-device e2e parity harness for seeed-local-voice.

Two modes:
  --mode mock    Load fixtures from bench/parity/fixtures/mock/*.json and
                 run the comparator. No hardware needed. Used for Mac
                 dry-runs and for testing the comparator logic itself.

  --mode remote  Discover devices via `fleet match --tags <list> --json`
                 and execute the per-device parity collector remotely
                 via `fleet exec <device> -- ...`. Per-device JSON is
                 written under bench/parity/results/<device>/<ts>.json.
                 The comparator then runs locally on the collected
                 reports.

The comparator dimensions (per spec §D3):
  - ASR CER (zh) / WER (en) regression vs baseline (>5% = flag)
  - ASR final-text completeness (missing_finals > 0 = hard fail)
  - TTS synthesis success rate (must be 1.0)
  - TTS TTFA vs device-class budget
  - V2V barge-in latency, stop intent, reconnect

Device TTFA budgets (from spec §D3 §Device budgets):
  orin-nano  : 200-500 ms
  orin-nx    : 150-300 ms
  rk3576     : 300-1500 ms
  others     : warn only; no hard budget unless --device-budget configures one

This script depends only on the Python standard library plus modules
that already exist locally (`bench/perf/runners.py` for normalization;
we duplicate normalization rules here intentionally so the script can
run on hosts without numpy / jiwer when in --mode mock). No new
pyproject deps are introduced.

Exit codes:
  0  - all devices PASS (or only warnings)
  2  - one or more hard gate failures (set by --fail-on-budget /
       --fail-on-cer-regression or by missing_finals > 0)
  3  - infrastructure failure (missing fixtures / no devices discovered)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "bench" / "parity" / "results"
DEFAULT_FIXTURES = REPO_ROOT / "bench" / "parity" / "fixtures" / "mock"
DEFAULT_BASELINE_DIR = REPO_ROOT / "bench" / "perf" / "results"

DEVICE_CLASS_BUDGETS = {
    # device_class -> (min_ttfa_ms, max_ttfa_ms)
    "orin-nano": (200, 500),
    "orin-nx": (150, 300),
    "rk3576": (300, 1500),
    "rk3588": (300, 1500),
    "rpi-hailo": (300, 1500),
}

CER_REGRESSION_PCT = 0.05  # 5% above baseline

DEVICE_CLASS_TAGS = {
    "jetson": ["jetson"],
    "rk": ["rockchip"],
    "rpi-hailo": ["hailo", "rpi"],
    "all": [],  # no tag filter
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _ts_for_path() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Mock fixture loader
# ---------------------------------------------------------------------------


def load_mock_fixtures(fixture_dir: Path) -> list[dict[str, Any]]:
    if not fixture_dir.exists():
        raise SystemExit(f"mock fixture dir not found: {fixture_dir}")
    fixtures = []
    for p in sorted(fixture_dir.glob("*.json")):
        try:
            fixtures.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            raise SystemExit(f"failed to parse {p}: {e}")
    if not fixtures:
        raise SystemExit(f"no fixtures found in {fixture_dir}")
    return fixtures


# ---------------------------------------------------------------------------
# Remote mode: fleet match + exec
# ---------------------------------------------------------------------------


def fleet_match(tags: list[str]) -> list[dict[str, Any]]:
    cmd = ["fleet", "match"]
    for t in tags:
        cmd += ["--tags", t]
    cmd.append("--json")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise SystemExit(
            "fleet CLI not found. Install per ~/.claude/CLAUDE.md fleet "
            "section or run with --mode mock."
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"fleet match failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:400]}"
        )
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as e:
        raise SystemExit(f"fleet match returned non-JSON: {e}")
    if isinstance(data, dict) and "devices" in data:
        data = data["devices"]
    if not isinstance(data, list):
        raise SystemExit(f"unexpected fleet match payload shape: {type(data)}")
    return data


def fleet_exec_collect(
    device_id: str, base_url: str, runs: int, warmup: int, timeout: float,
) -> dict[str, Any]:
    """Run a small inline Python snippet on the remote device that hits its
    local /health and a few /tts/stream + /asr endpoints, then prints
    JSON to stdout. We intentionally keep this self-contained so the
    remote does not need to checkout the parity harness — the collector
    payload is shipped with the command.
    """
    payload = _COLLECTOR_SCRIPT.replace("__BASE_URL__", base_url)
    payload = payload.replace("__RUNS__", str(runs))
    payload = payload.replace("__WARMUP__", str(warmup))
    payload = payload.replace("__TIMEOUT__", str(timeout))
    # fleet exec <device> -- <cmd>
    cmd = ["fleet", "exec", device_id, "--", "python3", "-c", payload]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout * runs * 3 + 60)
    except FileNotFoundError:
        raise SystemExit("fleet CLI not found.")
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {
            "device": device_id, "error": "fleet exec failed",
            "rc": proc.returncode,
            "stderr_tail": err[-2000:],
            "stdout_tail": out[-2000:],
        }
    # Extract the JSON object printed by the collector. Look for a
    # sentinel-prefixed line so nested `{` braces in tracebacks or
    # platform.platform() text never confuse the parser. The original
    # `rfind('{') .. rfind('}')` scheme picked up the LAST inner `{`
    # which sits inside the outer JSON, producing a partial-object
    # ValueError. Codex Week 3 BLOCKER 3a.
    sentinel = "__PARITY_RESULT__:"
    parsed = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if not line.startswith(sentinel):
            continue
        payload = line[len(sentinel):].strip()
        try:
            parsed = json.loads(payload)
            break
        except Exception:
            continue
    if parsed is None:
        # Backwards-compatible fallback for collectors that pre-date the
        # sentinel marker: try line-by-line JSON load on whole-line
        # candidates. Avoids the nested-brace pitfall of rfind('{').
        for line in reversed(out.splitlines()):
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                parsed = json.loads(line)
                break
            except Exception:
                continue
    if parsed is None:
        return {
            "device": device_id,
            "error": "no parseable JSON in collector stdout",
            "stdout_tail": out[-2000:],
            "stderr_tail": err[-2000:],
        }
    return parsed


# Minimal in-process collector that runs on the remote device. Uses only
# stdlib (no requests) so it works on a barebones python3 install. It
# checks /health, fires one TTS stream request, and dumps a JSON report.
# The full ASR / V2V matrix is left to operators since each backend has
# different streaming semantics; the harness records what /health says
# and a single TTS smoke per device.
_COLLECTOR_SCRIPT = r"""
import json, time, urllib.request, urllib.error, sys, socket, platform
BASE = "__BASE_URL__"
RUNS = int("__RUNS__")
WARMUP = int("__WARMUP__")
TIMEOUT = float("__TIMEOUT__")
out = {"device": socket.gethostname(), "platform": platform.platform(),
       "service_url": BASE, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    req = urllib.request.Request(BASE.rstrip("/") + "/health")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        out["health"] = json.loads(r.read().decode("utf-8"))
except Exception as e:
    out["health_error"] = str(e)
ttfas = []
for i in range(WARMUP + RUNS):
    t0 = time.perf_counter()
    try:
        data = json.dumps({"text": "hello"}).encode("utf-8")
        req = urllib.request.Request(BASE.rstrip("/") + "/tts/stream",
                                     data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            seen_header = False
            ttfa = None
            while True:
                chunk = r.read(4096)
                if not chunk:
                    break
                if not seen_header:
                    seen_header = True
                    if len(chunk) > 4:
                        ttfa = (time.perf_counter() - t0) * 1000
                        break
                else:
                    ttfa = (time.perf_counter() - t0) * 1000
                    break
        if i >= WARMUP and ttfa is not None:
            ttfas.append(ttfa)
    except Exception as e:
        if i >= WARMUP:
            ttfas.append(None)
        out.setdefault("tts_errors", []).append(str(e))
clean = [t for t in ttfas if t is not None]
clean.sort()
def pct(arr, p):
    if not arr: return None
    k = (len(arr) - 1) * p
    f = int(k); c = min(f + 1, len(arr) - 1)
    if f == c: return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)
out["tts"] = {
    "success_rate": (len(clean) / len(ttfas)) if ttfas else 0.0,
    "ttfa_p50_ms": pct(clean, 0.5),
    "ttfa_p95_ms": pct(clean, 0.95),
    "runs": len(ttfas), "clean": len(clean),
    "pcm_present": len(clean) > 0,
}
# Codex Week 3 BLOCKER 3b: remote collector intentionally collects only
# health + TTS smoke right now (each ASR/V2V backend has different
# streaming semantics, so we punt). Mark the missing sections with a
# `data_incomplete` flag so the comparator can refuse to silently PASS
# instead of treating absent fields as "no problems found".
out["asr"] = {"data_incomplete": True,
              "reason": "remote collector ships only TTS smoke; "
                        "operator must run ASR matrix separately"}
out["v2v"] = {"data_incomplete": True,
              "reason": "remote collector ships only TTS smoke; "
                        "operator must run V2V matrix separately"}
print("__PARITY_RESULT__:" + json.dumps(out))
"""


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def evaluate_device(report: dict[str, Any], opts: argparse.Namespace) -> dict[str, Any]:
    device_class = report.get("device_class") or _infer_device_class(report)
    budgets = DEVICE_CLASS_BUDGETS.get(device_class)
    flags: list[str] = []
    hard_fail = False
    warnings: list[str] = []

    # ASR
    asr = report.get("asr") or {}
    # Codex Week 3 BLOCKER 3b: a remote collector that returns
    # data_incomplete must NOT silently PASS. The fail/flag policy is:
    #   - data_incomplete + --strict-data → hard fail
    #   - data_incomplete otherwise        → flag (visible in summary)
    # The --skip-asr / --skip-v2v switches let operators acknowledge a
    # device class genuinely can't run that path (e.g. TTS-only device).
    if asr.get("data_incomplete") and not getattr(opts, "skip_asr", False):
        reason = asr.get("reason") or "asr data missing"
        flags.append(f"asr data_incomplete: {reason}")
        if getattr(opts, "strict_data", False):
            hard_fail = True
    for lang_key, rate_key in [("zh", "cer"), ("en", "wer")]:
        bucket = asr.get(lang_key)
        if not bucket:
            continue
        rate = bucket.get(rate_key)
        baseline = bucket.get(f"baseline_{rate_key}")
        if rate is None or baseline is None:
            warnings.append(f"asr {lang_key} {rate_key} or baseline missing")
            continue
        if rate > baseline * (1.0 + CER_REGRESSION_PCT):
            flags.append(
                f"asr {lang_key} {rate_key} regression "
                f"{rate:.3f} vs baseline {baseline:.3f} "
                f"(+{(rate / baseline - 1) * 100:.1f}%)"
            )
            if opts.fail_on_cer_regression:
                hard_fail = True
        missing = bucket.get("missing_finals", 0)
        if missing and missing > 0:
            flags.append(f"asr {lang_key} missing_finals={missing}")
            hard_fail = True  # always a hard fail per spec §D3 line 840

    # TTS
    tts = report.get("tts") or {}
    success_rate = tts.get("success_rate")
    if success_rate is not None and success_rate < 1.0:
        flags.append(f"tts success_rate={success_rate:.2f} (<1.0)")
        hard_fail = True
    if tts.get("pcm_present") is False:
        flags.append("tts pcm_present=false")
        hard_fail = True
    ttfa_p50 = tts.get("ttfa_p50_ms")
    budget_flag = None
    if ttfa_p50 is None:
        warnings.append("tts ttfa_p50_ms missing")
    elif budgets is None:
        warnings.append(
            f"no TTFA budget known for device_class={device_class!r}; "
            f"skipping hard budget gate"
        )
    else:
        lo, hi = budgets
        if ttfa_p50 > hi:
            budget_flag = f"ttfa_p50_ms={ttfa_p50:.0f} > budget_max={hi}"
            flags.append(budget_flag)
            if opts.fail_on_budget:
                hard_fail = True
        elif ttfa_p50 < lo:
            warnings.append(
                f"ttfa_p50_ms={ttfa_p50:.0f} < budget_min={lo} "
                f"(suspicious; verify measurement)"
            )

    # V2V
    v2v = report.get("v2v") or {}
    if v2v.get("data_incomplete") and not getattr(opts, "skip_v2v", False):
        reason = v2v.get("reason") or "v2v data missing"
        flags.append(f"v2v data_incomplete: {reason}")
        if getattr(opts, "strict_data", False):
            hard_fail = True
    if v2v:
        if v2v.get("stop_intent_ok") is False:
            flags.append("v2v stop_intent_ok=false")
            hard_fail = True
        if v2v.get("empty_final_reconnect_ok") is False:
            flags.append("v2v empty_final_reconnect_ok=false")
            hard_fail = True

    status = "FAIL" if hard_fail else ("FLAG" if flags else "PASS")
    return {
        "device": report.get("device"),
        "device_class": device_class,
        "service_url": report.get("service_url"),
        "backend": report.get("backend") or {},
        "ttfa_p50_ms": ttfa_p50,
        "ttfa_budget": (
            f"{budgets[0]}-{budgets[1]} ms" if budgets else "n/a"
        ),
        "ttfa_budget_flag": bool(budget_flag),
        "asr_zh_cer": ((asr.get("zh") or {}).get("cer")),
        "asr_en_wer": ((asr.get("en") or {}).get("wer")),
        "asr_regression_flag": any("regression" in f for f in flags),
        "tts_success_rate": success_rate,
        "v2v_barge_in_ms": v2v.get("barge_in_latency_ms"),
        "v2v_stop_intent": v2v.get("stop_intent_ok"),
        "v2v_reconnect": v2v.get("empty_final_reconnect_ok"),
        "status": status,
        "flags": flags,
        "warnings": warnings,
    }


def _infer_device_class(report: dict[str, Any]) -> str:
    """Best-effort class inference if explicit field is absent."""
    dev = (report.get("device") or "").lower()
    for cls in DEVICE_CLASS_BUDGETS:
        if cls in dev:
            return cls
    return "unknown"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_summary(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall_status = "PASS"
    for row in eval_rows:
        if row["status"] == "FAIL":
            overall_status = "FAIL"
            break
        if row["status"] == "FLAG" and overall_status != "FAIL":
            overall_status = "FLAG"
    return {
        "generated_at": _now_iso(),
        "device_count": len(eval_rows),
        "overall_status": overall_status,
        "devices": eval_rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-Device Parity Report")
    lines.append("")
    lines.append(f"- generated_at: {summary['generated_at']}")
    lines.append(f"- device_count: {summary['device_count']}")
    lines.append(f"- overall_status: **{summary['overall_status']}**")
    lines.append("")
    lines.append("## Cross-device Comparison")
    lines.append("")
    cols = [
        "Device", "Class", "Backend ASR", "Backend TTS",
        "ASR zh CER", "ASR en WER", "ASR regress",
        "TTS success", "TTS TTFA p50 (ms)", "TTS budget", "TTS budget flag",
        "V2V barge-in (ms)", "V2V stop intent", "V2V reconnect",
        "Status",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in summary["devices"]:
        be = row.get("backend") or {}
        cells = [
            str(row.get("device")),
            str(row.get("device_class")),
            str(be.get("asr") or "-"),
            str(be.get("tts") or "-"),
            _fmt_num(row.get("asr_zh_cer"), 3),
            _fmt_num(row.get("asr_en_wer"), 3),
            "YES" if row.get("asr_regression_flag") else "no",
            _fmt_num(row.get("tts_success_rate"), 2),
            _fmt_num(row.get("ttfa_p50_ms"), 0),
            str(row.get("ttfa_budget")),
            "YES" if row.get("ttfa_budget_flag") else "no",
            _fmt_num(row.get("v2v_barge_in_ms"), 0),
            _fmt_bool(row.get("v2v_stop_intent")),
            _fmt_bool(row.get("v2v_reconnect")),
            row.get("status", "?"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Flags and Warnings")
    for row in summary["devices"]:
        if not row.get("flags") and not row.get("warnings"):
            continue
        lines.append(f"### {row['device']} ({row['status']})")
        for f in row.get("flags", []):
            lines.append(f"- FLAG: {f}")
        for w in row.get("warnings", []):
            lines.append(f"- warn: {w}")
        lines.append("")
    if summary["overall_status"] == "PASS":
        lines.append("## Result: PASS")
    elif summary["overall_status"] == "FLAG":
        lines.append("## Result: FLAG (warnings; no hard fail)")
    else:
        lines.append("## Result: FAIL")
    lines.append("")
    return "\n".join(lines)


def _fmt_num(v: Any, prec: int) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_bool(v: Any) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "NO"
    return "-"


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def _collect_mock(opts: argparse.Namespace) -> list[dict[str, Any]]:
    return load_mock_fixtures(Path(opts.fixtures))


def _collect_remote(opts: argparse.Namespace) -> list[dict[str, Any]]:
    tags = DEVICE_CLASS_TAGS.get(opts.devices, []) if opts.devices in DEVICE_CLASS_TAGS \
        else [opts.devices]
    if not tags and opts.devices == "all":
        # No filter — match-all (fleet may have a special flag; we just
        # pass no --tags which most fleet CLIs interpret as "no filter").
        tags = []
    print(f"fleet match tags={tags!r}", file=sys.stderr, flush=True)
    devices = fleet_match(tags)
    if not devices:
        raise SystemExit("fleet match returned no devices")
    print(f"fleet match found {len(devices)} device(s):", file=sys.stderr)
    for d in devices:
        print(f"  - {d.get('id') or d.get('name') or d}", file=sys.stderr)
    reports: list[dict[str, Any]] = []
    for d in devices:
        dev_id = d.get("id") or d.get("name")
        if not dev_id:
            print(f"skipping device with no id: {d!r}", file=sys.stderr)
            continue
        url = d.get("service_url") or opts.base_url
        print(f"collecting from {dev_id} at {url}...", file=sys.stderr)
        r = fleet_exec_collect(
            dev_id, url, runs=opts.runs, warmup=opts.warmup, timeout=opts.timeout,
        )
        # Fold in known fleet metadata if not already present.
        r.setdefault("device", dev_id)
        r.setdefault("device_class", d.get("class") or d.get("device_class"))
        r.setdefault("service_url", url)
        reports.append(r)
    return reports


def _write_per_device(out_dir: Path, reports: list[dict[str, Any]]) -> list[Path]:
    written: list[Path] = []
    ts = _ts_for_path()
    for r in reports:
        dev = (r.get("device") or "unknown").replace("/", "_")
        d = out_dir / dev
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{ts}.json"
        p.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        written.append(p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-device e2e parity harness (mock + remote modes)",
    )
    ap.add_argument("--mode", choices=("mock", "remote"), default="mock",
                    help="mock = load local fixtures; remote = fleet exec live (default: %(default)s)")
    ap.add_argument("--devices", default="all",
                    help="device class selector (jetson, rk, rpi-hailo, all, "
                         "or a raw fleet tag) (default: %(default)s)")
    ap.add_argument("--base-url", default="http://localhost:8621",
                    help="default service URL if fleet metadata lacks one")
    ap.add_argument("--runs", type=int, default=10,
                    help="measured runs per device in remote mode (default: %(default)s)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup runs per device in remote mode (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in remote mode (default: %(default)s)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES),
                    help="mock fixture directory (mock mode)")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE_DIR),
                    help="baseline JSON directory (currently informational; "
                         "per-device baseline fields embedded in fixtures take precedence)")
    ap.add_argument("--fail-on-budget", action="store_true",
                    help="hard-fail when any device exceeds its TTFA budget")
    ap.add_argument("--fail-on-cer-regression", action="store_true",
                    help="hard-fail on any ASR CER/WER regression > 5%%")
    ap.add_argument("--skip-v2v", action="store_true",
                    help="ignore V2V flags in evaluation")
    ap.add_argument("--skip-asr", action="store_true",
                    help="ignore ASR flags in evaluation")
    ap.add_argument("--skip-tts", action="store_true",
                    help="ignore TTS flags in evaluation")
    ap.add_argument("--strict-data", action="store_true",
                    help="hard-fail when a per-device report is marked "
                         "data_incomplete (e.g. remote collector only "
                         "shipped TTS smoke without ASR/V2V coverage). "
                         "Codex Week 3 BLOCKER 3b: prevents silent PASS "
                         "on partial data.")
    args = ap.parse_args()

    # Codex Week 4 round 3 BLOCKER 2: remote mode collectors routinely
    # ship `data_incomplete: True` (the bundled collector script only
    # smoke-tests TTS), so the default flag-only behaviour would silently
    # PASS every remote run. Force --strict-data implicitly in remote mode
    # so missing ASR/V2V coverage hard-fails instead of being swept under
    # a "flag" tag. Mock mode (with fully synthetic fixtures) keeps the
    # original opt-in semantics.
    if args.mode == "remote" and not args.strict_data:
        print("[parity] remote mode → forcing --strict-data "
              "(round 3 BLOCKER 2: prevent silent PASS on incomplete data)",
              file=sys.stderr)
        args.strict_data = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mock":
        reports = _collect_mock(args)
    else:
        reports = _collect_remote(args)

    if not reports:
        print("no per-device reports collected", file=sys.stderr)
        return 3

    per_device_paths = _write_per_device(out_dir, reports)
    print(f"wrote {len(per_device_paths)} per-device JSON file(s) under {out_dir}/")

    # Apply skip flags by erasing irrelevant sections.
    for r in reports:
        if args.skip_asr:
            r["asr"] = {}
        if args.skip_tts:
            r["tts"] = {}
        if args.skip_v2v:
            r["v2v"] = {}

    eval_rows = [evaluate_device(r, args) for r in reports]
    summary = render_summary(eval_rows)
    ts = _ts_for_path()
    summary_json = out_dir / f"summary_{ts}.json"
    summary_md = out_dir / f"summary_{ts}.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    print()
    print(render_markdown(summary))
    print(f"wrote {summary_json}")
    print(f"wrote {summary_md}")

    if summary["overall_status"] == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
