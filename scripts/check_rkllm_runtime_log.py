#!/usr/bin/env python3
"""Verify RKLLM runtime logs prove the intended decoder artifact was loaded."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_PLATFORMS = ("rk3576", "rk3588")
ALLOWED_QUANTS = ("fp16", "w8a8", "w8a8_g128", "w4a16", "w4a16_g128")
EXPECTED_DTYPE = {
    "fp16": "FP16",
    "w8a8": "W8A8",
    "w8a8_g128": "W8A8_G128",
    "w4a16": "W4A16",
    "w4a16_g128": "W4A16_G128",
}

_LOAD_RE = re.compile(r"loading rkllm model from\s+(?P<path>\S+)", re.IGNORECASE)
_DTYPE_RE = re.compile(r"model_dtype:\s*(?P<dtype>[A-Za-z0-9_]+)", re.IGNORECASE)
_TARGET_RE = re.compile(r"target_platform:\s*(?P<platform>[A-Za-z0-9_]+)", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeLoad:
    model_path: str
    model_dtype: str | None
    target_platform: str | None


@dataclass(frozen=True)
class RuntimeLogCheck:
    ok: bool
    errors: list[str]
    warnings: list[str]
    expected_quant: str
    expected_platform: str
    loads: list[RuntimeLoad]
    selected_load: RuntimeLoad | None


def _tokens(path: str) -> list[str]:
    name = Path(path).name
    if name.endswith(".rkllm"):
        name = name.removesuffix(".rkllm")
    return name.split(".")


def parse_loads(text: str) -> list[RuntimeLoad]:
    loads: list[RuntimeLoad] = []
    pending_path: str | None = None
    for line in text.splitlines():
        load_match = _LOAD_RE.search(line)
        if load_match:
            if pending_path is not None:
                loads.append(
                    RuntimeLoad(
                        model_path=pending_path,
                        model_dtype=None,
                        target_platform=None,
                    )
                )
            pending_path = load_match.group("path")
            continue

        if pending_path is None:
            continue

        dtype_match = _DTYPE_RE.search(line)
        target_match = _TARGET_RE.search(line)
        if dtype_match or target_match:
            loads.append(
                RuntimeLoad(
                    model_path=pending_path,
                    model_dtype=dtype_match.group("dtype") if dtype_match else None,
                    target_platform=(
                        target_match.group("platform") if target_match else None
                    ),
                )
            )
            pending_path = None
    if pending_path is not None:
        loads.append(
            RuntimeLoad(model_path=pending_path, model_dtype=None, target_platform=None)
        )
    return loads


def check_runtime_log(
    text: str,
    *,
    quant: str,
    target_platform: str,
    artifact_basename: str | None = None,
    select: str = "last",
) -> RuntimeLogCheck:
    errors: list[str] = []
    warnings: list[str] = []
    if quant not in ALLOWED_QUANTS:
        errors.append(f"unsupported quant: {quant}")
    if target_platform not in ALLOWED_PLATFORMS:
        errors.append(f"unsupported target platform: {target_platform}")
    if select not in ("last", "first"):
        errors.append(f"unsupported select mode: {select}")

    loads = parse_loads(text)
    if not loads:
        errors.append("no RKLLM load line found")
        return RuntimeLogCheck(
            ok=False,
            errors=errors,
            warnings=warnings,
            expected_quant=quant,
            expected_platform=target_platform,
            loads=loads,
            selected_load=None,
        )

    selected = loads[-1] if select == "last" else loads[0]
    if artifact_basename:
        basename_matches = [
            load for load in loads if Path(load.model_path).name == artifact_basename
        ]
        if basename_matches:
            selected = basename_matches[-1] if select == "last" else basename_matches[0]

    tokens = _tokens(selected.model_path)
    if quant not in tokens:
        errors.append(
            f"loaded artifact lacks exact quant token {quant!r}: {selected.model_path}"
        )
    if target_platform not in tokens:
        errors.append(
            "loaded artifact lacks exact platform token "
            f"{target_platform!r}: {selected.model_path}"
        )
    if artifact_basename and Path(selected.model_path).name != artifact_basename:
        errors.append(
            "loaded artifact basename mismatch: "
            f"{Path(selected.model_path).name} != {artifact_basename}"
        )

    expected_dtype = EXPECTED_DTYPE.get(quant)
    if selected.model_dtype is None:
        errors.append("selected RKLLM load has no model_dtype line")
    elif expected_dtype and selected.model_dtype.upper() != expected_dtype:
        errors.append(
            f"model_dtype mismatch: {selected.model_dtype} != {expected_dtype}"
        )

    if selected.target_platform is None:
        warnings.append("selected RKLLM load has no target_platform in dtype line")
    elif selected.target_platform.lower() != target_platform:
        errors.append(
            "target_platform mismatch: "
            f"{selected.target_platform} != {target_platform.upper()}"
        )

    if len(loads) > 1:
        if artifact_basename:
            warnings.append(
                f"found {len(loads)} RKLLM load events; checked {artifact_basename}"
            )
        else:
            warnings.append(
                f"found {len(loads)} RKLLM load events; checked the {select} one"
            )

    return RuntimeLogCheck(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        expected_quant=quant,
        expected_platform=target_platform,
        loads=loads,
        selected_load=selected,
    )


def _read_log(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(errors="replace")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify RKLLM runtime logs match the intended artifact"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="-",
        help="Log file path, or '-' for stdin (default)",
    )
    parser.add_argument("--quant", required=True, choices=ALLOWED_QUANTS)
    parser.add_argument("--target-platform", required=True, choices=ALLOWED_PLATFORMS)
    parser.add_argument("--artifact-basename", help="Expected loaded .rkllm basename")
    parser.add_argument(
        "--select",
        choices=("last", "first"),
        default="last",
        help="Which RKLLM load event to validate when logs contain multiple loads",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = check_runtime_log(
        _read_log(args.log_file),
        quant=args.quant,
        target_platform=args.target_platform,
        artifact_basename=args.artifact_basename,
        select=args.select,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] RKLLM runtime log")
        if result.selected_load:
            print(f"  model : {result.selected_load.model_path}")
            print(f"  dtype : {result.selected_load.model_dtype or '(missing)'}")
            print(f"  target: {result.selected_load.target_platform or '(missing)'}")
        for warning in result.warnings:
            print(f"  WARN  : {warning}")
        for error in result.errors:
            print(f"  ERROR : {error}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
