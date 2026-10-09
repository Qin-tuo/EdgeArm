#!/usr/bin/env python3
"""Preflight checks for RKLLM decoder artifacts before device A/B.

The runtime selects RKLLM files by exact filename tokens:

    decoder_qwen3.w8a8.rk3576.rkllm

This script catches the common mistakes before copying an artifact to a board:
wrong platform token, wrong quant token, suspiciously tiny files, or an existing
artifact with the same exact quant/platform already present in a model dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_PLATFORMS = ("rk3576", "rk3588")
ALLOWED_QUANTS = ("fp16", "w8a8", "w8a8_g128", "w4a16", "w4a16_g128")


@dataclass(frozen=True)
class ArtifactCheck:
    artifact: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    size_bytes: int
    tokens: list[str]
    exact_existing_matches: list[str]
    confusable_existing_matches: list[str]


def _tokens(path: Path) -> list[str]:
    name = path.name
    if name.endswith(".rkllm"):
        name = name.removesuffix(".rkllm")
    return name.split(".")


def _scan_rkllm_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for subdir in (model_dir / "decoder", model_dir / "rkllm"):
        if subdir.exists():
            files.extend(sorted(subdir.glob("*.rkllm")))
    return files


def check_artifact(
    artifact: Path,
    *,
    target_platform: str,
    quant: str,
    model_dir: Path | None = None,
    min_size_mb: float = 100.0,
    allow_existing: bool = False,
) -> ArtifactCheck:
    errors: list[str] = []
    warnings: list[str] = []

    if target_platform not in ALLOWED_PLATFORMS:
        errors.append(f"unsupported target platform: {target_platform}")
    if quant not in ALLOWED_QUANTS:
        errors.append(f"unsupported quant: {quant}")

    if not artifact.exists():
        errors.append(f"artifact not found: {artifact}")
        size_bytes = 0
    elif not artifact.is_file():
        errors.append(f"artifact is not a file: {artifact}")
        size_bytes = 0
    else:
        size_bytes = artifact.stat().st_size

    if artifact.suffix != ".rkllm":
        errors.append(f"artifact must end with .rkllm: {artifact.name}")

    tokens = _tokens(artifact)
    if quant not in tokens:
        errors.append(f"artifact filename lacks exact quant token {quant!r}: {artifact.name}")
    if target_platform not in tokens:
        errors.append(
            f"artifact filename lacks exact platform token {target_platform!r}: {artifact.name}"
        )

    min_size_bytes = int(min_size_mb * 1024 * 1024)
    if size_bytes and size_bytes < min_size_bytes:
        errors.append(
            f"artifact is suspiciously small: {size_bytes} bytes < {min_size_mb:g} MiB"
        )

    exact_existing: list[str] = []
    confusable_existing: list[str] = []
    if model_dir is not None:
        if not model_dir.exists():
            warnings.append(f"model_dir does not exist yet: {model_dir}")
        else:
            artifact_resolved = artifact.resolve()
            for candidate in _scan_rkllm_files(model_dir):
                candidate_tokens = _tokens(candidate)
                exact = quant in candidate_tokens and target_platform in candidate_tokens
                confusable = (
                    quant in candidate.name
                    and target_platform in candidate.name
                    and not exact
                )
                if exact:
                    exact_existing.append(str(candidate))
                    if candidate.resolve() != artifact_resolved and not allow_existing:
                        errors.append(
                            "existing exact quant/platform artifact would collide: "
                            f"{candidate}"
                        )
                elif confusable:
                    confusable_existing.append(str(candidate))
            if confusable_existing:
                warnings.append(
                    "found substring-confusable artifacts; runtime must use exact token matching"
                )

    return ArtifactCheck(
        artifact=str(artifact),
        ok=not errors,
        errors=errors,
        warnings=warnings,
        size_bytes=size_bytes,
        tokens=tokens,
        exact_existing_matches=exact_existing,
        confusable_existing_matches=confusable_existing,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a RKLLM decoder artifact before RK device A/B"
    )
    parser.add_argument("artifact", help="Path to .rkllm artifact")
    parser.add_argument("--target-platform", required=True, choices=ALLOWED_PLATFORMS)
    parser.add_argument("--quant", required=True, choices=ALLOWED_QUANTS)
    parser.add_argument("--model-dir", help="Optional ASR model dir to scan for collisions")
    parser.add_argument("--min-size-mb", type=float, default=100.0)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = check_artifact(
        Path(args.artifact),
        target_platform=args.target_platform,
        quant=args.quant,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        min_size_mb=args.min_size_mb,
        allow_existing=args.allow_existing,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.artifact}")
        print(f"  tokens: {'.'.join(result.tokens)}")
        print(f"  size  : {result.size_bytes} bytes")
        for warning in result.warnings:
            print(f"  WARN  : {warning}")
        for error in result.errors:
            print(f"  ERROR : {error}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
