#!/usr/bin/env python3
"""Convert a prepared Qwen3-ASR decoder_hf directory to RKLLM artifacts.

This script intentionally assumes the decoder has already been extracted from
Qwen3-ASR into a normal Qwen3ForCausalLM HuggingFace directory. That matches
the existing `qwen3-asr-rknn/decoder_hf` builder layout and keeps this tool
focused on RKLLM conversion rather than model surgery.

Example:
    python scripts/convert_qwen3_asr_rkllm.py \
      --decoder-hf /home/harve/qwen3-asr-rknn/decoder_hf \
      --dataset /home/harve/qwen3-asr-rknn/data_quant.json \
      --out-dir /home/harve/qwen3-asr-rknn/rkllm \
      --target-platform rk3576 \
      --quant w8a8 \
      --quant-algorithm normal \
      --npu-cores 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NPU_CORES = {
    "rk3576": 2,
    "rk3588": 3,
}

HF_WEIGHT_SENTINELS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@dataclass(frozen=True)
class ConversionPlan:
    decoder_hf: Path
    dataset: Path | None
    out_path: Path
    target_platform: str
    quant: str
    quant_algorithm: str | None
    npu_cores: int
    do_quant: bool
    max_context: int
    optimization_level: int
    dtype: str


def _artifact_name(prefix: str, quant: str, target_platform: str) -> str:
    return f"{prefix}.{quant}.{target_platform}.rkllm"


def _has_hf_weights(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in HF_WEIGHT_SENTINELS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert prepared Qwen3-ASR decoder_hf to RKLLM"
    )
    parser.add_argument("--decoder-hf", required=True, help="Prepared decoder HF dir")
    parser.add_argument("--dataset", default=None, help="Quant calibration JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--target-platform",
        required=True,
        choices=("rk3576", "rk3588"),
        help="RK target platform",
    )
    parser.add_argument(
        "--quant",
        required=True,
        choices=("fp16", "w8a8", "w8a8_g128", "w4a16", "w4a16_g128"),
        help="RKLLM quantized dtype; fp16 disables quantization",
    )
    parser.add_argument(
        "--quant-algorithm",
        default=None,
        help="RKLLM quant algorithm, e.g. normal/grq/gdq. Defaults to normal for w8a8*.",
    )
    parser.add_argument("--npu-cores", type=int, default=None)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--dtype", default="float32", choices=("float16", "float32"))
    parser.add_argument("--prefix", default="decoder_qwen3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned output path without importing RKLLM.",
    )
    return parser


def plan_conversion(args: argparse.Namespace) -> ConversionPlan:
    decoder_hf = Path(args.decoder_hf)
    if not decoder_hf.exists():
        raise ValueError(f"decoder_hf not found: {decoder_hf}")
    if not (decoder_hf / "config.json").exists():
        raise ValueError(f"missing config.json in {decoder_hf}")
    if not _has_hf_weights(decoder_hf):
        expected = ", ".join(HF_WEIGHT_SENTINELS)
        raise ValueError(f"missing HF weights in {decoder_hf}; expected one of: {expected}")

    do_quant = args.quant != "fp16"
    dataset = Path(args.dataset) if args.dataset else None
    if do_quant and (dataset is None or not dataset.exists()):
        raise ValueError("quantized build requires --dataset")

    quant_algorithm = args.quant_algorithm
    if do_quant and quant_algorithm is None and args.quant.startswith("w8a8"):
        quant_algorithm = "normal"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _artifact_name(args.prefix, args.quant, args.target_platform)
    if out_path.exists() and not args.overwrite:
        raise ValueError(f"output exists; pass --overwrite: {out_path}")

    npu_cores = args.npu_cores or DEFAULT_NPU_CORES[args.target_platform]

    return ConversionPlan(
        decoder_hf=decoder_hf,
        dataset=dataset,
        out_path=out_path,
        target_platform=args.target_platform,
        quant=args.quant,
        quant_algorithm=quant_algorithm,
        npu_cores=npu_cores,
        do_quant=do_quant,
        max_context=args.max_context,
        optimization_level=args.optimization_level,
        dtype=args.dtype,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = plan_conversion(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("[INFO] Qwen3-ASR decoder RKLLM conversion")
    print(f"[INFO] decoder_hf      : {plan.decoder_hf}")
    print(f"[INFO] target_platform : {plan.target_platform}")
    print(f"[INFO] quant           : {plan.quant}")
    print(f"[INFO] quant_algorithm : {plan.quant_algorithm}")
    print(f"[INFO] npu_cores       : {plan.npu_cores}")
    print(f"[INFO] max_context     : {plan.max_context}")
    print(f"[INFO] out             : {plan.out_path}")
    print("=" * 72)

    if args.dry_run:
        print("[OK] dry-run validation passed")
        return 0

    try:
        from rkllm.api import RKLLM
    except Exception as exc:
        print(
            "[ERROR] failed to import rkllm.api.RKLLM. Run this script in an "
            "x86 Linux environment with rkllm-toolkit installed.",
            file=sys.stderr,
        )
        print(f"[ERROR] import detail: {exc}", file=sys.stderr)
        return 2

    llm = RKLLM()
    ret = llm.load_huggingface(
        model=str(plan.decoder_hf),
        device="cpu",
        dtype=plan.dtype,
    )
    if ret != 0:
        print(f"[ERROR] load_huggingface failed: {ret}", file=sys.stderr)
        return ret

    ret = llm.build(
        do_quantization=plan.do_quant,
        optimization_level=plan.optimization_level,
        quantized_dtype=plan.quant if plan.do_quant else None,
        quantized_algorithm=plan.quant_algorithm if plan.do_quant else None,
        target_platform=plan.target_platform,
        num_npu_core=plan.npu_cores,
        dataset=str(plan.dataset) if plan.do_quant else None,
        max_context=plan.max_context,
    )
    if ret != 0:
        print(f"[ERROR] build failed: {ret}", file=sys.stderr)
        return ret

    ret = llm.export_rkllm(str(plan.out_path))
    if ret != 0:
        print(f"[ERROR] export_rkllm failed: {ret}", file=sys.stderr)
        return ret

    size_mb = plan.out_path.stat().st_size / 1024 / 1024
    print(f"[OK] saved: {plan.out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
