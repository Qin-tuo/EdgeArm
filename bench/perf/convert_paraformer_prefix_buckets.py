#!/usr/bin/env python3
"""Convert Paraformer hybrid encoder-prefix ONNX into RKNN frame buckets."""

from __future__ import annotations

import argparse
from pathlib import Path


FLOAT_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "tf32": "tfloat32",
}


def convert_bucket(
    onnx_path: Path,
    out_path: Path,
    target: str,
    frames: int,
    precision: str,
    optimization_level: int,
) -> None:
    from rknn.api import RKNN

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            target_platform=target,
            optimization_level=optimization_level,
            float_dtype=FLOAT_DTYPES.get(precision, "float16"),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config ret={ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["speech", "encoder_pad_mask"],
            input_size_list=[[1, frames, 560], [1, frames]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx ret={ret}")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build ret={ret}")

        ret = rknn.export_rknn(str(out_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn ret={ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32"])
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--optimization-level", type=int, default=3)
    args = parser.parse_args()

    for frames in args.frames:
        out_path = (
            args.out_dir
            / args.target
            / f"encoder_prefix_to_block30.{frames}.{args.precision}.rknn"
        )
        print(f"[convert] frames={frames} -> {out_path}", flush=True)
        convert_bucket(
            args.onnx,
            out_path,
            args.target,
            frames,
            args.precision,
            args.optimization_level,
        )
        print(f"[done] {out_path} size={out_path.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
