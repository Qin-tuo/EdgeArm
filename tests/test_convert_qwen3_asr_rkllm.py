from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_qwen3_asr_rkllm.py"
    spec = importlib.util.spec_from_file_location("convert_qwen3_asr_rkllm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decoder_hf(tmp_path: Path) -> Path:
    decoder = tmp_path / "decoder_hf"
    decoder.mkdir()
    (decoder / "config.json").write_text("{}")
    (decoder / "model.safetensors").write_bytes(b"placeholder")
    return decoder


def _sharded_decoder_hf(tmp_path: Path) -> Path:
    decoder = tmp_path / "decoder_hf"
    decoder.mkdir()
    (decoder / "config.json").write_text("{}")
    (decoder / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (decoder / "model-00001-of-00002.safetensors").write_bytes(b"placeholder")
    (decoder / "model-00002-of-00002.safetensors").write_bytes(b"placeholder")
    return decoder


def _args(tmp_path: Path, **overrides):
    values = {
        "decoder_hf": str(overrides.pop("decoder_hf", None) or _decoder_hf(tmp_path)),
        "dataset": str(tmp_path / "data_quant.json"),
        "out_dir": str(tmp_path / "rkllm"),
        "target_platform": "rk3576",
        "quant": "w8a8",
        "quant_algorithm": None,
        "npu_cores": None,
        "max_context": 4096,
        "optimization_level": 1,
        "dtype": "float32",
        "prefix": "decoder_qwen3",
        "overwrite": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plan_conversion_defaults_w8a8_for_rk3576(tmp_path: Path):
    module = _load_module()
    (tmp_path / "data_quant.json").write_text("[]")

    plan = module.plan_conversion(_args(tmp_path))

    assert plan.do_quant is True
    assert plan.quant_algorithm == "normal"
    assert plan.npu_cores == 2
    assert plan.out_path.name == "decoder_qwen3.w8a8.rk3576.rkllm"


def test_plan_conversion_fp16_does_not_require_dataset(tmp_path: Path):
    module = _load_module()

    plan = module.plan_conversion(
        _args(tmp_path, target_platform="rk3588", quant="fp16", dataset=None)
    )

    assert plan.do_quant is False
    assert plan.dataset is None
    assert plan.npu_cores == 3
    assert plan.out_path.name == "decoder_qwen3.fp16.rk3588.rkllm"


def test_plan_conversion_accepts_sharded_safetensors(tmp_path: Path):
    module = _load_module()
    decoder = _sharded_decoder_hf(tmp_path)
    (tmp_path / "data_quant.json").write_text("[]")

    plan = module.plan_conversion(_args(tmp_path, decoder_hf=str(decoder)))

    assert plan.decoder_hf == decoder
    assert plan.do_quant is True


def test_plan_conversion_rejects_quant_without_dataset(tmp_path: Path):
    module = _load_module()

    try:
        module.plan_conversion(_args(tmp_path, dataset=None))
    except ValueError as exc:
        assert "requires --dataset" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_main_dry_run_does_not_import_rkllm(tmp_path: Path, capsys):
    module = _load_module()
    decoder = _decoder_hf(tmp_path)

    rc = module.main(
        [
            "--decoder-hf",
            str(decoder),
            "--out-dir",
            str(tmp_path / "rkllm"),
            "--target-platform",
            "rk3588",
            "--quant",
            "fp16",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "dry-run validation passed" in capsys.readouterr().out
