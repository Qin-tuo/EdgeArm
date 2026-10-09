from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_rkllm_artifact.py"
    spec = importlib.util.spec_from_file_location("check_rkllm_artifact", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, size: int = 1024) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_check_artifact_accepts_exact_quant_and_platform(tmp_path: Path):
    module = _load_module()
    artifact = _write(tmp_path / "decoder_qwen3.w8a8.rk3576.rkllm")

    result = module.check_artifact(
        artifact,
        target_platform="rk3576",
        quant="w8a8",
        min_size_mb=0,
    )

    assert result.ok is True
    assert result.errors == []


def test_check_artifact_rejects_substring_quant_match(tmp_path: Path):
    module = _load_module()
    artifact = _write(tmp_path / "decoder_qwen3.w8a8_g128.rk3576.rkllm")

    result = module.check_artifact(
        artifact,
        target_platform="rk3576",
        quant="w8a8",
        min_size_mb=0,
    )

    assert result.ok is False
    assert any("exact quant token" in e for e in result.errors)


def test_check_artifact_detects_existing_exact_collision(tmp_path: Path):
    module = _load_module()
    artifact = _write(tmp_path / "new" / "decoder_qwen3.w8a8.rk3576.rkllm")
    _write(tmp_path / "models" / "rkllm" / "decoder_qwen3.w8a8.rk3576.rkllm")

    result = module.check_artifact(
        artifact,
        target_platform="rk3576",
        quant="w8a8",
        model_dir=tmp_path / "models",
        min_size_mb=0,
    )

    assert result.ok is False
    assert result.exact_existing_matches
    assert any("would collide" in e for e in result.errors)


def test_check_artifact_warns_about_confusable_existing_files(tmp_path: Path):
    module = _load_module()
    artifact = _write(tmp_path / "new" / "decoder_qwen3.w8a8.rk3576.rkllm")
    _write(tmp_path / "models" / "rkllm" / "decoder_qwen3.w8a8_g128.rk3576.rkllm")

    result = module.check_artifact(
        artifact,
        target_platform="rk3576",
        quant="w8a8",
        model_dir=tmp_path / "models",
        min_size_mb=0,
    )

    assert result.ok is True
    assert result.confusable_existing_matches
    assert any("substring-confusable" in w for w in result.warnings)
