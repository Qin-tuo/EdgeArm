from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_rkllm_runtime_log.py"
    )
    spec = importlib.util.spec_from_file_location("check_rkllm_runtime_log", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_log_accepts_matching_artifact_dtype_and_platform():
    module = _load_module()
    log = """
I rkllm: loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w8a8.rk3588.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, max_context_limit: 4096, npu_core_num: 2, target_platform: RK3588, model_dtype: W8A8
"""

    result = module.check_runtime_log(log, quant="w8a8", target_platform="rk3588")

    assert result.ok is True
    assert result.selected_load.model_dtype == "W8A8"


def test_runtime_log_rejects_substring_quant_artifact():
    module = _load_module()
    log = """
I rkllm: loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w8a8_g128.rk3588.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3588, model_dtype: W8A8_G128
"""

    result = module.check_runtime_log(log, quant="w8a8", target_platform="rk3588")

    assert result.ok is False
    assert any("exact quant token" in e for e in result.errors)
    assert any("model_dtype mismatch" in e for e in result.errors)


def test_runtime_log_rejects_wrong_loaded_basename():
    module = _load_module()
    log = """
I rkllm: loading rkllm model from /opt/asr/models/rkllm/decoder_qwen3.w4a16_g128.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W4A16_G128
"""

    result = module.check_runtime_log(
        log,
        quant="w4a16_g128",
        target_platform="rk3576",
        artifact_basename="decoder_qwen3.w8a8.rk3576.rkllm",
    )

    assert result.ok is False
    assert any("basename mismatch" in e for e in result.errors)


def test_runtime_log_checks_last_load_by_default():
    module = _load_module()
    log = """
I rkllm: loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w4a16.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W4A16
I rkllm: loading rkllm model from /opt/asr/models/rkllm/decoder_qwen3.w4a16_g128.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W4A16_G128
"""

    result = module.check_runtime_log(
        log,
        quant="w4a16_g128",
        target_platform="rk3576",
    )

    assert result.ok is True
    assert result.selected_load.model_path.endswith("w4a16_g128.rk3576.rkllm")
    assert result.warnings


def test_runtime_log_selects_matching_basename_among_multiple_loads():
    module = _load_module()
    log = """
I rkllm: loading rkllm model from /opt/tts/models/talker.w4a16.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W4A16
I rkllm: loading rkllm model from /opt/asr/models/rkllm/decoder_qwen3.w8a8.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W8A8
I rkllm: loading rkllm model from /opt/tts/models/another.w4a16.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, target_platform: RK3576, model_dtype: W4A16
"""

    result = module.check_runtime_log(
        log,
        quant="w8a8",
        target_platform="rk3576",
        artifact_basename="decoder_qwen3.w8a8.rk3576.rkllm",
    )

    assert result.ok is True
    assert result.selected_load.model_path.endswith("decoder_qwen3.w8a8.rk3576.rkllm")
    assert any("checked decoder_qwen3.w8a8.rk3576.rkllm" in w for w in result.warnings)


def test_runtime_log_fails_when_no_load_found():
    module = _load_module()

    result = module.check_runtime_log("service started", quant="w8a8", target_platform="rk3588")

    assert result.ok is False
    assert "no RKLLM load line found" in result.errors
