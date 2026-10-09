import json
from pathlib import Path

import pytest

from server.core.rk_artifacts import RKArtifactError, _validate_runtime_contract


ROOT = Path(__file__).resolve().parents[2]


def test_rk_runtime_contract_accepts_expected_env(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "matcha_rknn")
    monkeypatch.setenv("MATCHA_USE_ORT", "1")
    monkeypatch.setenv("VOCOS_FRAMES", "256")

    _validate_runtime_contract(
        {
            "runtime_contract": {
                "env": {
                    "TTS_BACKEND": "matcha_rknn",
                    "MATCHA_USE_ORT": "1",
                    "VOCOS_FRAMES": "256",
                }
            }
        }
    )


def test_rk_runtime_contract_rejects_shape_drift(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "matcha_rknn")
    monkeypatch.setenv("MATCHA_USE_ORT", "0")
    monkeypatch.setenv("VOCOS_FRAMES", "256")

    with pytest.raises(RKArtifactError, match="MATCHA_USE_ORT"):
        _validate_runtime_contract(
            {
                "runtime_contract": {
                    "env": {
                        "TTS_BACKEND": "matcha_rknn",
                        "MATCHA_USE_ORT": "1",
                        "VOCOS_FRAMES": "256",
                    }
                }
            }
        )


def test_paraformer_hybrid_rknn_decoder_contract_matches_profile():
    _assert_paraformer_contract_matches_profile(
        "rk3576-paraformer-hybrid-rknn-decoder-2026-06-09",
        "configs/profiles/rk3576-paraformer-hybrid-rknn-decoder-matcha.json",
        "opt/asr/paraformer/rknn/rk3576/decoder.400x40.fp16.rknn",
    )


def test_rk3588_paraformer_hybrid_rknn_decoder_contract_matches_profile():
    _assert_paraformer_contract_matches_profile(
        "rk3588-paraformer-hybrid-rknn-decoder-2026-06-09",
        "configs/profiles/rk3588-paraformer-hybrid-rknn-decoder-matcha.json",
        "opt/asr/paraformer/rknn/rk3588/decoder.400x40.fp16.rknn",
    )


def test_legacy_paraformer_profile_names_select_current_rknn_decoder_artifacts():
    _assert_paraformer_contract_matches_profile(
        "rk3576-paraformer-hybrid-rknn-decoder-2026-06-09",
        "configs/profiles/rk3576-paraformer-matcha.json",
        "opt/asr/paraformer/rknn/rk3576/decoder.400x40.fp16.rknn",
    )
    _assert_paraformer_contract_matches_profile(
        "rk3588-paraformer-hybrid-rknn-decoder-2026-06-09",
        "configs/profiles/rk3588-paraformer-matcha.json",
        "opt/asr/paraformer/rknn/rk3588/decoder.400x40.fp16.rknn",
    )


def _assert_paraformer_contract_matches_profile(
    artifact_set: str,
    profile_path: str,
    decoder_path: str,
) -> None:
    manifest = json.loads((ROOT / "deploy/artifacts/rk_manifest.json").read_text())
    profile = json.loads((ROOT / profile_path).read_text())

    spec = manifest["artifact_sets"][artifact_set]
    contract_env = spec["runtime_contract"]["env"]
    profile_env = profile["env"]

    assert profile_env["RK_ARTIFACT_SET"] == artifact_set
    for key in (
        "ASR_BACKEND",
        "PARAFORMER_MODEL_DIR",
        "PARAFORMER_RKNN_DIR",
        "PARAFORMER_RKNN_ENCODER_MODE",
        "PARAFORMER_RKNN_DECODER",
        "PARAFORMER_RKNN_ENC_PRECISION",
        "PARAFORMER_RKNN_DEC_PRECISION",
        "PARAFORMER_ENCODER_SUFFIX_ONNX",
        "PARAFORMER_FBANK_CMVN",
        "PARAFORMER_STREAM_DECODE",
        "PARAFORMER_STREAM_PROCESS_SEC",
    ):
        assert contract_env[key] == profile_env[key]

    assert contract_env["PARAFORMER_RKNN_DECODER"] == "rknn"
    assert any(
        item["path"] == decoder_path for item in spec["files"]
    )
