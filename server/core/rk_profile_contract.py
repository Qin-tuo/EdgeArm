"""Configuration drift checks for the sole Qwen3-ASR + Matcha RK3588 profile.

This checks declared settings, not target hardware correctness or performance.
Only bounded configuration values are included in status payloads.
"""

from __future__ import annotations

import os
from typing import Mapping


RK_RELEASE_PROFILES = frozenset({"rk3588-default"})

_COMMON_EXPECTED: dict[str, str] = {
    "ASR_MAX_NEW_TOKENS": "64",
    # Preserve full multi-sentence utterances; actual quality needs board tests.
    "ASR_FINAL_STOP_ON_PUNCT": "0",
    "ASR_FINAL_STOP_MIN_CHARS": "8",
    "ASR_FINAL_STOP_MIN_CHUNKS": "2",
    "ASR_NPU_CORE_MASK": "NPU_CORE_1",
    "QWEN3_ASR_CHUNK_CONFIRM": "0",
    "QWEN3_ASR_STREAM_MODE": "true_streaming",
    "QWEN3_ASR_STREAM_TRUE": "1",
    "QWEN3_ASR_TRUE_ROLL_SEC": "5",
    "QWEN3_ASR_TRUE_PARTIAL_TOKENS": "8",
    "QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS": "1500",
    "QWEN3_ASR_TRUE_PARTIAL_WARMUP": "2",
    "QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S": "2.5",
    # Keep backend endpoint above client silence plus finalize budget.
    "VAD_ENDPOINT_SILENCE_MS": "1500",
    "MATCHA_USE_ORT": "1",
    "MATCHA_MODEL_SEQ_LEN": "80",
    "MATCHA_STREAM_CHUNK_MS": "40",
}

_PLATFORM_EXPECTED = {"rk3588": {"QWEN3_ASR_VAD_FINAL_ASYNC": "1", "MATCHA_MIN_MEL_FRAMES": "72", "VOCOS_FRAMES": "256"}}
_PROFILE_KEYS = (*_COMMON_EXPECTED, "RK_PLATFORM", "QWEN3_ASR_VAD_FINAL_ASYNC", "MATCHA_MIN_MEL_FRAMES", "VOCOS_FRAMES")


def _profile_name(profile: Mapping[str, object] | None) -> str:
    if not profile:
        return ""
    return str(profile.get("name") or "").strip()


def is_release_profile(profile: Mapping[str, object] | None) -> bool:
    """Return whether *profile* is the selected product profile."""
    return _profile_name(profile) in RK_RELEASE_PROFILES


def _device_for(name: str, profile_env: Mapping[str, object], env: Mapping[str, str]) -> str | None:
    value = profile_env.get("RK_PLATFORM") or env.get("RK_PLATFORM")
    if value:
        return str(value)
    if name.startswith("rk3588-"):
        return "rk3588"
    return None


def runtime_status(
    profile: Mapping[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compare a selected RK release profile with the effective process env.

    ``verified`` checks JSON/process consistency; it is not a hardware verdict.
    """
    name = _profile_name(profile)
    profile_env_raw = (profile or {}).get("env") or {}
    profile_env = profile_env_raw if isinstance(profile_env_raw, Mapping) else {}
    actual = os.environ if env is None else env

    if name not in RK_RELEASE_PROFILES:
        return {
            "required": False,
            "profile": name or None,
            "device": _device_for(name, profile_env, actual),
            "contract": "not_applicable",
            "verified": True,
            "settings": {},
            "missing_profile": [],
            "missing_runtime": [],
            "mismatches": {},
        }

    device = _device_for(name, profile_env, actual)
    expected = dict(_COMMON_EXPECTED)
    expected.update(_PLATFORM_EXPECTED["rk3588"])
    expected["RK_PLATFORM"] = "rk3588"
    keys = _PROFILE_KEYS

    missing_profile = [key for key in keys if str(profile_env.get(key, "")).strip() == ""]
    missing_runtime = [key for key in keys if str(actual.get(key, "")).strip() == ""]
    mismatches: dict[str, dict[str, str]] = {}
    for key, wanted in expected.items():
        profile_value = str(profile_env.get(key, ""))
        actual_value = str(actual.get(key, ""))
        if profile_value and profile_value != wanted:
            mismatches[key] = {
                "expected": wanted,
                "profile": profile_value,
                "runtime": actual_value,
            }
        elif actual_value and actual_value != wanted:
            mismatches[key] = {
                "expected": wanted,
                "profile": profile_value or wanted,
                "runtime": actual_value,
            }

    settings = {key: str(actual.get(key, "<missing>")) for key in keys}
    return {
        "required": True,
        "profile": name,
        "device": device,
        "contract": "rk-qwen3-true-streaming-matcha-v1",
        "verified": not missing_profile and not missing_runtime and not mismatches,
        "settings": settings,
        "missing_profile": missing_profile,
        "missing_runtime": missing_runtime,
        "mismatches": mismatches,
    }


def format_failure(status: Mapping[str, object]) -> str:
    """Format a short actionable startup/readiness diagnostic."""
    details: list[str] = []
    if status.get("missing_profile"):
        details.append("missing profile keys=" + ",".join(map(str, status["missing_profile"])))
    if status.get("missing_runtime"):
        details.append("missing runtime keys=" + ",".join(map(str, status["missing_runtime"])))
    mismatches = status.get("mismatches") or {}
    if mismatches:
        details.append("mismatches=" + ",".join(map(str, mismatches)))
    return (
        f"RK profile {status.get('profile')!r} does not satisfy "
        f"{status.get('contract')}: " + ("; ".join(details) or "unknown drift")
    )
