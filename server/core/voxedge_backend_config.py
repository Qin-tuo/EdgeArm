"""RK3588 config adaptation retained from OpenVoiceStream."""
from __future__ import annotations
import os
from typing import Optional


def build_rk_asr_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``RKASRConfig`` from env.

    Mirrors the env reads in legacy ``app/backends/rk/asr.py``: ``RK_PLATFORM``
    (in ``__init__``) plus the per-call energy-split reads inside
    ``_split_at_silence_energy``. ``long_audio_threshold_s`` was a module
    constant (``_LONG_AUDIO_THRESHOLD_S = 15.0``), so it has no env override.

    env → RKASRConfig field map (legacy):
      RK_PLATFORM                 → platform (legacy default "rk3576")
      ASR_ENERGY_SPLIT_RMS        → energy_split_rms (0.003)
      ASR_ENERGY_MIN_SILENCE_MS   → energy_min_silence_ms (120)
      (constant _LONG_AUDIO_THRESHOLD_S) → long_audio_threshold_s (15.0)
    """
    from voxedge.backends.rk.asr import RKASRConfig

    if env is None:
        env = os.environ

    try:
        energy_split_rms = float(env.get("ASR_ENERGY_SPLIT_RMS", "0.003"))
    except ValueError:
        energy_split_rms = 0.003
    try:
        energy_min_silence_ms = int(env.get("ASR_ENERGY_MIN_SILENCE_MS", "120"))
    except ValueError:
        energy_min_silence_ms = 120

    return RKASRConfig(
        platform=env.get("RK_PLATFORM", "rk3588"),
        energy_split_rms=energy_split_rms,
        energy_min_silence_ms=energy_min_silence_ms,
        long_audio_threshold_s=15.0,
    )

def build_rk_tts_config(
    profile: Optional[dict] = None,
    env: Optional[dict] = None,
):
    """Build a ``RKTTSConfig`` from env.

    The RK adapter delegates backend selection to rkvoice-stream via the
    ``TTS_BACKEND`` env (read inside rkvoice-stream, not here). The only
    product-layer config is ``model_id`` (legacy ``OVS_TTS_MODEL_ID`` →
    backend-name fallback "rk").

    env → RKTTSConfig field map (see voxedge rk/tts header):
      OVS_TTS_MODEL_ID → model_id ("rk")
    """
    from voxedge.backends.rk.tts import RKTTSConfig

    if env is None:
        env = os.environ

    return RKTTSConfig(model_id=env.get("OVS_TTS_MODEL_ID") or "rk")

_ASR_CONFIG_BUILDERS = {"rk.asr": build_rk_asr_config}
_TTS_CONFIG_BUILDERS = {"rk.tts": build_rk_tts_config}


def build_config_for_spec(spec, kind, profile=None):
    """Build the voxedge config dataclass for ``spec`` (kind='asr'|'tts')."""
    builders = _ASR_CONFIG_BUILDERS if kind == "asr" else _TTS_CONFIG_BUILDERS
    builder = builders.get(spec)
    if builder is None:
        return None
    return builder(profile=profile)

def concurrency_capability_for_spec(spec, cls, kind, profile=None):
    """ConcurrencyCapability for a voxedge backend without loading models.

    Returns ``None`` when ``spec`` is not a known voxedge spec (caller falls
    back to the legacy classmethod path).

    NOTE: we deliberately do NOT call ``cls(config=config)`` — some backends do
    heavy work in ``__init__`` (e.g. RK calls ``create_asr()`` which imports
    ``rkvoice_stream`` / inits the NPU), which would crash or init hardware
    during a pure capability probe. Every backend's ``concurrency_capability``
    only reads ``self._config`` (or returns a constant / is a classmethod), so
    we build a config-bearing stub via ``__new__`` and call the method on it,
    skipping ``__init__`` entirely.
    """
    config = build_config_for_spec(spec, kind, profile)
    if config is None:
        return None
    stub = cls.__new__(cls)
    # voxedge backends are not consistent about the attribute they read their
    # config from -- paraformer/trt_edge_llm use ``_config``, sensevoice uses
    # ``_cfg``. Setting only one produced a stub whose concurrency_capability()
    # raised AttributeError, which the caller swallowed into
    # ConcurrencyCapability.default() (max_concurrent=1): every knob for that
    # backend looked wired up and silently did nothing.
    stub._config = config
    stub._cfg = config
    try:
        return stub.concurrency_capability()
    except TypeError:
        # classmethod-style ``concurrency_capability(cls, profile=None)``
        # (sherpa / rk) bound through the class.
        return cls.concurrency_capability(profile)