"""Whisper log-mel feature extractor implemented with numpy only.

This is a drop-in equivalent of the path used by
``transformers.WhisperFeatureExtractor`` for Qwen3 ASR (feature_size=128,
sampling_rate=16000, n_fft=400, hop_length=160). It is intentionally a
narrow port of the Whisper feature pipeline so we can drop the
``transformers`` and ``librosa/scipy`` runtime dependencies.

Spec: docs/plans/asr-mel-librosa-2026-04-27.md
"""

from __future__ import annotations

import numpy as np

# Constants pinned to Whisper / Qwen3 ASR usage.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mask = freq >= min_log_hz
    mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mask = mels >= min_log_mel
    freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
    return freqs


def _build_mel_filterbank() -> np.ndarray:
    n_freqs = N_FFT // 2 + 1
    fftfreqs = np.linspace(0.0, SAMPLE_RATE / 2, n_freqs, dtype=np.float64)
    min_mel = _hz_to_mel(np.array([FMIN], dtype=np.float64))[0]
    max_mel = _hz_to_mel(np.array([FMAX], dtype=np.float64))[0]
    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, N_MELS + 2, dtype=np.float64))

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fftfreqs[np.newaxis, :]
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:N_MELS + 2] - mel_f[:N_MELS])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def _get_mel_state(cache: dict, chunk_length: int) -> dict:
    """Return cached mel filter state, building it lazily.

    The mel filter matrix only depends on (sr, n_fft, n_mels, fmin, fmax).
    Keep chunk_length in the key to avoid colliding with older cache entries.
    """
    key = ("numpy_slaney", chunk_length)
    state = cache.get(key)
    if state is not None and state.get("backend") == "numpy_slaney":
        return state

    state = {"backend": "numpy_slaney", "mel_basis": _build_mel_filterbank()}
    cache[key] = state
    return state


def compute_whisper_log_mel(
    audio: np.ndarray,
    chunk_length: int,
    cache: dict,
) -> np.ndarray:
    """Compute Whisper log-mel features as ``[1, 128, T]`` float32.

    Mirrors the transformers ``WhisperFeatureExtractor`` numpy path:
      * pad/trim to ``chunk_length * 16000`` samples
      * centered STFT with ``n_fft=400``, ``hop_length=160``, periodic Hann,
        reflect padding
      * drop the final STFT frame
      * power spectrum (``|stft|**2``)
      * Slaney mel filter bank, base-10 log, floor ``1e-10``
      * Whisper dynamic range clamp + ``(x + 4) / 4`` normalization
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    n_samples = int(chunk_length) * SAMPLE_RATE
    if audio.shape[0] < n_samples:
        audio = np.pad(
            audio,
            (0, n_samples - audio.shape[0]),
            mode="constant",
            constant_values=0.0,
        )
    else:
        audio = audio[:n_samples]

    state = _get_mel_state(cache, chunk_length)
    mel_basis = state["mel_basis"]

    # Centered STFT, periodic Hann window, reflect padding to match
    # transformers/librosa behavior without importing scipy/librosa.
    audio = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
        writeable=False,
    )
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    stft = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1).T

    # Drop final frame to match transformers torch path (stft[..., :-1]).
    magnitudes = np.abs(stft[:, :-1]).astype(np.float32) ** 2.0

    mel_spec = mel_basis @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec[np.newaxis, :, :].astype(np.float32, copy=False)
