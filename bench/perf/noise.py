"""Noise mixing for robustness testing.

Avoids 12 GB MUSAN download — generates 3 noise types from primitives:
  - white:  Gaussian
  - pink:   1/f via filtered Gaussian
  - babble: concat of other utterances in corpus (loop to length)

Mixes clean speech at target SNR dB. Returns new WAV bytes.

Generated SHA256s won't match committed manifest, so the noisy variant
runs as its own scenario and reports WER degradation vs clean baseline.
"""
from __future__ import annotations
import io, wave
from pathlib import Path

import numpy as np


def _read_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def _write_wav(samples: np.ndarray, sr: int) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def gen_white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def gen_pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Voss-McCartney approximation of 1/f noise."""
    rows = 16
    arr = rng.standard_normal((rows, n)).astype(np.float32)
    # Each row updates at half the rate of the previous one
    for r in range(rows):
        period = 1 << r
        # Hold each value for `period` samples
        if period > 1:
            mask = np.repeat(arr[r, ::period], period)[:n]
            arr[r] = mask
    return arr.sum(axis=0) / rows


def make_babble(corpus_samples: list[np.ndarray], n: int, rng) -> np.ndarray:
    """Concatenate randomly-shuffled, randomly-offset slices from corpus."""
    pool = np.concatenate(corpus_samples) if corpus_samples else gen_white(n, rng)
    if len(pool) < n:
        pool = np.tile(pool, n // len(pool) + 1)
    start = int(rng.integers(0, max(1, len(pool) - n)))
    return pool[start:start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so resulting SNR = snr_db, then add."""
    if len(noise) < len(speech):
        noise = np.tile(noise, len(speech) // len(noise) + 1)
    noise = noise[: len(speech)]
    # Active region of speech (avoid scaling by leading silence)
    speech_power = np.mean(speech ** 2) + 1e-12
    noise_power  = np.mean(noise  ** 2) + 1e-12
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = (target_noise_power / noise_power) ** 0.5
    return speech + noise * scale


def add_noise(wav_bytes: bytes, snr_db: float, noise_type: str = "babble",
              babble_pool: list[bytes] | None = None,
              seed: int = 42) -> bytes:
    """Return new WAV bytes with noise added at target SNR."""
    speech, sr = _read_wav(wav_bytes)
    rng = np.random.default_rng(seed)
    if noise_type == "white":
        noise = gen_white(len(speech), rng)
    elif noise_type == "pink":
        noise = gen_pink(len(speech), rng)
    elif noise_type == "babble":
        if not babble_pool:
            noise = gen_white(len(speech), rng)
        else:
            pool_samples = [_read_wav(b)[0] for b in babble_pool]
            noise = make_babble(pool_samples, len(speech), rng)
    else:
        raise ValueError(f"unknown noise_type: {noise_type}")
    mixed = mix_at_snr(speech, noise, snr_db)
    return _write_wav(mixed, sr)
