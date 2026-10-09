"""Client-side VAD for utterance segmentation.

Two backends:
- silero: silero-vad onnx (preferred, accurate). pip extra: `silero-vad`
- energy: pure numpy energy threshold (fallback, always available)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EnergyVAD:
    """Simple RMS-based VAD. Threshold tuned for typical built-in mics."""

    name = "energy"

    def __init__(self, threshold: float = 0.012, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate

    def is_speech(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) == 0:
            return False
        rms = float(np.sqrt(np.mean(samples * samples)))
        return rms > self.threshold

    def reset(self) -> None:
        pass


class SileroVAD:
    """silero-vad onnx run directly through onnxruntime — TORCH-FREE.

    The ``silero_vad`` Python package does ``import torch`` at module import
    time (utils_vad.py), so it cannot be imported at all on a torch-purged
    image. We therefore locate the bundled ``silero_vad.onnx`` file via the
    package's resource path (without importing it) and drive the
    ``onnxruntime.InferenceSession`` ourselves with numpy in/out.

    Silero v5/v6 onnx I/O (16kHz):
      inputs : ``input`` [1, 64+512], ``state`` [2,1,128], ``sr`` int64 scalar
      outputs: ``output`` [1,1] speech prob, ``stateN`` [2,1,128] new state
    Each 512-sample window must be prepended with the trailing 64 samples
    (``context_size``) of the previous window; the recurrent ``state`` is
    carried across calls. Skipping the context makes every prob ~0 (silent),
    which is exactly the stall this class exists to avoid.
    """

    name = "silero"

    _CONTEXT = 64  # context_size for 16kHz (32 for 8kHz)

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        import onnxruntime as ort  # late import; torch-free

        self.threshold = threshold
        self.sample_rate = sample_rate
        # silero expects 32ms windows at 16kHz = 512 samples
        self._win = 512 if sample_rate == 16000 else 256
        self._context_size = 64 if sample_rate == 16000 else 32

        onnx_path = self._locate_onnx_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._sr = np.array(self.sample_rate, dtype=np.int64)
        self._buf = np.zeros(0, dtype=np.float32)
        self.reset()

    @staticmethod
    def _locate_onnx_model() -> str:
        """Find the bundled silero_vad.onnx WITHOUT importing the package
        (its __init__ pulls in torch). Uses the module spec's file location.
        """
        import importlib.util
        import os

        spec = importlib.util.find_spec("silero_vad")
        if spec is None or spec.origin is None:
            raise ImportError("silero_vad package not installed")
        pkg_dir = os.path.dirname(spec.origin)
        onnx_path = os.path.join(pkg_dir, "data", "silero_vad.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"silero_vad.onnx not found at {onnx_path}"
            )
        return onnx_path

    def is_speech(self, pcm: bytes) -> bool:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])
        any_speech = False
        while len(self._buf) >= self._win:
            win = self._buf[: self._win].reshape(1, -1).astype(np.float32)
            self._buf = self._buf[self._win :]
            x = np.concatenate([self._ctx, win], axis=1)
            out = self._session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            prob = float(np.asarray(out[0]).reshape(-1)[0])
            self._state = out[1]
            self._ctx = x[:, -self._context_size :]
            if prob >= self.threshold:
                any_speech = True
        return any_speech

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, self._context_size), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)


def create_vad(backend: str, sample_rate: int = 16000, threshold: float | None = None):
    """Build a VAD by name. `auto` tries silero, falls back to energy."""
    if backend in ("silero", "auto"):
        try:
            return SileroVAD(
                threshold=threshold if threshold is not None else 0.5,
                sample_rate=sample_rate,
            )
        except Exception as e:
            if backend == "silero":
                raise
            logger.info("silero VAD unavailable (%s), falling back to energy VAD", e)
    return EnergyVAD(
        threshold=threshold if threshold is not None else 0.012,
        sample_rate=sample_rate,
    )


__all__ = ["EnergyVAD", "SileroVAD", "create_vad"]
