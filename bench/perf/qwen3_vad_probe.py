"""Probe Qwen3 true-streaming VAD endpoint behavior on fixed WAV samples.

This script mirrors the accumulator used by
``rkvoice_stream.backends.asr.qwen3.streaming.Qwen3TrueStreamingASRStream`` so
endpoint-policy experiments can be debugged without running the decoder.
"""
from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV")
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV")
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz")
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class ProbeResult:
    path: str
    backend: str
    audio_s: float
    speech_s: float
    silence_ms: float
    pending_ms: float
    endpoint_s: float | None
    endpoint_audio_s: float | None
    endpoint_speech_s: float | None
    endpoint_silence_ms: float | None
    speech_frames: int
    nonspeech_frames: int
    transitions: list[dict]


class _Accumulator:
    def __init__(
        self,
        *,
        sustain_frames: int,
        endpoint_silence_ms: int,
        min_speech_s: float,
        min_audio_s: float,
    ) -> None:
        self.sustain_frames = sustain_frames
        self.endpoint_silence_samples = int(endpoint_silence_ms * SAMPLE_RATE / 1000)
        self.min_speech_samples = int(min_speech_s * SAMPLE_RATE)
        self.min_audio_samples = int(min_audio_s * SAMPLE_RATE)
        self.audio_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.pending_silence_samples = 0
        self.consec_speech_frames = 0
        self.last_is_speech: bool | None = None
        self.speech_frames = 0
        self.nonspeech_frames = 0
        self.transitions: list[dict] = []
        self.endpoint_sample: int | None = None
        self.endpoint_speech_samples: int | None = None
        self.endpoint_silence_samples_seen: int | None = None

    def update(self, is_speech: bool, n_samples: int) -> None:
        self.audio_samples += n_samples
        if is_speech:
            self.speech_frames += 1
            self.consec_speech_frames += 1
            self.speech_samples += n_samples
            if self.consec_speech_frames >= self.sustain_frames:
                self.pending_silence_samples = 0
                self.silence_samples = 0
            elif self.speech_samples > 0:
                self.pending_silence_samples += n_samples
        else:
            self.nonspeech_frames += 1
            self.consec_speech_frames = 0
            if self.speech_samples > 0:
                self.silence_samples += self.pending_silence_samples + n_samples
                self.pending_silence_samples = 0

        if self.last_is_speech is None or self.last_is_speech != is_speech:
            self.transitions.append(
                {
                    "t_s": round(self.audio_samples / SAMPLE_RATE, 3),
                    "is_speech": is_speech,
                    "speech_s": round(self.speech_samples / SAMPLE_RATE, 3),
                    "silence_ms": round(self.silence_samples * 1000 / SAMPLE_RATE, 1),
                }
            )
        self.last_is_speech = is_speech

        if self.endpoint_sample is None and self.endpoint_ready():
            self.endpoint_sample = self.audio_samples
            self.endpoint_speech_samples = self.speech_samples
            self.endpoint_silence_samples_seen = self.silence_samples

    def endpoint_ready(self) -> bool:
        return (
            self.audio_samples >= self.min_audio_samples
            and self.speech_samples >= self.min_speech_samples
            and self.silence_samples >= self.endpoint_silence_samples
        )


def _probe_webrtc(
    samples: np.ndarray,
    *,
    aggr: int,
    frame_ms: int,
    sustain_frames: int,
    endpoint_silence_ms: int,
    min_speech_s: float,
    min_audio_s: float,
) -> _Accumulator:
    import webrtcvad

    vad = webrtcvad.Vad(aggr)
    frame_samples = int(frame_ms * SAMPLE_RATE / 1000)
    acc = _Accumulator(
        sustain_frames=sustain_frames,
        endpoint_silence_ms=endpoint_silence_ms,
        min_speech_s=min_speech_s,
        min_audio_s=min_audio_s,
    )
    used = (len(samples) // frame_samples) * frame_samples
    pcm = (np.clip(samples[:used], -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    frame_bytes = frame_samples * 2
    for i in range(used // frame_samples):
        frame = pcm[i * frame_bytes:(i + 1) * frame_bytes]
        acc.update(vad.is_speech(frame, SAMPLE_RATE), frame_samples)
    return acc


def _probe_silero(
    samples: np.ndarray,
    *,
    model_path: Path,
    sustain_frames: int,
    endpoint_silence_ms: int,
    min_speech_s: float,
    min_audio_s: float,
) -> _Accumulator:
    from rkvoice_stream.vad.silero import SileroVAD, VAD_WINDOW_SIZE

    vad = SileroVAD(str(model_path))
    acc = _Accumulator(
        sustain_frames=sustain_frames,
        endpoint_silence_ms=endpoint_silence_ms,
        min_speech_s=min_speech_s,
        min_audio_s=min_audio_s,
    )
    used = (len(samples) // VAD_WINDOW_SIZE) * VAD_WINDOW_SIZE
    for start in range(0, used, VAD_WINDOW_SIZE):
        frame = samples[start:start + VAD_WINDOW_SIZE]
        vad.feed(frame)
        acc.update(bool(vad.is_speech), VAD_WINDOW_SIZE)
    return acc


def _result(
    path: Path,
    backend: str,
    samples: np.ndarray,
    acc: _Accumulator,
    *,
    max_transitions: int,
) -> ProbeResult:
    return ProbeResult(
        path=str(path),
        backend=backend,
        audio_s=round(len(samples) / SAMPLE_RATE, 3),
        speech_s=round(acc.speech_samples / SAMPLE_RATE, 3),
        silence_ms=round(acc.silence_samples * 1000 / SAMPLE_RATE, 1),
        pending_ms=round(acc.pending_silence_samples * 1000 / SAMPLE_RATE, 1),
        endpoint_s=(
            None if acc.endpoint_sample is None
            else round(acc.endpoint_sample / SAMPLE_RATE, 3)
        ),
        endpoint_audio_s=(
            None if acc.endpoint_sample is None
            else round(acc.endpoint_sample / SAMPLE_RATE, 3)
        ),
        endpoint_speech_s=(
            None if acc.endpoint_speech_samples is None
            else round(acc.endpoint_speech_samples / SAMPLE_RATE, 3)
        ),
        endpoint_silence_ms=(
            None if acc.endpoint_silence_samples_seen is None
            else round(acc.endpoint_silence_samples_seen * 1000 / SAMPLE_RATE, 1)
        ),
        speech_frames=acc.speech_frames,
        nonspeech_frames=acc.nonspeech_frames,
        transitions=acc.transitions[:max_transitions],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--backend", choices=["webrtc", "silero"], default="webrtc")
    parser.add_argument("--webrtc-aggr", type=int, default=2)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--endpoint-silence-ms", type=int, default=400)
    parser.add_argument("--min-speech-s", type=float, default=0.5)
    parser.add_argument("--min-audio-s", type=float, default=0.0)
    parser.add_argument("--silero-model", type=Path, default=None)
    parser.add_argument("--max-transitions", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    for wav_path in args.wav:
        samples = _read_wav(wav_path)
        if args.backend == "webrtc":
            acc = _probe_webrtc(
                samples,
                aggr=args.webrtc_aggr,
                frame_ms=args.frame_ms,
                sustain_frames=args.sustain_frames,
                endpoint_silence_ms=args.endpoint_silence_ms,
                min_speech_s=args.min_speech_s,
                min_audio_s=args.min_audio_s,
            )
        else:
            if args.silero_model is None:
                raise SystemExit("--silero-model is required for --backend silero")
            acc = _probe_silero(
                samples,
                model_path=args.silero_model,
                sustain_frames=args.sustain_frames,
                endpoint_silence_ms=args.endpoint_silence_ms,
                min_speech_s=args.min_speech_s,
                min_audio_s=args.min_audio_s,
            )
        rows.append(
            _result(
                wav_path,
                args.backend,
                samples,
                acc,
                max_transitions=args.max_transitions,
            )
        )

    if args.json:
        print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))
    else:
        for r in rows:
            endpoint = "none" if r.endpoint_s is None else f"{r.endpoint_s:.3f}s"
            print(
                f"{Path(r.path).name}: backend={r.backend} audio={r.audio_s:.2f}s "
                f"speech={r.speech_s:.2f}s silence={r.silence_ms:.0f}ms "
                f"endpoint={endpoint} speech_frames={r.speech_frames} "
                f"nonspeech_frames={r.nonspeech_frames}"
            )
            print("  transitions:", json.dumps(r.transitions, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
