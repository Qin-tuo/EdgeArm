"""Voice clone runner.

Pipeline per iteration:
  1. POST reference WAV to /tts/clone/embedding  →  speaker_embedding_b64
  2. POST /tts/clone {text, embedding}           →  synthesized WAV
  3. Compute speaker similarity between reference WAV and synthesized WAV
     using resemblyzer (third-party embedding, independent of server's
     internal embedding space). Cosine similarity ∈ [-1, 1].

Skips gracefully (returns []) if the backend reports 501 (no clone support).
"""
from __future__ import annotations
import base64, json, time
from pathlib import Path
from typing import Optional

import requests


def _resemblyzer_embed(wav_bytes: bytes):
    """Lazy import of resemblyzer + numpy. Returns 256-d numpy array."""
    import io, numpy as np, wave
    from resemblyzer import VoiceEncoder, preprocess_wav
    # Decode WAV → float32 mono
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # resemblyzer expects 16k mono float32 in [-1, 1]
    if sr != 16000:
        try:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000)
        except ImportError:
            # crude resample
            ratio = 16000 / sr
            idx = (np.arange(int(len(pcm) * ratio)) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, len(pcm) - 1)
            pcm = pcm[idx]
    # cache encoder on function attribute
    if not hasattr(_resemblyzer_embed, "_enc"):
        _resemblyzer_embed._enc = VoiceEncoder()
    pcm = preprocess_wav(pcm, source_sr=16000)
    return _resemblyzer_embed._enc.embed_utterance(pcm)


def _cosine(a, b) -> float:
    import numpy as np
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_clone(base_url: str, references: list[Path], texts: list[dict],
              warmup: int = 1, runs: int = 5,
              skip_similarity: bool = False) -> list[dict]:
    """references = list of WAV file paths (one ref voice each).
       texts      = list of {id, lang, text} dicts."""
    if not references:
        return [{"error": "no reference voices provided"}]
    if not texts:
        return [{"error": "no clone texts provided"}]

    records: list[dict] = []
    total = warmup + runs
    for k in range(total):
        label = "warmup" if k < warmup else "steady"
        ref_path = references[k % len(references)]
        text_entry = texts[k % len(texts)]
        ref_bytes = ref_path.read_bytes()

        # Step 1: extract embedding (timed)
        t0 = time.monotonic()
        try:
            r = requests.post(f"{base_url.rstrip('/')}/tts/clone/embedding",
                              files={"file": ("ref.wav", ref_bytes, "audio/wav")},
                              timeout=60)
        except Exception as e:
            records.append({"label": label, "k": k, "error": f"embedding request: {e}"})
            continue
        if r.status_code == 501:
            print("  Backend reports voice_clone not supported — aborting.")
            return [{"error": "backend has no voice_clone capability"}]
        if r.status_code != 200:
            records.append({"label": label, "k": k, "error": f"embedding HTTP {r.status_code}: {r.text[:100]}"})
            continue
        embed_b64 = r.json()["speaker_embedding_b64"]
        t_embed = time.monotonic()

        # Step 2: synthesize (timed)
        t_synth_start = time.monotonic()
        try:
            r = requests.post(f"{base_url.rstrip('/')}/tts/clone",
                              json={"text": text_entry["text"],
                                    "speaker_embedding_b64": embed_b64,
                                    "language": text_entry.get("lang", "zh")},
                              timeout=120)
        except Exception as e:
            records.append({"label": label, "k": k, "error": f"synth request: {e}"})
            continue
        if r.status_code != 200:
            records.append({"label": label, "k": k,
                            "error": f"synth HTTP {r.status_code}: {r.text[:100]}"})
            continue
        synth_bytes = r.content
        t_synth_end = time.monotonic()

        # Step 3: similarity (optional, slow)
        similarity: Optional[float] = None
        if not skip_similarity:
            try:
                ref_emb = _resemblyzer_embed(ref_bytes)
                synth_emb = _resemblyzer_embed(synth_bytes)
                similarity = _cosine(ref_emb, synth_emb)
            except ImportError as e:
                similarity = None
                if k == 0:
                    print(f"  (similarity disabled: {e})")
            except Exception as e:
                if k == 0:
                    print(f"  (similarity error: {e})")

        # parse server-reported RTF if present
        try:
            audio_dur = float(r.headers.get("X-Audio-Duration", 0))
        except Exception:
            audio_dur = 0.0

        rec = {
            "label": label, "k": k,
            "ref": ref_path.name, "text_id": text_entry["id"], "lang": text_entry.get("lang", "zh"),
            "embed_ms": (t_embed - t0) * 1000,
            "synth_ms": (t_synth_end - t_synth_start) * 1000,
            "total_ms": (t_synth_end - t0) * 1000,
            "synth_audio_dur_s": audio_dur,
            "rtf": (t_synth_end - t_synth_start) / audio_dur if audio_dur else 0.0,
        }
        if similarity is not None:
            rec["similarity"] = similarity
        records.append(rec)
        sim_str = f"  sim={similarity:.3f}" if similarity is not None else ""
        print(f"  [{label:6s} {k+1:02d}] ref={ref_path.name:20s} text={text_entry['id']:14s} "
              f"embed={rec['embed_ms']:.0f}ms synth={rec['synth_ms']:.0f}ms RTF={rec['rtf']:.3f}{sim_str}")
    return records
