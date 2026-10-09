# Kokoro TRT Reproduction

This document freezes the validated Kokoro v1.0 TensorRT split path for Jetson
Orin Nano / JetPack 6.2 / TensorRT 10.3 / CUDA 12.6.

## Runtime Path

```text
TRT encoder prefix FP16
-> CPU length regulator
-> TRT decoder backbone FP16
-> TRT source BF16
-> TRT generator rest FP16
-> CPU post/ISTFT
```

The runtime does not use ONNX Runtime CUDA EP or TensorRT EP. ONNX Runtime is
used only for CPU sidecar subgraphs.

## Profiles

| Profile | Segment tokens | Stream chunk | Intended use |
|---|---:|---:|---|
| `jetson-kokoro-trt` | 64 | 40 ms | Default performance profile. |
| `jetson-kokoro-trt-perf` | 64 | 40 ms | Explicit alias for the default performance profile. |
| `jetson-kokoro-trt-quality` | 48 | 40 ms | More conservative segmentation for long text quality gates. |
| `jetson-kokoro-trt-long` | 96 | 60 ms | Longer per-segment budget; exercises the 256-512 frame bucket more often. |

Start the profile with:

```bash
OVS_PROFILE=jetson-kokoro-trt docker compose -f deploy/docker-compose.yml up -d
```

Use `HF_ENDPOINT=https://hf-mirror.com` on networks where direct Hugging Face
downloads are slow or blocked.

## Artifact Set

The frozen artifact record is:

- Manifest: `deploy/artifacts/kokoro_trt_manifest.json`
- HF repo: `harvestsu/seeed-local-voice-artifacts`
- Bundle: `models/kokoro-multi-lang-v1_0/engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz`
- Size: `279478777`
- SHA-256: `4e6e11099624e9900807851c8721fdd61179d0a6a3ebf9132a82765da199c0cb`

The bundle contains seven TensorRT engines plus the CPU ONNX sidecars required
by the runtime. Startup resolves engines through the normal profile
`required_engines` path: local cache, then HF bundle, then local Jetson build.

## Build Or Repack

On the target Jetson, after engines already exist in the model directory:

```bash
python3 scripts/build_engine_bundle.py \
  --profile configs/profiles/jetson-kokoro-trt.json \
  --out /tmp/seeed-local-voice-kokoro-artifacts \
  --skip-build
```

To rebuild missing engines instead of packaging existing files, omit
`--skip-build`. The build fallback uses
`scripts/build_kokoro_split_generator_trt.sh`.

## HTTP Verification

For a TTS-only Kokoro profile, point the verifier at Kokoro for TTS and at a
separate local ASR service for ASR:

```bash
python3 scripts/verify_tts_asr_roundtrip.py \
  --tts-url http://127.0.0.1:8621 \
  --asr-url http://127.0.0.1:8622 \
  --language en \
  --out-dir /tmp/kokoro-roundtrip
```

To validate the streaming endpoint:

```bash
python3 scripts/verify_tts_asr_roundtrip.py \
  --tts-url http://127.0.0.1:8621 \
  --asr-url http://127.0.0.1:8622 \
  --language en \
  --streaming \
  --out-dir /tmp/kokoro-roundtrip-stream
```

The script writes WAV files and a `report.json` under the output directory.

## Validated Nano Result

On `orin-nano`, isolated service validation on `127.0.0.1:8001` loaded the
Kokoro profile, resolved all seven engines as cache hits, loaded
`long_bucket=True`, and returned:

| Endpoint | Duration | Wall time | RTF estimate |
|---|---:|---:|---:|
| `/tts` | 19.625 s | 1.415 s | 0.072 |
| `/tts/stream` | 19.625 s | 1.146 s | 0.058 |

Direct backend probe with a 126-token case produced `frame_t=311`, selected the
long bucket, and measured RTF `0.065`.
