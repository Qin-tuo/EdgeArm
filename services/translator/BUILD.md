# Translator service — build notes

## TL;DR — running the pinned image on Jetson

A pre-built `linux/arm64` slim CUDA image is published to Seeed's internal
registry; you do not have to rebuild from source unless the JetPack version
or model changes.

```bash
# 1) `docker login sensecraft-missionpack.seeed.cn` if not already done.
# 2) Compose pulls the right image automatically:
docker compose -f deploy/docker-compose.yml up -d translator

# Or pull + run manually:
docker pull sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:translator-cuda-jetson-v1
docker run -d --name translator --runtime=nvidia -p 9001:9001 \
  -v translator-models:/models:ro \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib:/host-libs:/host-nvidia-libs:/host-cuda \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TRANSLATOR_DEVICE=cuda \
  -e TRANSLATOR_MODEL_PATH=/models/nllb-200-distilled-600m-ct2-int8 \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:translator-cuda-jetson-v1

# 3) Sanity:
curl -s http://localhost:9001/health
# {"status":"ok","model":"nllb-200-distilled-600M","device":"cuda"}

curl -s -X POST http://localhost:9001/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好世界","src_lang":"zho_Hans","tgt_lang":"eng_Latn"}'
```

### Pinned image

| Tag | Digest | Size | Built against |
|---|---|---|---|
| `translator-cuda-jetson-v2` | `sha256:c47bde5b5794c7f9f385de2b48be6f1185dd5c2987000962112c5982f278065c` | 579 MB | JetPack 6.x (L4T R36.4.x), CUDA 12.6, cuDNN 9.3, Python 3.10, ct2 v4.7.2; threadpool for non-blocking translate |
| `translator-cuda-jetson-v1` | `sha256:ee4765c6fc2b0a1e2719aa136ccc0ff59e8b918a4dcd8ac04e312e165927d636` | 579 MB | Same as v2 minus the threadpool; deprecated |

### Model

The image expects the NLLB-200 CT2 model at
`/models/nllb-200-distilled-600m-ct2-int8/` inside the `translator-models`
named volume. Bring-up on a fresh device:

```bash
# On a host with HF access (or behind HF mirror via HF_ENDPOINT):
pip install --user ctranslate2 transformers sentencepiece huggingface_hub torch
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir /tmp/nllb-200-distilled-600m-ct2-int8 \
  --quantization int8
# (sentencepiece.bpe.model is NOT created by the converter — copy it from
#  the HF cache manually:)
cp ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/*/sentencepiece.bpe.model \
   /tmp/nllb-200-distilled-600m-ct2-int8/

# Stage into the named volume:
docker volume create translator-models
docker run --rm -v translator-models:/dest -v /tmp/nllb-200-distilled-600m-ct2-int8:/src \
  alpine cp -r /src /dest/nllb-200-distilled-600m-ct2-int8
```

The CT2-converted model is ~605 MB on disk.

---

## Two Docker paths ship side-by-side

| File | Image tag | When to use |
|---|---|---|
| `Dockerfile` | `translator-latest` | x86 / cloud / "good enough" CPU only. Self-contained `pip install ctranslate2` (CPU-only on arm64). |
| `Dockerfile.slim` | `translator-slim-v2` | Jetson Orin NX with CUDA. ~580 MB. Requires a one-time on-device build of CTranslate2 (see below). |

The slim path on Orin NX delivers **14–35× speedup** over the CPU path:

| Length | CPU p50 | CUDA p50 (slim-v2) |
|---|---|---|
| `你好世界` (4 char) | 4432 ms | **125 ms** |
| 26-char zh | 5297 ms | **370 ms** |
| 52-char zh | 2630 ms | **639 ms** |

## Why bare-metal-built, not extracted from a prebuilt image

`pypi` `ctranslate2` aarch64 wheels are CPU-only. `dustynv/ctranslate2` does not
exist on Docker Hub. Multi-stage extraction from `dustynv/faster-whisper`
(which does have CUDA-built CT2) collides with glibc/libstdc++ ABI gaps when
the runtime base differs from the builder. Compiling on the Jetson host
itself sidesteps every ABI question because the resulting `.so` is built
against the exact same toolchain as the runtime container's Ubuntu 22.04 +
the host's libcudnn / libcublas (bind-mounted at runtime).

## How to build CT2 on Orin NX (one-time per JetPack version)

Prereqs already present on a stock JetPack 6.x Orin (`R36.4.x`): `cmake >= 3.18`,
`gcc 11.4`, `git`, `python3.10-dev`, `pybind11-dev`, `libopenblas-dev`,
`libcudnn9-dev-cuda-12`, `cuda-toolkit-12-6`. Verify with `dpkg -l | grep -E
'cuda-toolkit-12-6|libcudnn9-dev'`.

```bash
# 1. Clone CT2 + initialize submodules
mkdir -p /home/harvest/ct2-build && cd /home/harvest/ct2-build
git clone --recursive -b v4.7.2 https://github.com/OpenNMT/CTranslate2.git
# (cutlass + thrust may need manual seeding from a fast mirror; see "Gotchas" below.)

# 2. Patch out the vendored thrust includes — CT2 v4.7.2 pins thrust 1.16.0
#    which is API-incompatible with CUDA 12.6's bundled CUB 2.5.0
#    (PtxVersion / SyncStream / MaxSmOccupancy / AliasTemporaries moved out
#    of thrust::cub namespace). Letting system thrust 2.5 win fixes the
#    nvcc errors.
cd CTranslate2
sed -i.bak '557,562 s|^|# |' CMakeLists.txt   # comments the vendored thrust target_include_directories

# 3. Configure + compile
rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_DYNAMIC_LOADING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
    -DCMAKE_INSTALL_PREFIX=/home/harvest/ct2-prefix \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . -j$(nproc)   # ~3-4 min on Orin NX

# 4. Library install (CLI link may fail on libopenblas rpath — that's fine,
#    we only need the library + headers; the ct2-translator CLI is not used)
mkdir -p /home/harvest/ct2-prefix/lib /home/harvest/ct2-prefix/include
cp libctranslate2.so* /home/harvest/ct2-prefix/lib/
cp -r ../include/* /home/harvest/ct2-prefix/include/

# 5. Python wheel
cd ../python
python3 -m pip install --user pybind11>=3.0 setuptools wheel
CTRANSLATE2_ROOT=/home/harvest/ct2-prefix \
LD_LIBRARY_PATH=/home/harvest/ct2-prefix/lib \
python3 setup.py bdist_wheel
# Produces dist/ctranslate2-4.7.2-cp310-cp310-linux_aarch64.whl
```

## How to build the slim Docker image (also on Orin NX)

```bash
cd /path/to/seeed-local-voice/services/translator

# Build context expects ./wheels/ and ./ct2-artifacts/ to be populated.
# Copy the ct2 artifacts and the service deps:
mkdir -p ct2-artifacts
cp /home/harvest/ct2-prefix/lib/libctranslate2.so* ct2-artifacts/
cp /home/harvest/ct2-build/CTranslate2/python/dist/ctranslate2-*.whl ct2-artifacts/

# wheels/ holds pre-downloaded cp310 aarch64 service deps (fastapi, uvicorn,
# sentencepiece, pydantic, numpy<2 + transitive). Download once on a fast
# network:
mkdir -p wheels
pip download --dest wheels \
    --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
    --platform manylinux_2_17_aarch64 --platform manylinux_2_27_aarch64 \
    --python-version 310 --implementation cp --only-binary :all: \
    'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'sentencepiece>=0.2' \
    'pydantic>=2.7' 'numpy<2' exceptiongroup

docker build -f Dockerfile.slim -t seeed-local-voice:translator-slim-v2 .
```

`ct2-artifacts/` and `wheels/` are intentionally gitignored — they're large
binaries tied to a specific JetPack version. Rebuild them whenever the host
CUDA / cuDNN / Python version moves.

## How to publish to the Seeed registry

After a successful build + bench on the Jetson, push the new image so other
hosts can pull it without rebuilding from source:

```bash
TAG=translator-cuda-jetson-v1   # bump the v1 suffix per JetPack / CT2 / model update
REG=sensecraft-missionpack.seeed.cn/solution/seeed-local-voice

docker tag seeed-local-voice:translator-slim-v2 ${REG}:${TAG}
docker push ${REG}:${TAG}
# Capture the digest from the push output and record it in:
#   - this file's "Pinned image" table (TL;DR section)
#   - deploy/docker-compose.yml (TRANSLATOR_IMAGE default)
#   - your fleet release notes
```

Pin a new tag (`-v2`, `-v3`, …) instead of overwriting `-v1` whenever the
upstream NLLB-200 weights / quantization / Jetson JetPack version changes,
so older deployments can always pin back to a known-good image.

## Run

```bash
docker run -d --name translator --runtime=nvidia -p 9001:9001 \
  -v translator-models:/models:ro \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib:/host-libs:/host-nvidia-libs:/host-cuda \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TRANSLATOR_DEVICE=cuda \
  -e TRANSLATOR_MODEL_PATH=/models/nllb-200-distilled-600m-ct2-int8 \
  seeed-local-voice:translator-slim-v2
```

`NVIDIA_VISIBLE_DEVICES=all` + `NVIDIA_DRIVER_CAPABILITIES=all` are required
on Jetson's CSV-mode container runtime — without them `/dev/nvmap` is not
mapped into the container and CUDA init fails with `NvRmMemInitNvmap failed`.

## Gotchas

- **cutlass + thrust submodule download** can stall on CN networks. The
  CT2 git submodule URLs hit GitHub directly. Fallback: download the
  tarballs on a fast network (Mac / WSL via proxy) and extract into
  `third_party/cutlass/` and `third_party/thrust/` manually:
  ```
  curl -L https://github.com/NVIDIA/cutlass/archive/bbe579a9e3beb6ea6626d9227ec32d0dae119a49.tar.gz | tar xz -C third_party/cutlass --strip-components=1
  curl -L https://github.com/NVIDIA/thrust/archive/d997cd37a95b0fa2f1a0cd4697fd1188a842fbc8.tar.gz | tar xz -C third_party/thrust --strip-components=1
  ```
  Pin the SHAs to whatever `git submodule status` reports for your CT2 tag.

- **pybind11 must be >= 3.0** for CT2 v4.7.2 Python bindings (uses
  `options.disable_enum_members_docstring`). Ubuntu 22.04's apt
  `pybind11-dev 2.9.1` is too old — install via pip.

- **The Dockerfile builds the runtime image only.** It cannot reproduce the
  CT2 build itself; the `.so` and `.whl` come from the host. Trying to
  `docker build` from Mac without `ct2-artifacts/` populated will fail.

- **Image tag history**: `translator-latest` (4.22 GB, CPU only),
  `translator-slim` (17 GB, dustynv-based intermediate experiment — can be
  pruned), `translator-slim-v2` (579 MB, bare-metal CUDA). Use `-slim-v2`
  in production on Orin.
