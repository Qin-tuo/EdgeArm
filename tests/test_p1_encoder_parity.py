#!/usr/bin/env python3
"""
P1 encoder parity test: ORT (Docker) vs native TRT (host).
Run on NX host. Uses cuda-python (not pycuda).

Usage: ssh harvest@<nx> 'python3 /home/harvest/jetson-voice/tests/test_p1_encoder_parity.py'
"""

import os
import sys
import subprocess
import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/harvest/qwen3-asr-v2/qwen3-asr-v2")
ENGINE_PATH = os.environ.get("ENGINE_PATH", os.path.join(MODEL_DIR, "asr_encoder_fp16.engine"))
CONTAINER = "reachy_speech-speech-1"
DUMP_SCRIPT = os.environ.get("DUMP_SCRIPT", "/home/harvest/jetson-voice/tests/dump_ort_encoder.py")


def run_ort(mel_npy_path, out_npy_path):
    # /tmp is isolated between host and container. docker cp the input in,
    # run, docker cp the output out.
    container_in = "/tmp/" + os.path.basename(mel_npy_path)
    container_out = "/tmp/" + os.path.basename(out_npy_path)
    subprocess.check_call(["docker", "cp", mel_npy_path, f"{CONTAINER}:{container_in}"])
    cmd = ["docker", "exec", CONTAINER, "python3", DUMP_SCRIPT, MODEL_DIR, container_in, container_out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ORT failed: {r.stderr}")
    subprocess.check_call(["docker", "cp", f"{CONTAINER}:{container_out}", out_npy_path])
    subprocess.run(["docker", "exec", CONTAINER, "rm", "-f", container_in, container_out],
                   capture_output=True)


def run_trt(mel, engine_path):
    import tensorrt as trt
    from cuda import cudart

    def _check(call):
        err, *rest = call
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")
        return rest[0] if rest else None

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    mel_name = None
    feat_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        nl = n.lower()
        if "mel" in nl:
            mel_name = n
        elif "feature" in nl or "audio" in nl:
            feat_name = n
    if mel_name is None or feat_name is None:
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        modes = [engine.get_tensor_mode(n) for n in names]
        mel_name = next(n for n, m in zip(names, modes) if m == trt.TensorIOMode.INPUT)
        feat_name = next(n for n, m in zip(names, modes) if m == trt.TensorIOMode.OUTPUT)

    ctx.set_input_shape(mel_name, tuple(mel.shape))
    out_shape = tuple(ctx.get_tensor_shape(feat_name))
    out_elem = int(np.prod(out_shape))

    d_mel = _check(cudart.cudaMalloc(mel.nbytes))
    d_out = _check(cudart.cudaMalloc(out_elem * 4))
    _check(cudart.cudaMemcpy(d_mel, mel.tobytes(), mel.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    ctx.set_tensor_address(mel_name, int(d_mel))
    ctx.set_tensor_address(feat_name, int(d_out))

    stream = _check(cudart.cudaStreamCreate())
    ctx.execute_async_v3(stream_handle=int(stream))
    _check(cudart.cudaStreamSynchronize(stream))

    out = np.empty(out_shape, dtype=np.float32)
    _check(cudart.cudaMemcpy(out.ctypes.data, d_out, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    _check(cudart.cudaFree(d_mel))
    _check(cudart.cudaFree(d_out))
    _check(cudart.cudaStreamDestroy(stream))
    return out


def compare(ort, trt, label):
    if len(trt.shape) == 2 and len(ort.shape) == 3:
        trt = trt[np.newaxis, :, :]
    elif len(ort.shape) == 2 and len(trt.shape) == 3:
        ort = ort[np.newaxis, :, :]
    t_dim = 1 if len(ort.shape) >= 3 else 0
    min_t = min(ort.shape[t_dim], trt.shape[t_dim])
    if t_dim == 1:
        ort_c = ort[:, :min_t, :]
        trt_c = trt[:, :min_t, :]
    else:
        ort_c = ort[:min_t, :]
        trt_c = trt[:min_t, :]
    d = ort_c.astype(np.float64) - trt_c.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "ort_shape": list(ort.shape),
        "trt_shape": list(trt.shape),
        "finite_trt": int(np.sum(np.isfinite(trt))),
        "total": int(trt.size),
    }


def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"ERROR: Engine not found: {ENGINE_PATH}")
        sys.exit(1)

    print(f"Engine: {ENGINE_PATH}")
    print(f"Models: {MODEL_DIR}")

    for label, nf in [("40f", 40), ("200f", 200), ("1000f", 1000), ("3000f", 3000)]:
        print(f"\n--- {label}: mel (1,128,{nf}) ---")
        np.random.seed(42)
        mel = np.random.randn(1, 128, nf).astype(np.float32)

        mel_path = f"/tmp/p1_mel_{label}.npy"
        out_path = f"/tmp/p1_ort_{label}.npy"
        np.save(mel_path, mel)

        try:
            run_ort(mel_path, out_path)
            ort_out = np.load(out_path)
        except Exception as e:
            print(f"  [SKIP] ORT failed: {e}")
            continue

        try:
            trt_out = run_trt(mel, ENGINE_PATH)
        except Exception as e:
            print(f"  [FAIL] TRT failed: {e}")
            continue

        r = compare(ort_out, trt_out, label)
        print(f"  ORT shape={r['ort_shape']} TRT shape={r['trt_shape']}")
        print(f"  max_abs={r['max_abs']:.6e} mean_abs={r['mean_abs']:.6e} finite={r['finite_trt']}/{r['total']}")

        try:
            os.unlink(mel_path)
            os.unlink(out_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
