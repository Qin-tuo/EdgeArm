#!/usr/bin/env python3
"""Dump ORT encoder output to .npy. Run inside Docker container.
Usage: python3 dump_ort_encoder.py /path/to/model_dir /tmp/mel.npy /tmp/ort_out.npy
"""
import sys
import os
import numpy as np
import onnxruntime as ort

model_dir, mel_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

mel = np.load(mel_path)

for name in ["encoder_fp16.onnx", "encoder.onnx"]:
    path = os.path.join(model_dir, name)
    if os.path.exists(path):
        break
else:
    raise FileNotFoundError(f"No encoder ONNX in {model_dir}")

so = ort.SessionOptions()
sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"],
                            provider_options=[{"device_id": 0}])
out = sess.run(None, {"mel": mel})
np.save(out_path, out[0])
print(f"ORT: {path} -> shape={out[0].shape} dtype={out[0].dtype}")
