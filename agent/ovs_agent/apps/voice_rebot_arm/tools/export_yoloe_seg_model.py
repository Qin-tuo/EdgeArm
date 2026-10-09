"""Export a fixed-vocabulary YOLOE-seg ONNX for the Phase B grasp segmenter.

YOLOE is open-vocabulary: ``set_classes`` bakes a chosen text class list (and
NMS) into the exported ONNX graph, so the runtime
:class:`~ovs_agent.apps.voice_rebot_arm.perception.yolo_onnx.YoloOnnxSegmenter`
runs no NMS itself and just maps row ``cls_id`` → the names you exported with.

ONNX contract the runtime expects (validated):
  * output0: ``[1, 300, 4 + 1 + 1 + 32]`` rows
    ``[x1, y1, x2, y2, conf, cls_id, *32 mask_coeffs]`` (640-letterbox xyxy)
  * output1: ``[1, 32, 160, 160]`` mask prototypes

The exported ``NAMES`` order MUST match what you pass to ``YoloOnnxSegmenter``
(and the grasp config ``yolo_classes``).

Usage:
    python export_yoloe_seg_model.py \
        --weights yoloe-26s-seg.pt \
        --classes box "cardboard box" carton package \
        --out yoloe-26s-seg-box.onnx

Real-machine note (2026-06-11, seeed-orin-nx): a box vocab exported this way
detected the placed cardboard box (conf ~0.35, CPU EP) and produced a valid
camera-frame grasp pose. Weights `yoloe-26s-seg.pt` come from the grasp repo;
they are NOT vendored here (30 MB). Run this on any host with ultralytics.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yoloe-26s-seg.pt",
                    help="YOLOE-seg .pt weights (open-vocab base).")
    ap.add_argument("--classes", nargs="+", required=True,
                    help="Text class vocabulary to bake in (order = cls_id).")
    ap.add_argument("--out", default=None,
                    help="Output .onnx path (default: <weights stem>-custom.onnx).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    from ultralytics import YOLOE

    model = YOLOE(args.weights)
    model.set_classes(args.classes)
    print("CLASSES SET:", args.classes)

    path = model.export(format="onnx", opset=args.opset, simplify=True, imgsz=args.imgsz)
    print("EXPORT PATH:", path)

    out = args.out or (os.path.splitext(os.path.basename(args.weights))[0] + "-custom.onnx")
    if os.path.abspath(path) != os.path.abspath(out):
        shutil.copyfile(path, out)

    h = hashlib.md5()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print("OUT:", out, "SIZE:", os.path.getsize(out), "MD5:", h.hexdigest())

    import onnx
    mo = onnx.load(out)
    print("=== ONNX OUTPUTS ===")
    for o in mo.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in o.type.tensor_type.shape.dim]
        print(" ", o.name, dims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
