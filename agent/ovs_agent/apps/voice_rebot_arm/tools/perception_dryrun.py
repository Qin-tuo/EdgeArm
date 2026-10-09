"""Phase B perception dry-run (camera-frame only, NO arm/serial/RebotArm).

1 frame: Orbbec color(+depth) -> YoloOnnxSegmenter (TRT/CUDA/CPU EP) ->
ordinary_grasp.estimate_grasps -> print + save visualization jpg. Exits.
"""
from __future__ import annotations
import os
import sys
import time
import traceback

import numpy as np
import cv2

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = "/home/seeed/perception_dryrun"
sys.path.insert(0, HERE)

from perception.yolo_onnx import YoloOnnxSegmenter, DEFAULT_PROVIDERS  # noqa: E402
from perception.ordinary_grasp import estimate_grasps, detection_count  # noqa: E402

MODEL = os.path.join(HERE, "yoloe-26s-seg.onnx")
# Vocabulary order MUST match export (step2_export.py: set_classes(["person","bus"])).
NAMES = ["person", "bus"]
OUT_JPG = os.path.join(HERE, "dryrun_vis.jpg")


def capture_one_frame():
    """Open Orbbec, grab ONE color(+depth) frame, close. Returns (bgr, depth_mm, K)."""
    import pyorbbecsdk as ob

    color_bgr = None
    depth_mm = None
    K = None

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    pipeline = None
    try:
        try:
            ob.Context().set_logger_severity(ob.OBLogSeverity.FATAL)
        except Exception:
            pass
        pipeline = ob.Pipeline()
        cfg = ob.Config()

        # color stream
        plist = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        cp = None
        for fmt in (ob.OBFormat.MJPG, ob.OBFormat.RGB):
            try:
                cp = plist.get_video_stream_profile(1280, 720, fmt, 30)
                break
            except Exception:
                pass
        if cp is None:
            cp = plist.get_default_video_stream_profile()
        cfg.enable_stream(cp)

        # depth stream
        try:
            dplist = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            try:
                dp = dplist.get_video_stream_profile(1280, 720, ob.OBFormat.Y16, 30)
            except Exception:
                dp = dplist.get_default_video_stream_profile()
            cfg.enable_stream(dp)
        except Exception:
            pass

        try:
            cfg.set_align_mode(ob.OBAlignMode.HW_MODE)
        except Exception:
            pass

        pipeline.start(cfg)

        # intrinsics
        try:
            intr = pipeline.get_camera_param().rgb_intrinsic
            K = np.array(
                [[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
                dtype=np.float64,
            )
        except Exception:
            pass

        # warm a few frames so auto-exposure/depth settle, take the last good one
        deadline = time.time() + 5.0
        while time.time() < deadline:
            frames = pipeline.wait_for_frames(500)
            if frames is None:
                continue
            cf = frames.get_color_frame()
            if cf is None:
                continue
            w, h = cf.get_width(), cf.get_height()
            raw = np.asanyarray(cf.get_data(), dtype=np.uint8)
            fmt = cf.get_format()
            try:
                if fmt == ob.OBFormat.MJPG:
                    color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                elif fmt == ob.OBFormat.RGB:
                    color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                else:
                    color_bgr = raw.reshape(h, w, 3)
            except Exception:
                color_bgr = None
            df = frames.get_depth_frame()
            if df is not None:
                try:
                    dw, dh = df.get_width(), df.get_height()
                    depth_mm = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
                except Exception:
                    depth_mm = None
            if color_bgr is not None:
                break
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass
        os.dup2(saved, 2)
        os.close(saved)
    return color_bgr, depth_mm, K


def main() -> int:
    print("=== Phase B perception dry-run ===")
    print("model:", MODEL, "exists:", os.path.exists(MODEL))

    # 1) capture
    t0 = time.time()
    color_bgr, depth_mm, K = capture_one_frame()
    print(f"[capture] took {time.time()-t0:.2f}s")
    if color_bgr is None:
        print("[capture] FAILED: no color frame")
        return 2
    print(f"[capture] color shape={color_bgr.shape} dtype={color_bgr.dtype}")
    if depth_mm is not None:
        print(
            f"[capture] depth shape={depth_mm.shape} dtype={depth_mm.dtype} "
            f"min={int(depth_mm.min())} max={int(depth_mm.max())} "
            f"nonzero={int(np.count_nonzero(depth_mm))}"
        )
    else:
        print("[capture] depth: NONE")
    if K is not None:
        print(f"[capture] K=\n{K}")

    h0, w0 = color_bgr.shape[:2]
    if depth_mm is None:
        depth_mm = np.zeros((h0, w0), dtype=np.uint16)
        print("[capture] using zero depth fallback (grasp 3D pose will be rejected)")
    if K is None:
        # crude pinhole fallback
        K = np.array([[w0, 0, w0 / 2], [0, w0, h0 / 2], [0, 0, 1]], dtype=np.float64)
        print("[capture] using fallback K (no SDK intrinsics)")

    # align depth to color size if needed
    if depth_mm.shape[:2] != (h0, w0):
        depth_mm = cv2.resize(depth_mm, (w0, h0), interpolation=cv2.INTER_NEAREST)

    # 2) detection
    # EP override: OVS_ORT_PROVIDERS=cpu avoids the memory-heavy TRT/CUDA engine
    # build (production voice stack leaves <300MB free on this Orin NX -> OOM).
    ep_env = os.environ.get("OVS_ORT_PROVIDERS", "").strip().lower()
    if ep_env == "cpu":
        providers = ("CPUExecutionProvider",)
    elif ep_env == "cuda":
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    else:
        providers = DEFAULT_PROVIDERS
    print("[detect] requested providers:", providers)
    seg = YoloOnnxSegmenter(MODEL, NAMES, providers=providers)
    t1 = time.time()
    results = seg.predict(color_bgr, conf=0.25)
    print(f"[detect] inference took {time.time()-t1:.2f}s")
    sess = seg._session
    print("[detect] EP actually used:", sess.get_providers() if sess else "n/a")

    r = results[0]
    n = detection_count(r)
    print(f"[detect] detections: {n}")
    for i in range(n):
        b = r.boxes[i]
        cls_id = int(np.asarray(b.cls[0]).reshape(-1)[0])
        conf = float(np.asarray(b.conf[0]).reshape(-1)[0])
        xyxy = [int(v) for v in np.asarray(b.xyxy[0])[:4]]
        label = r.names.get(cls_id, str(cls_id))
        print(f"  [{i}] cls={cls_id}({label}) conf={conf:.3f} bbox={xyxy}")

    # 3) grasps (camera frame)
    grasps = estimate_grasps([r], depth_mm, K)
    print(f"[grasp] {len(grasps)} grasp poses")
    for i, g in enumerate(grasps):
        print(f"  grasp[{i}] class={g.class_name} conf={g.conf:.3f} "
              f"valid={g.is_valid} reject={g.rejected_reason}")
        print(f"           center_px={g.center_px} bbox={g.bbox_xyxy} "
              f"angle_deg={g.angle_deg:.1f}")
        print(f"           position(cam,m)={None if g.position is None else g.position.tolist()}")
        print(f"           jaw_width_m={g.jaw_width_m:.4f} object_length_m={g.object_length_m:.4f} "
              f"valid_depth_px={g.valid_depth_pixels}")
        if g.rotation is not None:
            print(f"           rotation(cam)=\n{g.rotation}")

    # 4) visualization
    vis = color_bgr.copy()
    for i in range(n):
        b = r.boxes[i]
        x1, y1, x2, y2 = [int(v) for v in np.asarray(b.xyxy[0])[:4]]
        cls_id = int(np.asarray(b.cls[0]).reshape(-1)[0])
        conf = float(np.asarray(b.conf[0]).reshape(-1)[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{r.names.get(cls_id, cls_id)} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if r.masks is not None:
        for m in r.masks.data:
            mm = (np.asarray(m) > 0.5).astype(np.uint8)
            if mm.shape[:2] != (h0, w0):
                mm = cv2.resize(mm, (w0, h0), interpolation=cv2.INTER_NEAREST)
            overlay = vis.copy()
            overlay[mm > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    for g in grasps:
        pts = np.round(g.short_edge_points).astype(int)
        if pts.shape == (2, 2):
            cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (255, 0, 255), 3)
        cv2.circle(vis, g.center_px, 5, (255, 255, 0), -1)
    cv2.imwrite(OUT_JPG, vis)
    print(f"[vis] saved {OUT_JPG} exists={os.path.exists(OUT_JPG)} "
          f"size={os.path.getsize(OUT_JPG) if os.path.exists(OUT_JPG) else 0}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
