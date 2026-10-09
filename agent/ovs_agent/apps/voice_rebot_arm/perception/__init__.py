"""Torch-free vision-grasp perception for the reBot B601-DM voice app.

Phase B. Vendored from ``reBot-DevArm-Grasp`` with the torch / ultralytics
runtime dependency removed: YOLOE-seg inference runs on onnxruntime
(TensorRT EP on Jetson, CPU EP on Mac for tests) and the seg post-process
is reimplemented in pure numpy / cv2 (see :mod:`yolo_onnx`).

Heavy / device-only imports (onnxruntime, camera SDKs) are deferred so this
package imports on a Mac without those wheels installed.
"""

from __future__ import annotations

__all__ = [
    "YoloOnnxSegmenter",
    "YoloResult",
    "estimate_grasps",
    "select_best_grasp",
    "transform_grasp_pose_to_base",
]


def __getattr__(name: str):  # noqa: D401 - lazy re-export
    # Lazy attribute access keeps ``import ...perception`` cheap and
    # SDK-free; the submodules pull cv2/numpy (always present) but never
    # torch.
    if name in ("YoloOnnxSegmenter", "YoloResult"):
        from .yolo_onnx import YoloOnnxSegmenter, YoloResult

        return {"YoloOnnxSegmenter": YoloOnnxSegmenter, "YoloResult": YoloResult}[name]
    if name in ("estimate_grasps", "select_best_grasp"):
        from .ordinary_grasp import estimate_grasps, select_best_grasp

        return {"estimate_grasps": estimate_grasps, "select_best_grasp": select_best_grasp}[name]
    if name == "transform_grasp_pose_to_base":
        from .transforms import transform_grasp_pose_to_base

        return transform_grasp_pose_to_base
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
