"""Torch-free RGBD camera drivers, vendored from reBot-DevArm-Grasp.

The concrete SDKs (``pyorbbecsdk`` for Orbbec Gemini 2, ``pyrealsense2`` for
RealSense) are imported lazily inside each driver's ``open()`` / ``get_frame()``
so importing this package on a Mac without a camera SDK installed succeeds.
``make_camera`` only instantiates a driver (no ``open()``), so it is also
SDK-free until the caller opens the stream.
"""

from __future__ import annotations

from typing import Optional

from .base import CameraDriver
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera

__all__ = ["CameraDriver", "OrbbecGemini2", "RealsenseCamera", "make_camera"]


def make_camera(cfg: dict, calib_dir: Optional[str] = None) -> CameraDriver:
    """Instantiate the camera driver named by ``cfg["camera"]["type"]``.

    Args:
        cfg: config dict with a ``camera`` block (``type``, ``color_width``,
            ``color_height``, ``fps``).
        calib_dir: optional directory holding ``intrinsics.npz`` (distortion
            coefficients). When ``None`` the driver falls back to zero
            distortion and reads intrinsics live from the SDK. (Unlike the
            upstream repo there is no implicit ``config/calibration/<type>/``
            tree in this vendored package.)

    Returns:
        An unopened :class:`CameraDriver`; the caller must call ``.open()``.

    Raises:
        ValueError: when ``camera.type`` is not a supported driver.
    """
    cam_cfg = cfg.get("camera", {})
    cam_type = str(cam_cfg.get("type", "")).lower()
    w = int(cam_cfg.get("color_width", 1280))
    h = int(cam_cfg.get("color_height", 720))
    fps = int(cam_cfg.get("fps", 30))

    if "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir)
    if "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir)
    raise ValueError(
        f"unsupported camera type: {cam_type!r}\n"
        f"set camera.type to one of: "
        f"orbbec_gemini2 | realsense_d435i | realsense_d405"
    )
