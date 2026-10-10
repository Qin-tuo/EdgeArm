"""SOArmActuator adapted from OVS: one bus, explicit enable, real calibration.

The in-tree follower and Actuator/factory contracts are reused. Cancellation
sets an event before taking the bus lock. Reconnect never enables motion.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path

from ovs_agent.actuators.base import Actuator, ActuatorClosed
from ovs_agent.actuators.factory import register_actuator
from .minimal_so_follower import (
    MinimalSOFollower, MinimalSOFollowerConfig, SO_FOLLOWER_MOTORS, load_calibration,
)


class SOArmActuator(Actuator):
    def __init__(self, port: str, calibration_dir: str, arm_id: str = "so101",
                 max_speed: float = 10.0, feedback_max_age_s: float = 0.5,
                 calibration_sha256: str = "") -> None:
        if not port.startswith("/dev/serial/by-id/"):
            raise ValueError("use a stable /dev/serial/by-id/ SO-101 port")
        if not math.isfinite(max_speed) or not 0 < max_speed <= 10:
            raise ValueError("max_speed must be within (0, 10]")
        if not math.isfinite(feedback_max_age_s) or feedback_max_age_s <= 0:
            raise ValueError("feedback_max_age_s must be positive")
        if not arm_id or Path(arm_id).name != arm_id or arm_id in (".", ".."):
            raise ValueError("invalid arm_id")
        self.config = MinimalSOFollowerConfig(
            port=port, id=arm_id, calibration_dir=Path(calibration_dir), use_degrees=True,
        )
        self.max_speed = max_speed
        self.feedback_max_age_s = feedback_max_age_s
        self.calibration_sha256 = calibration_sha256
        self._robot = None
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._gate_lock = threading.Lock()
        self._stop_generation = 0
        self._closed = False
        self._motion_enabled = False
        self._torque_state = "off"
        self._latest_obs = {}
        self._observed_at = 0.0

    def connect(self) -> None:
        with self._lock:
            if self._closed:
                raise ActuatorClosed("actuator has shut down")
            if self._robot is not None:
                return
            robot = MinimalSOFollower(self.config)
            try:
                robot.connect(calibrate=False)
                self._robot = robot
                self.update_cache()
            except BaseException:
                robot.disconnect()
                self._robot = None
                raise
            self._motion_enabled = False
            self._torque_state = "off"

    def disconnect(self) -> None:
        self._closed = True
        self.request_stop()
        with self._lock:
            if self._robot is not None:
                try:
                    self._robot.disconnect()
                finally:
                    self._robot = None
                    self._torque_state = "off"
                    self._latest_obs = {}
                    self._observed_at = 0.0

    def update_cache(self) -> dict:
        with self._lock:
            if self._robot is None:
                return {}
            started = time.monotonic()
            obs = self._robot.get_observation()
            if time.monotonic() - started > self.feedback_max_age_s:
                self.request_stop()
                raise RuntimeError("joint feedback exceeded age budget")
            if set(obs) != set(self.observation_features()) or any(not math.isfinite(v) for v in obs.values()):
                raise RuntimeError("incomplete or invalid joint feedback")
            self._latest_obs = dict(obs)
            self._observed_at = time.monotonic()
            return dict(obs)

    def get_cached_observation(self) -> dict:
        with self._lock:
            return dict(self._latest_obs)

    def observation_features(self) -> dict:
        return {f"{name}.pos": {"type": "float", "unit": "percent" if name == "gripper" else "degree"}
                for name in SO_FOLLOWER_MOTORS}

    def feedback(self) -> dict:
        with self._lock:
            return {"joints": dict(self._latest_obs), "sampled_at": self._observed_at,
                    "age_s": time.monotonic() - self._observed_at if self._observed_at else None,
                    "motion_enabled": self._motion_enabled, "torque": self._torque_state}

    def set_torque(self, enable: bool) -> None:
        # Only the local operator path calls this; it is never an LLM tool.
        if type(enable) is not bool:
            raise ValueError("enable must be a boolean")
        if not enable:
            self.request_stop()
        with self._gate_lock:
            # Failed/repeated enables cannot leave an earlier grant active.
            self._motion_enabled = False
            ticket = self._stop_generation
        with self._lock:
            robot = self._robot
            if robot is None or self._closed:
                raise RuntimeError("arm unavailable")
            if enable:
                path = self.config.calibration_dir / f"{self.config.id}.json"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if not self.calibration_sha256 or digest != self.calibration_sha256:
                    raise RuntimeError("operator-reviewed calibration hash is required")
                if load_calibration(self.config.id, self.config.calibration_dir) != robot.calibration:
                    raise RuntimeError("calibration changed; restart with motion disabled")
                obs = self.update_cache()
                robot.send_action(obs)  # Stage current positions before enabling torque.
                try:
                    robot.bus.enable_torque()
                except BaseException:
                    self._torque_state = "unknown"
                    raise
                self._torque_state = "on"
                with self._gate_lock:
                    if ticket != self._stop_generation or self._closed:
                        self._motion_enabled = False
                        return
                    self._cancel.clear()
                    self._motion_enabled = True
            else:
                try:
                    robot.bus.disable_torque()
                    self._torque_state = "off"
                except BaseException:
                    self._torque_state = "unknown"
                    raise

    @property
    def torque_enabled(self) -> bool:
        return self._torque_state == "on"

    def request_stop(self) -> None:
        with self._gate_lock:
            self._stop_generation += 1
            self._cancel.set()
            self._motion_enabled = False

    def stop_motion(self) -> dict:
        self.request_stop()
        with self._lock:
            if self._robot is None:
                return {"status": "unavailable", "confirmed": False}
            if not self.torque_enabled:
                return {"status": "motion_disabled", "confirmed": False, "torque": self._torque_state}
            obs = self.update_cache()
            self._robot.send_action(obs)  # Hold all joints, including the gripper.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            with self._lock:
                current = self.update_cache()
                stable = all(abs(current[k] - obs[k]) <= 1.0 for k in obs)
                obs = current
            if stable:
                return {"status": "hold_feedback_stable", "confirmed": True, "hardware_validated": False}
        return {"status": "hold_unconfirmed", "confirmed": False}

    def execute_sequence(self, frames: list[dict], *, cancel_event=None) -> bool:
        if not frames or len(frames) > 128:
            raise ValueError("sequence must contain 1..128 frames")
        normalized = []
        for frame in frames:
            joints, delay = frame.get("joints"), frame.get("delay")
            if not isinstance(joints, dict) or set(joints) != set(self.observation_features()):
                raise ValueError("each frame requires all six joint targets")
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0.05 <= delay <= 2:
                raise ValueError("frame delay must be within [0.05, 2] seconds")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in joints.values()):
                raise ValueError("joint targets must be finite numbers")
            normalized.append((dict(joints), float(delay)))
        with self._lock:
            if not self._motion_enabled or self._robot is None or self._closed:
                return False
            previous = self.update_cache()
            for joints, delay in normalized:
                if any(abs(joints[k] - previous[k]) / delay > self.max_speed for k in joints):
                    raise ValueError("sequence exceeds configured joint/gripper speed bound")
                self._robot.validate_action(joints)
                previous = joints
        for joints, delay in normalized:
            with self._lock:
                if self._cancel.is_set() or (cancel_event and cancel_event.is_set()) or not self._motion_enabled:
                    return False
                observed = self.update_cache()
                if any(abs(joints[k] - observed[k]) / delay > self.max_speed for k in joints):
                    raise ValueError("joint feedback deviates from sequence speed bound")
                self._robot.send_action(joints)
            if self._cancel.wait(delay) or (cancel_event and cancel_event.is_set()):
                return False
        self.update_cache()
        return True  # Dispatch is separate from verified grasp/place success.


def _make_so_arm(config: dict) -> SOArmActuator:
    return SOArmActuator(**config)


register_actuator("so_arm", _make_so_arm)
