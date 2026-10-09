"""so_arm.py — SOArmActuator: the SO-ARM concrete :class:`Actuator`.

Thin wrapper around the LeRobot-compatible SOFollower (in-tree
``MinimalSOFollower``, no heavy ``lerobot`` dependency):

  - Construct a SOFollower client bound to the USB serial port discovered
    by entrypoint.sh.
  - Execute a normalized sequence of ``{joints, delay}`` frames.
  - Cache the latest observation under a lock so the observation server
    can serve it without touching the serial port.

The voice pipeline is the sole owner of the serial port — verify steps
never touch it directly; they pull from the cache via the HTTP server.

This is the EXACT behaviour of the original ``robot_arm.RobotArm``,
re-homed behind the :class:`Actuator` ABC. The only semantic change is
that torque state now lives behind the public ``torque_enabled`` property
and ``set_torque`` updates it in lockstep with the physical bus (callers
no longer poke a private ``_torque_state`` field).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from ovs_agent.actuators.base import Actuator
from ovs_agent.actuators.factory import register_actuator

# Imports deferred so unit tests / static checks don't require the
# lerobot package to be installed.
try:  # pragma: no cover — runtime import guard
    # Prefer the slim in-tree MinimalSOFollower (no lerobot dependency).
    from .minimal_so_follower import (
        MinimalSOFollower as SOFollower,
    )
    from .minimal_so_follower import (
        MinimalSOFollowerConfig as SOFollowerConfig,
    )
    _LEROBOT_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        from lerobot.robots.so_follower.config_so_follower import (
            SO101FollowerConfig as SOFollowerConfig,
        )
        from lerobot.robots.so_follower.so_follower import SOFollower
        _LEROBOT_AVAILABLE = True
    except Exception:
        SOFollower = None  # type: ignore[assignment]
        SOFollowerConfig = None  # type: ignore[assignment]
        _LEROBOT_AVAILABLE = False


class SOArmActuator(Actuator):
    """Owns the SO-ARM serial connection + observation cache."""

    def __init__(
        self,
        port: str,
        arm_id: str = "voice_arm",
        move_delay: float = 1.5,
        gesture_delay: float = 0.4,
    ) -> None:
        self._port = port
        self._arm_id = arm_id
        self._move_delay = float(move_delay)
        self._gesture_delay = float(gesture_delay)
        self._robot: Optional[Any] = None
        self._latest_obs: Dict[str, Any] = {}
        self._schema: Dict[str, Any] = {}
        self._lock = threading.Lock()
        # Torque state is the single source of truth for "can we move?".
        # set_torque() updates this in lockstep with the physical bus, so
        # any caller (HTTP /test or the voice pipeline) can check it via
        # the public ``torque_enabled`` property before issuing a motion
        # command. Default "on": we assume torque-enabled at startup,
        # matching observation_server's state initialization.
        self._torque_state: str = "on"

    # ── lifecycle ────────────────────────────────────────────────────

    def _ensure_default_calibration(self) -> None:
        """Seed an identity calibration so sync_read can normalize.

        lerobot's FeetechMotorsBus refuses to read motor positions if the bus
        has no calibration registered (RuntimeError "no calibration registered").
        Proper calibration happens via a separate tool (PLAN.md Phase 0 —
        another platform handles it). For initial bring-up / read-only
        validation, we drop in an identity calibration that lets sync_read
        return raw-ish values (range_min/max = full encoder span, no offset).
        Motion commands sent against this stub are NOT safe — keep torque OFF
        as the default state until real calibration is in place.
        """
        import json
        from pathlib import Path
        # Default lerobot calibration path
        cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_path = cal_dir / f"{self._arm_id}.json"
        if cal_path.exists():
            return  # respect any real calibration the user has dropped in
        # sts3215 motors: 12-bit encoder, 0-4095 over their range.
        # In DEGREES mode the bus normalizes to [-180, 180]. Identity means
        # we treat encoder midpoint (2048) as 0°, full sweep as ±180°.
        default_motor = {"drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        cal = {
            "shoulder_pan":  {**default_motor, "id": 1},
            "shoulder_lift": {**default_motor, "id": 2},
            "elbow_flex":    {**default_motor, "id": 3},
            "wrist_flex":    {**default_motor, "id": 4},
            "wrist_roll":    {**default_motor, "id": 5},
            "gripper":       {**default_motor, "id": 6},
        }
        cal_path.write_text(json.dumps(cal, indent=2))
        print(f"[SOArmActuator] wrote stub calibration to {cal_path}")

    def connect(self) -> None:
        if not _LEROBOT_AVAILABLE:
            raise RuntimeError(
                "lerobot is not installed; cannot connect to SO-ARM."
            )
        self._ensure_default_calibration()
        # SO101FollowerConfig = RobotConfig + SOFollowerConfig (id from former,
        # port/use_degrees/etc from latter). Pass arm_id as `id` for the
        # calibration JSON lookup.
        config = SOFollowerConfig(  # type: ignore[misc]
            id=self._arm_id,
            port=self._port,
            use_degrees=True,  # actions.yaml uses degrees, not radians
        )
        self._robot = SOFollower(config)  # type: ignore[misc]
        # calibrate=False: skip lerobot's interactive `input("Move arm to middle
        # and press ENTER")` prompt — container has no stdin. Calibration is
        # handled by a separate tool (see PLAN.md Phase 0 — "标定另平台处理").
        # /observation reads still work uncalibrated; send_action with uncalibrated
        # arm may behave unexpectedly so torque-off is the default safe state.
        self._robot.connect(calibrate=False)
        # Seed schema from the lerobot feature set so the HTTP server
        # can publish it before any frame is read.
        self._schema = self._derive_schema()
        # Prime the observation cache so verify panels don't flash NaN
        # at startup.
        self.update_cache()

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[SOArmActuator] disconnect error: {exc}")
            finally:
                self._robot = None

    # ── observation cache ────────────────────────────────────────────

    def update_cache(self) -> Dict[str, Any]:
        """Read a fresh observation from the arm and update the cache.

        The lock now covers the actual serial read (not just the dict
        swap) so it can't race with execute_sequence which also drives
        the Feetech bus. Without this, concurrent access trips
        "TxRxResult Port is in use!".
        """
        if self._robot is None:
            return {}
        # SOFollower returns a flat dict {joint.pos: float, ...}
        with self._lock:
            obs = self._robot.get_observation()
            self._latest_obs = dict(obs)
        return obs

    def get_cached_observation(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_obs)

    def observation_features(self) -> Dict[str, Any]:
        """Return the schema used by GET /observation/schema."""
        with self._lock:
            return dict(self._schema)

    # ── action dispatch ──────────────────────────────────────────────

    def execute_action(
        self,
        name: str,
        actions_map: Dict[str, Any],
    ) -> bool:
        """Look up `name` in actions.yaml content and dispatch.

        actions_map is expected to be `{"sequences": {<name>: [frames]}}`
        (the unified schema; single-frame poses are 1-frame sequences).
        Returns True if the action was found and sent, False otherwise.

        Safety: refuses to dispatch when torque is off. A relaxed arm
        plus motion commands risks the arm flopping out of position;
        callers should re-enable torque via /torque/on first.
        """
        if not name or name == "none":
            return False
        if self._robot is None:
            print(f"[SOArmActuator] No arm connected, dropping action {name!r}")
            return False
        if not self.torque_enabled:
            # Voice pipeline ends up here when the user disabled torque
            # via the verify panel and then issued a voice command.
            # Log loudly so the operator notices in container logs.
            print(
                f"[SOArmActuator] REFUSING action {name!r}: torque is "
                f"{self._torque_state!r}. Enable torque first.",
            )
            return False

        sequences = (actions_map or {}).get("sequences") or {}
        frames = sequences.get(name)
        if frames is None:
            print(f"[SOArmActuator] action {name!r} not found in actions.yaml")
            return False
        return self.execute_sequence(frames)

    def execute_sequence(
        self,
        frames: list,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Execute a normalized sequence (list of {joints, delay} frames).

        If `cancel_event` is provided and fires mid-sequence, the loop
        breaks early. Always refreshes the observation cache when done.
        """
        if self._robot is None or not frames:
            return False
        # Hold the serial-bus lock for the whole sequence. Option (A):
        # observation cache will be stale for the (typically <5s) duration
        # of a motion, but readers already tolerate stale frames. The
        # alternative (per-frame lock acquire/release) lets update_cache
        # interleave but adds complexity for marginal gain.
        with self._lock:
            for frame in frames:
                if cancel_event is not None and cancel_event.is_set():
                    break
                joints = frame.get("joints") if isinstance(frame, dict) else None
                if not isinstance(joints, dict):
                    continue
                self._send_frame(joints)
                delay = float(frame.get("delay", self._gesture_delay)) if isinstance(frame, dict) else self._gesture_delay
                time.sleep(delay)
            # Refresh cache while we still hold the lock — saves a
            # second acquire and guarantees post-sequence obs is fresh.
            try:
                obs = self._robot.get_observation()
                self._latest_obs = dict(obs)
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[SOArmActuator] update_cache after sequence failed: {exc}")
        return True

    # ── torque control ───────────────────────────────────────────────

    def set_torque(self, enable: bool) -> None:
        """Enable or disable joint torque (whole arm).

        Drives the physical bus AND records the new state on
        ``torque_enabled`` so the voice pipeline + HTTP server share a
        single source of truth. Raises RuntimeError if no LeRobot client
        is connected.
        """
        if self._robot is None:
            raise RuntimeError("arm not connected")
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            raise RuntimeError("arm has no .bus attribute (lerobot version mismatch?)")
        # Serialize against update_cache / execute_sequence on the same bus.
        with self._lock:
            if enable:
                enable_fn = getattr(bus, "enable_torque", None)
                if callable(enable_fn):
                    enable_fn()
                    self._torque_state = "on"
                    return
            else:
                disable_fn = getattr(bus, "disable_torque", None)
                if callable(disable_fn):
                    disable_fn()
                    self._torque_state = "off"
                    return
        raise RuntimeError("bus does not expose enable_torque/disable_torque")

    @property
    def torque_enabled(self) -> bool:
        """Whether joint torque is currently enabled (public read)."""
        return self._torque_state == "on"

    # ── internals ────────────────────────────────────────────────────

    def _send_frame(self, frame: Dict[str, float]) -> None:
        """Send one joint-angle frame to the arm."""
        if self._robot is None:
            return
        # SOFollower.send_action accepts a partial dict — joints not
        # present keep their last commanded value.
        self._robot.send_action({k: float(v) for k, v in frame.items()})

    def _derive_schema(self) -> Dict[str, Any]:
        """Best-effort schema derivation from SOFollower.observation_features."""
        if self._robot is None:
            return {}
        features: Iterable[str] = []
        try:
            features = getattr(self._robot, "observation_features", None) or []
            if callable(features):  # in case it's a method on this lerobot version
                features = features()
        except Exception:  # pragma: no cover
            features = []
        schema: Dict[str, Any] = {}
        for field in features:
            schema[str(field)] = {"type": "float"}
        return schema


def _make_so_arm(config: dict) -> Actuator:
    """Build a :class:`SOArmActuator` from the actuator config dict.

    Accepts the historical ``arm_port`` alias for ``port`` through the
    ``metadata.arm`` compat window.
    """
    port = config.get("port", config.get("arm_port"))
    if port is None:
        raise ValueError("so_arm actuator requires a 'port' in config")
    return SOArmActuator(
        port=port,
        arm_id=config.get("arm_id", "voice_arm"),
        move_delay=float(config.get("move_delay", 1.5)),
        gesture_delay=float(config.get("gesture_delay", 0.4)),
    )


# Self-register with the framework on import so the factory can build us by
# name. The owning app (apps/voice_arm) imports this module, which keeps the
# framework driver-agnostic.
register_actuator("so_arm", _make_so_arm)


__all__ = ["SOArmActuator", "_make_so_arm"]
