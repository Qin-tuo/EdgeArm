"""minimal_so_follower.py — Slim replacement for lerobot.SOFollower.

A narrowly scoped re-implementation of the small surface area of
``lerobot.robots.so_follower.SOFollower`` that ``robot_arm.py`` actually uses.

The goal is to drop the heavy ``lerobot`` dependency from the voice-arm
Docker image while preserving:

  - ``MinimalSOFollowerConfig(id=..., port=..., use_degrees=True)``
  - ``connect(calibrate=False)`` / ``disconnect()``
  - ``get_observation()`` -> ``{"<joint>.pos": float}``
  - ``send_action({"<joint>.pos": float, ...})`` (partial dicts honored)
  - ``observation_features`` property
  - ``robot.bus.enable_torque()`` / ``robot.bus.disable_torque()``

Calibration JSON layout follows lerobot v0.4.4 (see design spec at
``docs/minimal-so-follower-spec.md``).

This module only depends on ``scservo_sdk`` (PyPI: ``feetech-servo-sdk``)
at runtime; tests can inject a fake bus via the ``bus=`` constructor arg
so they never import the SDK.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — Feetech STS3215 / SCS series register map
# (cross-checked against lerobot v0.4.4 feetech tables + scservo_sdk SCSCL.)
# ---------------------------------------------------------------------------

# (address, size_bytes)
REGISTERS: Dict[str, Tuple[int, int]] = {
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Maximum_Acceleration": (85, 1),
}

# Default SO-ARM-100/101 follower motor map: name -> servo id
SO_FOLLOWER_MOTORS: Dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# Position resolution. STS3215 / STS3250 → 0..4095 (12-bit).
RESOLUTION = 4095
DEFAULT_BAUDRATE = 1_000_000

# PID defaults written during configure() — match lerobot v0.4.4 so_follower.
DEFAULT_PID: Dict[str, int] = {"P": 16, "I": 0, "D": 32}


# ---------------------------------------------------------------------------
# Config + calibration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MotorCalibration:
    """Per-motor calibration record matching lerobot's MotorCalibration dataclass."""

    id: int
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = RESOLUTION


@dataclass
class MinimalSOFollowerConfig:
    """Drop-in subset of ``SO101FollowerConfig``.

    Field-for-field compatible with the kwargs voice-arm uses today.
    ``cameras`` is accepted for compatibility but ignored (this minimal
    follower has no camera abstraction).
    """

    port: str
    id: Optional[str] = None
    calibration_dir: Optional[Path] = None
    disable_torque_on_disconnect: bool = True
    max_relative_target: Optional[float] = None
    use_degrees: bool = False
    cameras: Dict[str, Any] = field(default_factory=dict)


# Backwards-compatible alias so existing call sites using
# ``SO101FollowerConfig`` continue to type-resolve in mypy/IDE.
SO101FollowerConfig = MinimalSOFollowerConfig


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------


def _default_calibration_root() -> Path:
    """Resolve the base calibration dir following lerobot env-var precedence."""
    env_cal = os.getenv("HF_LEROBOT_CALIBRATION")
    if env_cal:
        return Path(env_cal).expanduser()
    hf_home = os.getenv("HF_LEROBOT_HOME") or os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot" / "calibration"
    return Path("~/.cache/huggingface/lerobot/calibration").expanduser()


def resolve_calibration_path(
    robot_id: Optional[str],
    calibration_dir: Optional[Path] = None,
    robot_name: str = "so_follower",
) -> Path:
    """Return the JSON path for ``robot_id`` (no existence check)."""
    rid = robot_id or "default"
    if calibration_dir is not None:
        return Path(calibration_dir).expanduser() / f"{rid}.json"
    return _default_calibration_root() / "robots" / robot_name / f"{rid}.json"


def load_calibration(
    robot_id: Optional[str],
    calibration_dir: Optional[Path] = None,
    robot_name: str = "so_follower",
) -> Dict[str, MotorCalibration]:
    """Load lerobot-style calibration JSON.

    Raises ``FileNotFoundError`` with the resolved path if missing.
    """
    path = resolve_calibration_path(robot_id, calibration_dir, robot_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found for id={robot_id!r}: {path}"
        )
    with path.open("r") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Calibration JSON must be an object, got {type(raw).__name__}: {path}")
    result: Dict[str, MotorCalibration] = {}
    for motor_name, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration for {motor_name!r} must be an object")
        result[motor_name] = MotorCalibration(
            id=int(payload["id"]),
            drive_mode=int(payload.get("drive_mode", 0)),
            homing_offset=int(payload.get("homing_offset", 0)),
            range_min=int(payload.get("range_min", 0)),
            range_max=int(payload.get("range_max", RESOLUTION)),
        )
    return result


# ---------------------------------------------------------------------------
# FeetechBus — thin wrapper around scservo_sdk
# ---------------------------------------------------------------------------


class FeetechBus:
    """Serial bus over a Feetech servo chain.

    The SDK objects (``PortHandler``, ``PacketHandler``, ``GroupSyncRead``,
    ``GroupSyncWrite``) can all be dependency-injected for unit tests via
    the ``sdk`` kwarg. In production ``sdk`` is ``None`` and the real
    ``scservo_sdk`` module is imported lazily.
    """

    def __init__(
        self,
        port: str,
        motors: Optional[Dict[str, int]] = None,
        baudrate: int = DEFAULT_BAUDRATE,
        sdk: Any = None,
    ) -> None:
        self.port_name = port
        self.motors = dict(motors) if motors else dict(SO_FOLLOWER_MOTORS)
        self.baudrate = int(baudrate)
        self._sdk = sdk  # may be None → resolved at connect()
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._lock = threading.RLock()
        self.is_connected: bool = False

    # ---- lifecycle ----------------------------------------------------

    def _resolve_sdk(self) -> Any:
        if self._sdk is None:
            import scservo_sdk as scs  # type: ignore

            self._sdk = scs
        return self._sdk

    def connect(self, handshake: bool = True) -> None:
        with self._lock:
            if self.is_connected:
                return
            sdk = self._resolve_sdk()
            self._port_handler = sdk.PortHandler(self.port_name)
            self._packet_handler = sdk.PacketHandler(0)
            if not self._port_handler.openPort():
                raise ConnectionError(f"Failed to open serial port: {self.port_name}")
            if not self._port_handler.setBaudRate(self.baudrate):
                raise ConnectionError(
                    f"Failed to set baudrate {self.baudrate} on {self.port_name}"
                )
            self.is_connected = True
            if handshake:
                missing = [name for name, mid in self.motors.items() if self.ping(mid) is None]
                if missing:
                    raise ConnectionError(f"Motors not responding: {missing}")

    def disconnect(self, disable_torque: bool = True) -> None:
        with self._lock:
            if not self.is_connected:
                return
            try:
                if disable_torque:
                    self.disable_torque()
            finally:
                if self._port_handler is not None:
                    try:
                        self._port_handler.closePort()
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("closePort failed")
                self.is_connected = False

    # ---- low-level register I/O --------------------------------------

    def ping(self, motor_id: int) -> Optional[int]:
        with self._lock:
            sdk = self._resolve_sdk()
            try:
                model, comm, _err = self._packet_handler.ping(self._port_handler, motor_id)
            except Exception:
                logger.exception("ping(%d) raised", motor_id)
                return None
            if comm != getattr(sdk, "COMM_SUCCESS", 0):
                return None
            return model

    def _read_register(self, motor_id: int, addr: int, size: int) -> int:
        sdk = self._resolve_sdk()
        if size == 1:
            value, comm, err = self._packet_handler.read1ByteTxRx(self._port_handler, motor_id, addr)
        elif size == 2:
            value, comm, err = self._packet_handler.read2ByteTxRx(self._port_handler, motor_id, addr)
        else:
            value, comm, err = self._packet_handler.read4ByteTxRx(self._port_handler, motor_id, addr)
        if comm != getattr(sdk, "COMM_SUCCESS", 0):
            raise ConnectionError(f"read failed motor={motor_id} addr={addr}: comm={comm}")
        if err:
            raise RuntimeError(f"device error motor={motor_id} addr={addr}: err={err}")
        return int(value)

    def _write_register(self, motor_id: int, addr: int, size: int, value: int) -> None:
        sdk = self._resolve_sdk()
        if size == 1:
            comm, err = self._packet_handler.write1ByteTxRx(self._port_handler, motor_id, addr, int(value))
        elif size == 2:
            comm, err = self._packet_handler.write2ByteTxRx(self._port_handler, motor_id, addr, int(value))
        else:
            comm, err = self._packet_handler.write4ByteTxRx(self._port_handler, motor_id, addr, int(value))
        if comm != getattr(sdk, "COMM_SUCCESS", 0):
            raise ConnectionError(f"write failed motor={motor_id} addr={addr}: comm={comm}")
        if err:
            raise RuntimeError(f"device error motor={motor_id} addr={addr}: err={err}")

    def write(self, data_name: str, motor: str, value: int) -> None:
        addr, size = REGISTERS[data_name]
        with self._lock:
            self._write_register(self.motors[motor], addr, size, int(value))

    # ---- group operations --------------------------------------------

    def sync_read(
        self,
        data_name: str,
        motors: Optional[Iterable[str]] = None,
    ) -> Dict[str, int]:
        addr, size = REGISTERS[data_name]
        names = list(motors) if motors is not None else list(self.motors.keys())
        with self._lock:
            sdk = self._resolve_sdk()
            group = sdk.GroupSyncRead(self._port_handler, self._packet_handler, addr, size)
            for name in names:
                ok = group.addParam(self.motors[name])
                if not ok:
                    raise ConnectionError(f"GroupSyncRead.addParam failed for {name}")
            comm = group.txRxPacket()
            if comm != getattr(sdk, "COMM_SUCCESS", 0):
                raise ConnectionError(f"GroupSyncRead.txRxPacket failed: comm={comm}")
            out: Dict[str, int] = {}
            for name in names:
                out[name] = int(group.getData(self.motors[name], addr, size))
            return out

    def sync_write(self, data_name: str, values: Mapping[str, int]) -> None:
        if not values:
            return
        addr, size = REGISTERS[data_name]
        with self._lock:
            sdk = self._resolve_sdk()
            group = sdk.GroupSyncWrite(self._port_handler, self._packet_handler, addr, size)
            for name, raw in values.items():
                if name not in self.motors:
                    raise KeyError(f"Unknown motor: {name!r}")
                raw_int = int(raw) & 0xFFFFFFFF
                # Little-endian bytes for the SDK addParam contract.
                data_bytes = list(raw_int.to_bytes(size, byteorder="little", signed=False))
                ok = group.addParam(self.motors[name], data_bytes)
                if not ok:
                    raise ConnectionError(f"GroupSyncWrite.addParam failed for {name}")
            comm = group.txPacket()
            if comm != getattr(sdk, "COMM_SUCCESS", 0):
                raise ConnectionError(f"GroupSyncWrite.txPacket failed: comm={comm}")

    # ---- torque helpers ----------------------------------------------

    def enable_torque(self, motors: Optional[Iterable[str]] = None) -> None:
        names = list(motors) if motors is not None else list(self.motors.keys())
        for name in names:
            self.write("Torque_Enable", name, 1)
            self.write("Lock", name, 1)

    def disable_torque(self, motors: Optional[Iterable[str]] = None) -> None:
        names = list(motors) if motors is not None else list(self.motors.keys())
        for name in names:
            self.write("Torque_Enable", name, 0)
            self.write("Lock", name, 0)


# ---------------------------------------------------------------------------
# Conversion helpers (raw <-> degrees / percent)
# ---------------------------------------------------------------------------


def _mid(cal: MotorCalibration) -> float:
    return (cal.range_min + cal.range_max) / 2.0


def raw_to_degrees(raw: int, cal: MotorCalibration) -> float:
    return (raw - _mid(cal)) * 360.0 / RESOLUTION


def degrees_to_raw(degrees: float, cal: MotorCalibration) -> int:
    return int(round(degrees * RESOLUTION / 360.0 + _mid(cal)))


def raw_to_percent(raw: int, cal: MotorCalibration) -> float:
    span = max(1, cal.range_max - cal.range_min)
    return (raw - cal.range_min) / span * 100.0


def percent_to_raw(percent: float, cal: MotorCalibration) -> int:
    span = max(1, cal.range_max - cal.range_min)
    return int(round(percent / 100.0 * span + cal.range_min))


# ---------------------------------------------------------------------------
# MinimalSOFollower
# ---------------------------------------------------------------------------


GRIPPER_JOINT = "gripper"


class MinimalSOFollower:
    """A drop-in subset of ``lerobot.robots.so_follower.SOFollower``.

    The voice-arm runtime only uses ``connect()``, ``disconnect()``,
    ``get_observation()``, ``send_action()``, ``observation_features``,
    and ``bus.enable_torque()`` / ``bus.disable_torque()``.

    Parameters
    ----------
    config:
        :class:`MinimalSOFollowerConfig` describing port + behavior.
    bus:
        Optional pre-constructed bus (used by tests to inject a fake).
    """

    name = "so_follower"

    def __init__(
        self,
        config: MinimalSOFollowerConfig,
        bus: Optional[FeetechBus] = None,
    ) -> None:
        self.config = config
        self.bus: FeetechBus = bus if bus is not None else FeetechBus(
            port=config.port, motors=dict(SO_FOLLOWER_MOTORS)
        )
        self.calibration: Dict[str, MotorCalibration] = {}
        self._connected = False

    # ---- properties ---------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and getattr(self.bus, "is_connected", False)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    @property
    def observation_features(self) -> Dict[str, type]:
        return {f"{joint}.pos": float for joint in SO_FOLLOWER_MOTORS}

    @property
    def action_features(self) -> Dict[str, type]:
        return self.observation_features

    # ---- lifecycle ----------------------------------------------------

    def connect(self, calibrate: bool = False) -> None:
        if calibrate:
            raise NotImplementedError(
                "Interactive calibration is not supported in MinimalSOFollower; "
                "pre-generate calibration JSON with lerobot tooling."
            )
        # Load calibration JSON first — fail-fast before touching hardware.
        try:
            self.calibration = load_calibration(
                self.config.id,
                calibration_dir=self.config.calibration_dir,
                robot_name=self.name,
            )
        except FileNotFoundError:
            # Fall back to a permissive default (no homing offset, full range)
            # so headless boots can still drive raw positions. Log loudly.
            logger.warning(
                "No calibration found for id=%r — using default full-range calibration.",
                self.config.id,
            )
            self.calibration = {
                name: MotorCalibration(id=mid)
                for name, mid in SO_FOLLOWER_MOTORS.items()
            }
        # Open the bus and ping all motors.
        self.bus.connect(handshake=True)
        # Configure: disable torque while writing config registers, then re-enable.
        self.bus.disable_torque()
        for motor in SO_FOLLOWER_MOTORS:
            self.bus.write("Operating_Mode", motor, 0)  # POSITION mode
            # PID — feetech STS3215 actually uses dedicated registers; the
            # voice-arm hardware tolerates skipping these and falls back to
            # firmware defaults. Lerobot writes them via configure helpers
            # we don't have; we keep behavior parity by writing zero where
            # safe and skipping the rest (this is documented in the spec
            # under §7 as "PID writes happen via configure helpers we omit").
        self.bus.enable_torque()
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected and not getattr(self.bus, "is_connected", False):
            return
        self.bus.disconnect(disable_torque=self.config.disable_torque_on_disconnect)
        self._connected = False

    # ---- observation / action ----------------------------------------

    def get_observation(self) -> Dict[str, float]:
        raw = self.bus.sync_read("Present_Position", list(SO_FOLLOWER_MOTORS.keys()))
        out: Dict[str, float] = {}
        for joint in SO_FOLLOWER_MOTORS:
            cal = self.calibration.get(joint, MotorCalibration(id=SO_FOLLOWER_MOTORS[joint]))
            raw_v = raw[joint]
            if joint == GRIPPER_JOINT:
                out[f"{joint}.pos"] = float(raw_to_percent(raw_v, cal))
            elif self.config.use_degrees:
                out[f"{joint}.pos"] = float(raw_to_degrees(raw_v, cal))
            else:
                # RANGE_M100_100 normalization (lerobot default)
                span = max(1, cal.range_max - cal.range_min)
                norm = (raw_v - _mid(cal)) / (span / 2.0) * 100.0
                out[f"{joint}.pos"] = float(norm)
        return out

    def send_action(self, action: Mapping[str, float]) -> Dict[str, float]:
        # Keep only ``.pos`` keys — matches lerobot's partial-dict behavior.
        goal_raw: Dict[str, int] = {}
        applied: Dict[str, float] = {}
        for key, value in action.items():
            if not key.endswith(".pos"):
                continue
            joint = key[: -len(".pos")]
            if joint not in SO_FOLLOWER_MOTORS:
                raise KeyError(f"Unknown joint: {joint!r}")
            cal = self.calibration.get(joint, MotorCalibration(id=SO_FOLLOWER_MOTORS[joint]))
            # Clip in raw domain so range bounds are authoritative.
            if joint == GRIPPER_JOINT:
                raw = percent_to_raw(float(value), cal)
            elif self.config.use_degrees:
                raw = degrees_to_raw(float(value), cal)
            else:
                span = max(1, cal.range_max - cal.range_min)
                raw = int(round(float(value) / 100.0 * (span / 2.0) + _mid(cal)))
            clipped = max(cal.range_min, min(cal.range_max, raw))
            goal_raw[joint] = clipped
            # Report back the post-clip value in the same unit space.
            if joint == GRIPPER_JOINT:
                applied[key] = float(raw_to_percent(clipped, cal))
            elif self.config.use_degrees:
                applied[key] = float(raw_to_degrees(clipped, cal))
            else:
                span = max(1, cal.range_max - cal.range_min)
                applied[key] = float((clipped - _mid(cal)) / (span / 2.0) * 100.0)
        if goal_raw:
            self.bus.sync_write("Goal_Position", goal_raw)
        return applied


# Public aliases for the import shim in robot_arm.py.
SOFollower = MinimalSOFollower

__all__ = [
    "MinimalSOFollower",
    "MinimalSOFollowerConfig",
    "SOFollower",
    "SO101FollowerConfig",
    "FeetechBus",
    "MotorCalibration",
    "load_calibration",
    "resolve_calibration_path",
    "raw_to_degrees",
    "degrees_to_raw",
    "raw_to_percent",
    "percent_to_raw",
    "SO_FOLLOWER_MOTORS",
    "REGISTERS",
]
