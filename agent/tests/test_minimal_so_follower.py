"""Unit tests for MinimalSOFollower.

These tests never touch real hardware nor import ``scservo_sdk``.
They use a fake bus that records ``sync_write`` calls and returns
deterministic ``sync_read`` values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from ovs_agent.apps.voice_arm.minimal_so_follower import (  # noqa: E402
    REGISTERS,
    RESOLUTION,
    SO_FOLLOWER_MOTORS,
    FeetechBus,
    MinimalSOFollower,
    MinimalSOFollowerConfig,
    MotorCalibration,
    degrees_to_raw,
    load_calibration,
    raw_to_degrees,
)

SAMPLE_CAL_JSON: Dict[str, Dict[str, int]] = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": -120, "range_min": 720,  "range_max": 3370},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset":  340, "range_min": 850,  "range_max": 3260},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset":   90, "range_min": 910,  "range_max": 3180},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": -260, "range_min": 940,  "range_max": 3140},
    "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset":    0, "range_min":   0,  "range_max": 4095},
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": -980, "range_min": 2020, "range_max": 3540},
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBus:
    """In-memory stand-in for FeetechBus — protocol-compatible with what
    ``MinimalSOFollower`` uses."""

    def __init__(self, motors: Optional[Dict[str, int]] = None) -> None:
        self.motors = dict(motors or SO_FOLLOWER_MOTORS)
        self.is_connected = False
        # Records:
        self.connect_calls: List[bool] = []
        self.disconnect_calls: List[bool] = []
        self.register_writes: List[Tuple[str, str, int]] = []
        self.sync_writes: List[Tuple[str, Dict[str, int]]] = []
        self.enable_torque_calls: int = 0
        self.disable_torque_calls: int = 0
        # Configurable read values
        self.read_values: Dict[str, int] = {name: 2000 for name in self.motors}

    def connect(self, handshake: bool = True) -> None:  # noqa: D401
        self.connect_calls.append(handshake)
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.is_connected = False

    def write(self, data_name: str, motor: str, value: int) -> None:
        self.register_writes.append((data_name, motor, int(value)))

    def sync_read(
        self,
        data_name: str,
        motors: Optional[Iterable[str]] = None,
    ) -> Dict[str, int]:
        names = list(motors) if motors is not None else list(self.motors.keys())
        return {n: self.read_values[n] for n in names}

    def sync_write(self, data_name: str, values: Mapping[str, int]) -> None:
        self.sync_writes.append((data_name, dict(values)))

    def enable_torque(self, motors: Optional[Iterable[str]] = None) -> None:
        self.enable_torque_calls += 1

    def disable_torque(self, motors: Optional[Iterable[str]] = None) -> None:
        self.disable_torque_calls += 1


@pytest.fixture
def cal_dir(tmp_path: Path) -> Path:
    """Create a temp dir holding a `voice_arm.json` calibration file."""
    p = tmp_path / "cal"
    p.mkdir()
    (p / "voice_arm.json").write_text(json.dumps(SAMPLE_CAL_JSON))
    return p


@pytest.fixture
def follower(cal_dir: Path) -> Tuple[MinimalSOFollower, FakeBus]:
    bus = FakeBus()
    cfg = MinimalSOFollowerConfig(
        id="voice_arm",
        port="/dev/null",
        calibration_dir=cal_dir,
        use_degrees=True,
    )
    return MinimalSOFollower(cfg, bus=bus), bus


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_connect_opens_port_and_loads_calibration(follower):
    f, bus = follower
    f.connect(calibrate=False)
    assert bus.connect_calls == [True]
    assert len(f.calibration) == 6
    assert set(f.calibration.keys()) == set(SO_FOLLOWER_MOTORS.keys())
    # Operating_Mode written for every joint = 6 writes
    op_writes = [w for w in bus.register_writes if w[0] == "Operating_Mode"]
    assert len(op_writes) == 6
    # Torque cycled: disable_torque then enable_torque during configure
    assert bus.disable_torque_calls >= 1
    assert bus.enable_torque_calls >= 1


def test_disconnect_disables_torque_when_flag_set(follower):
    f, bus = follower
    f.connect(calibrate=False)
    f.disconnect()
    assert bus.disconnect_calls[-1] is True

    # Now flip the flag.
    bus2 = FakeBus()
    cfg2 = MinimalSOFollowerConfig(
        id="voice_arm",
        port="/dev/null",
        calibration_dir=f.config.calibration_dir,
        disable_torque_on_disconnect=False,
        use_degrees=True,
    )
    f2 = MinimalSOFollower(cfg2, bus=bus2)
    f2.connect(calibrate=False)
    f2.disconnect()
    assert bus2.disconnect_calls[-1] is False


def test_get_observation_returns_degrees(follower):
    f, bus = follower
    f.connect(calibrate=False)
    cal = f.calibration["shoulder_pan"]
    # Drive shoulder_pan to its midpoint -> 0 degrees.
    mid = int((cal.range_min + cal.range_max) / 2)
    bus.read_values["shoulder_pan"] = mid
    obs = f.get_observation()
    assert "shoulder_pan.pos" in obs
    assert obs["shoulder_pan.pos"] == pytest.approx(0.0, abs=1e-6)
    # Try a non-midpoint value.
    bus.read_values["shoulder_pan"] = mid + 100
    obs = f.get_observation()
    assert obs["shoulder_pan.pos"] == pytest.approx(100 * 360 / RESOLUTION, abs=1e-3)
    # Gripper uses 0-100% regardless of use_degrees=True
    gcal = f.calibration["gripper"]
    bus.read_values["gripper"] = gcal.range_min
    obs = f.get_observation()
    assert obs["gripper.pos"] == pytest.approx(0.0, abs=1e-6)
    bus.read_values["gripper"] = gcal.range_max
    obs = f.get_observation()
    assert obs["gripper.pos"] == pytest.approx(100.0, abs=1e-6)


def test_send_action_partial_dict_does_not_move_omitted_joints(follower):
    f, bus = follower
    f.connect(calibrate=False)
    bus.sync_writes.clear()
    # Only command wrist_roll.
    f.send_action({"wrist_roll.pos": 10.0})
    assert len(bus.sync_writes) == 1
    data_name, values = bus.sync_writes[0]
    assert data_name == "Goal_Position"
    assert set(values.keys()) == {"wrist_roll"}
    # Non-`.pos` keys should be ignored.
    bus.sync_writes.clear()
    f.send_action({"wrist_roll.pos": 5.0, "shoulder_pan.vel": 999})
    assert len(bus.sync_writes) == 1
    _, values = bus.sync_writes[0]
    assert set(values.keys()) == {"wrist_roll"}


def test_send_action_clips_to_range(follower):
    f, bus = follower
    f.connect(calibrate=False)
    bus.sync_writes.clear()
    # 9999 degrees is way out of bounds → should clip to range_max.
    f.send_action({"shoulder_pan.pos": 9999.0})
    _, values = bus.sync_writes[-1]
    cal = f.calibration["shoulder_pan"]
    assert values["shoulder_pan"] == cal.range_max
    # Negative extreme → clipped to range_min.
    bus.sync_writes.clear()
    f.send_action({"shoulder_pan.pos": -9999.0})
    _, values = bus.sync_writes[-1]
    assert values["shoulder_pan"] == cal.range_min


def test_observation_features_matches_joint_names():
    cfg = MinimalSOFollowerConfig(port="/dev/null", id="voice_arm")
    f = MinimalSOFollower(cfg, bus=FakeBus())
    feats = f.observation_features
    assert len(feats) == 6
    for joint in SO_FOLLOWER_MOTORS:
        key = f"{joint}.pos"
        assert key in feats
        assert feats[key] is float


def test_enable_torque_writes_correct_register():
    bus = FeetechBus(port="/dev/null", sdk=MagicMock())
    # Stub the lower-level write path so we just record register writes.
    recorded: List[Tuple[str, str, int]] = []
    bus.write = lambda data_name, motor, value: recorded.append((data_name, motor, int(value)))  # type: ignore
    bus.enable_torque()
    # 6 motors * 2 writes each (Torque_Enable=1, Lock=1) = 12
    torque_writes = [(d, m, v) for (d, m, v) in recorded if d == "Torque_Enable"]
    lock_writes = [(d, m, v) for (d, m, v) in recorded if d == "Lock"]
    assert len(torque_writes) == 6
    assert len(lock_writes) == 6
    assert all(v == 1 for (_, _, v) in torque_writes)
    assert all(v == 1 for (_, _, v) in lock_writes)
    # disable_torque flips to 0
    recorded.clear()
    bus.disable_torque()
    assert all(v == 0 for (d, _, v) in recorded if d in ("Torque_Enable", "Lock"))


def test_calibration_missing_raises_file_not_found(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        load_calibration("ghost_arm", calibration_dir=empty)
    assert "ghost_arm" in str(excinfo.value)


def test_bus_mock_injectable():
    """MinimalSOFollower must accept any protocol-compatible bus."""
    bus = MagicMock(spec=FakeBus)
    bus.is_connected = False
    bus.sync_read.return_value = {name: 2000 for name in SO_FOLLOWER_MOTORS}

    cfg = MinimalSOFollowerConfig(port="/dev/null", id="voice_arm", use_degrees=True)
    f = MinimalSOFollower(cfg, bus=bus)
    # Skip real calibration JSON: trigger the FileNotFoundError fallback path.
    f.connect(calibrate=False)
    bus.connect.assert_called_once_with(handshake=True)
    # Operating mode written 6 times
    op_writes = [c for c in bus.write.call_args_list if c.args[0] == "Operating_Mode"]
    assert len(op_writes) == 6
    bus.is_connected = True
    obs = f.get_observation()
    assert len(obs) == 6
    f.send_action({"shoulder_pan.pos": 0.0})
    bus.sync_write.assert_called_once()


def test_calibrate_true_raises():
    cfg = MinimalSOFollowerConfig(port="/dev/null", id="voice_arm")
    f = MinimalSOFollower(cfg, bus=FakeBus())
    with pytest.raises(NotImplementedError):
        f.connect(calibrate=True)


def test_raw_degrees_roundtrip():
    cal = MotorCalibration(id=1, range_min=720, range_max=3370)
    for deg in (-90.0, -45.0, 0.0, 45.0, 90.0):
        raw = degrees_to_raw(deg, cal)
        back = raw_to_degrees(raw, cal)
        assert back == pytest.approx(deg, abs=0.2)


def test_registers_have_expected_addresses():
    # Sanity check: addresses match lerobot v0.4.4 STS/SMS table.
    assert REGISTERS["Torque_Enable"] == (40, 1)
    assert REGISTERS["Goal_Position"] == (42, 2)
    assert REGISTERS["Present_Position"] == (56, 2)
    assert REGISTERS["Operating_Mode"] == (33, 1)
    assert REGISTERS["Lock"] == (55, 1)
