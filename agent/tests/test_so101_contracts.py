"""Software-only contracts: no microphone, camera, MCU, NPU or motor access."""
import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ovs_agent.actuators.base import ActuatorClosed
from ovs_agent.apps.voice_so101.contracts import Scene, Contact, ObjectCandidate, SensorStore
from ovs_agent.apps.voice_so101.robot_plugin import SO101Plugin
from ovs_agent.apps.voice_so101 import so_arm
from ovs_agent.apps.voice_so101.minimal_so_follower import MinimalSOFollower, MinimalSOFollowerConfig, SO_FOLLOWER_MOTORS
from ovs_agent.tools.registry import ToolRegistry
from .test_minimal_so_follower import FakeBus


def samples(store, seq=1):
    now = time.monotonic()
    store.publish_scene(Scene(seq, now, now, 'intrinsics-extrinsics-v1',
                              (ObjectCandidate('cup1', 'cup', (0.1, 0.2, 0.1), 0.9, True, True),)))
    store.publish_contact(Contact(seq, now, now, 'adc-v1', 1.0, 'N', True, True))


def test_sensor_slots_reject_stale_future_reordered_and_invalid_data():
    store = SensorStore()
    samples(store)
    with pytest.raises(ValueError, match='sequence'):
        samples(store)
    now = time.monotonic()
    for sampled, arrived in ((now - 2, now), (now + 1, now + 1)):
        with pytest.raises(ValueError, match='stale or future'):
            store.publish_contact(Contact(2, sampled, arrived, 'adc1', 1, 'N', True, True))
    with pytest.raises(ValueError, match='booleans'):
        store.publish_contact(Contact(2, now, now, 'adc1', 1, 'N', True, 'true'))
    samples(store, 2)
    assert store.scene.sequence == 2 and store.contact.sequence == 2
    assert store.ready()
    store.max_age_s = 0.000001
    assert not store.ready()


@pytest.fixture
def actuator(tmp_path, monkeypatch):
    calibration = {name: dict(id=mid, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000)
                   for name, mid in SO_FOLLOWER_MOTORS.items()}
    path = tmp_path / 'so101.json'
    path.write_text(json.dumps(calibration))
    bus = FakeBus()
    follower = MinimalSOFollower(MinimalSOFollowerConfig(
        '/dev/serial/by-id/test', 'so101', tmp_path, use_degrees=True), bus)
    monkeypatch.setattr(so_arm, 'MinimalSOFollower', lambda cfg: follower)
    arm = so_arm.SOArmActuator('/dev/serial/by-id/test', str(tmp_path),
                               calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    arm.connect()
    yield arm, bus, path
    arm.disconnect()


def test_motion_defaults_disabled_and_missing_calibration_never_opens_bus(tmp_path, actuator):
    arm, bus, _ = actuator
    assert not arm.torque_enabled and not arm.feedback()['motion_enabled']
    assert bus.enable_torque_calls == 0
    assert arm.execute_sequence([{'joints': arm.get_cached_observation(), 'delay': 0.1}]) is False
    missing = FakeBus()
    follower = MinimalSOFollower(MinimalSOFollowerConfig('unused', 'missing', tmp_path), missing)
    with pytest.raises(FileNotFoundError):
        follower.connect()
    assert not missing.connect_calls
    assert not (tmp_path / 'missing.json').exists()


def test_operator_enable_pins_calibration_and_stages_hold_before_torque(actuator):
    arm, bus, path = actuator
    arm.calibration_sha256 = ''
    with pytest.raises(RuntimeError, match='hash'):
        arm.set_torque(True)
    arm.calibration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    arm.set_torque(True)
    assert bus.sync_writes[-1][0] == 'Goal_Position'
    assert bus.enable_torque_calls == 1 and arm.torque_enabled
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='hash'):
        arm.set_torque(True)
    assert not arm.feedback()['motion_enabled']
    assert arm.torque_enabled  # Rejected enable revokes motion without dropping a load.


def test_preflight_rejects_unsafe_sequence_without_any_goal_write(actuator):
    arm, bus, _ = actuator
    arm.set_torque(True)
    count = len(bus.sync_writes)
    joints = arm.get_cached_observation()
    invalid = dict(joints, **{'shoulder_pan.pos': 200})
    with pytest.raises(ValueError, match='speed bound'):
        arm.execute_sequence([{'joints': joints, 'delay': 0.1}, {'joints': invalid, 'delay': 0.1}])
    assert len(bus.sync_writes) == count
    with pytest.raises(ValueError, match='finite'):
        arm.execute_sequence([{'joints': dict(joints, **{'gripper.pos': float('nan')}), 'delay': 0.1}])
    assert len(bus.sync_writes) == count


def test_stop_interrupts_frame_wait_holds_gripper_and_requires_reenable(actuator):
    arm, bus, _ = actuator
    arm.set_torque(True)
    joints = arm.get_cached_observation()
    result = []
    t = threading.Thread(target=lambda: result.append(arm.execute_sequence([{'joints': joints, 'delay': 2}])))
    t.start()
    time.sleep(0.03)
    started = time.monotonic()
    stopped = arm.stop_motion()
    t.join(timeout=1)
    assert not t.is_alive() and result == [False]
    assert time.monotonic() - started < 0.5
    assert stopped['confirmed'] and not stopped['hardware_validated']
    assert not arm.feedback()['motion_enabled']
    assert 'gripper' in bus.sync_writes[-1][1]
    assert arm.torque_enabled  # Holding does not drop a carried object.
    arm.disconnect()
    with pytest.raises(ActuatorClosed):
        arm.connect()


def test_stop_during_enable_does_not_rearm(actuator):
    arm, bus, _ = actuator
    entered, release = threading.Event(), threading.Event()
    def enable():
        entered.set()
        assert release.wait(1)
    bus.enable_torque = enable
    t = threading.Thread(target=lambda: arm.set_torque(True))
    t.start()
    assert entered.wait(1)
    arm.request_stop()
    release.set()
    t.join(1)
    assert not arm.feedback()['motion_enabled']


class FakeArm:
    def __init__(self):
        self.stopped = False
        self.dispatches = 0
    def feedback(self):
        return {'motion_enabled': not self.stopped, 'sampled_at': time.monotonic()}
    def update_cache(self):
        return {}
    def request_stop(self):
        self.stopped = True
    def stop_motion(self):
        return {'status': 'hold_feedback_stable', 'confirmed': True, 'hardware_validated': False}
    def execute_sequence(self, frames):
        self.dispatches += 1
        return not self.stopped
    def disconnect(self):
        pass


@pytest.fixture
def robot():
    plugin = SO101Plugin(SimpleNamespace(tool_registry=ToolRegistry()), {'zones': ['tray']})
    assert plugin.setup()
    yield plugin
    plugin._worker.shutdown(wait=True)


@pytest.mark.asyncio
async def test_tool_surface_is_semantic_strict_and_disabled_by_default(robot):
    schemas = robot.app.tool_registry.list_openai_tools()
    assert {x['function']['name'] for x in schemas} == {'robot_query', 'robot_grasp_place', 'robot_cancel'}
    assert all(x['function']['parameters']['additionalProperties'] is False for x in schemas)
    assert robot.arm is None and not robot.query()['grasp_available']
    assert (await robot.app.tool_registry.dispatch('robot_query', {'joints': {}}, None))['success'] is False
    result = await robot.submit('cup1', 'tray', 'req1', robot.boot_id, time.monotonic() + 3)
    assert result['status'] == 'rejected' and 'disabled' in result['error']


@pytest.mark.asyncio
async def test_expiry_boot_id_deduplication_and_cancel_token(robot):
    old_boot = robot.boot_id
    assert (await robot.submit('cup1', 'tray', 'x', old_boot, time.monotonic() - 1))['status'] == 'rejected'
    assert (await robot.submit('cup1', 'tray', 'x', 'old-boot', time.monotonic() + 2))['status'] == 'rejected'
    first = await robot.submit('cup1', 'tray', 'x', old_boot, time.monotonic() + 2)
    assert await robot.submit('cup1', 'tray', 'x', old_boot, time.monotonic() + 2) == first
    assert 'reused' in (await robot.submit('other', 'tray', 'x', old_boot, time.monotonic() + 2))['error']
    await robot.cancel()
    assert robot.boot_id != old_boot
    assert 'stale' in (await robot.submit('cup1', 'tray', 'new', old_boot, time.monotonic() + 2))['error']


@pytest.mark.asyncio
async def test_accepted_dispatch_is_never_reported_completed_without_new_evidence(robot):
    robot.arm = FakeArm()
    robot._feedback = robot.arm.feedback()
    samples(robot.sensors)
    robot.grasp_backend = SimpleNamespace(plan=lambda *a: [{}], verify=lambda *a: True)
    accepted = await robot.submit('cup1', 'tray', 'task1', robot.boot_id, time.monotonic() + 3)
    assert accepted['status'] == 'accepted'
    await robot._active
    assert robot.records['task1']['status'] == 'fault'
    assert 'new scene/contact' in robot.records['task1']['error']
    assert robot.arm.stopped and robot.arm.dispatches == 1


@pytest.mark.asyncio
async def test_cancel_during_planning_blocks_late_dispatch_and_second_motion(robot):
    robot.arm = FakeArm()
    robot._feedback = robot.arm.feedback()
    samples(robot.sensors)
    entered, release = threading.Event(), threading.Event()
    def plan(*args):
        entered.set()
        assert release.wait(2)
        return [{}]
    robot.grasp_backend = SimpleNamespace(plan=plan, verify=lambda *a: True)
    result = await robot.submit('cup1', 'tray', 'task1', robot.boot_id, time.monotonic() + 3)
    assert result['status'] == 'accepted'
    while not entered.is_set():
        await asyncio.sleep(0.001)
    assert 'owns' in (await robot.submit('cup1', 'tray', 'task2', robot.boot_id, time.monotonic() + 3))['error']
    stopped = await robot.cancel()
    release.set()
    await robot._active
    assert stopped['confirmed'] and robot.arm.dispatches == 0
    assert robot.records['task1']['status'] == 'cancelled'


@pytest.mark.parametrize('field,value', [
    ('range_min', None), ('range_max', None), ('drive_mode', None),
    ('homing_offset', None), ('id', True), ('range_min', '1000'),
])
def test_partial_or_coerced_calibration_never_opens_port(tmp_path, field, value):
    calibration = {name: dict(id=mid, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000)
                   for name, mid in SO_FOLLOWER_MOTORS.items()}
    if value is None:
        del calibration['gripper'][field]
    else:
        calibration['gripper'][field] = value
    (tmp_path / 'so101.json').write_text(json.dumps(calibration))
    bus = FakeBus()
    follower = MinimalSOFollower(MinimalSOFollowerConfig('unused', 'so101', tmp_path), bus)
    with pytest.raises(ValueError, match='explicit integer'):
        follower.connect()
    assert not bus.connect_calls


def test_slow_feedback_revokes_motion_and_prevents_goal_write(actuator):
    arm, bus, _ = actuator
    arm.set_torque(True)
    previous_read = bus.sync_read
    def slow_read(*args):
        time.sleep(0.03)
        return previous_read(*args)
    bus.sync_read = slow_read
    arm.feedback_max_age_s = 0.01
    count = len(bus.sync_writes)
    with pytest.raises(RuntimeError, match='age budget'):
        arm.execute_sequence([{'joints': arm.get_cached_observation(), 'delay': 0.1}])
    assert len(bus.sync_writes) == count
    assert not arm.feedback()['motion_enabled']


@pytest.mark.asyncio
async def test_completion_requires_post_dispatch_samples_and_independent_verifier(robot):
    robot.arm = FakeArm()
    robot._feedback = robot.arm.feedback()
    samples(robot.sensors)
    verified = []
    def verify(scene, contact, feedback, object_id, zone_id):
        verified.append((scene.sequence, contact.sequence, object_id, zone_id))
        return True
    robot.grasp_backend = SimpleNamespace(plan=lambda *a: [{}], verify=verify)
    await robot.submit('cup1', 'tray', 'task1', robot.boot_id, time.monotonic() + 3)
    deadline = time.monotonic() + 2
    while robot.state != 'verifying' and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert robot.state == 'verifying'
    assert robot.records['task1']['status'] == 'accepted' and not verified
    samples(robot.sensors, 2)
    await asyncio.wait_for(robot._active, timeout=2)
    assert verified == [(2, 2, 'cup1', 'tray')]
    assert robot.records['task1']['status'] == 'completed'
    assert not robot.arm.stopped


@pytest.mark.asyncio
async def test_cancelled_caller_cannot_discard_queued_hardware_stop(robot):
    robot.arm = FakeArm()
    held = threading.Event()
    original_stop = robot.arm.stop_motion
    def stop():
        held.set()
        return original_stop()
    robot.arm.stop_motion = stop
    occupied, release = threading.Event(), threading.Event()
    def busy_bus():
        occupied.set()
        assert release.wait(2)
    worker = robot._worker.submit(busy_bus)
    while not occupied.is_set():
        await asyncio.sleep(0.001)
    caller = asyncio.create_task(robot.cancel())
    await asyncio.sleep(0)
    assert robot.arm.stopped  # Gate revoked even while the worker is busy.
    caller.cancel()
    await asyncio.sleep(0)
    assert not held.is_set() and not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=2)
    assert held.is_set() and worker.done()
