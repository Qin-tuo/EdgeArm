"""Container-side tests for ActionsManager + observation_server HTTP routes.

Runs without lerobot/uvicorn/portaudio — we stub RobotArm with a fake.
Invocation:
    pytest solutions/respeaker_flex_soarm/assets/docker/test_actions_manager.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from ovs_agent.actuators.actions import (  # noqa: E402
    MAX_FRAMES,
    ActionsError,
    ActionsManager,
)

# ── helpers ─────────────────────────────────────────────────────────


def _frame(delay: float = 0.4, **overrides: float) -> Dict[str, Any]:
    joints = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }
    # kwargs can't contain '.', so we accept underscore-suffixed names
    # (e.g. shoulder_pan_pos) and translate to the joint dict key.
    for raw_k, v in overrides.items():
        key = raw_k[:-4] + ".pos" if raw_k.endswith("_pos") else raw_k
        joints[key] = float(v)
    return {"joints": joints, "delay": delay}


@pytest.fixture
def mgr(tmp_path):
    yaml_path = tmp_path / "actions.yaml"
    # Seed a single sequence so etag is meaningful.
    yaml_path.write_text(
        yaml.safe_dump({"sequences": {"home": [_frame(delay=1.5)]}}, sort_keys=False),
        encoding="utf-8",
    )
    return ActionsManager(str(yaml_path))


# ── description / new schema support ───────────────────────────────


def test_load_legacy_schema_no_description(tmp_path) -> None:
    """Old actions.yaml (bare list under sequence name) still loads, description="" """
    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"sequences": {"home": [_frame(delay=1.5)]}}, sort_keys=False),
        encoding="utf-8",
    )
    m = ActionsManager(str(yaml_path))
    assert m.get_sequence("home") is not None
    assert m.get_description("home") == ""
    descs = m.list_with_descriptions()
    assert descs == [{"name": "home", "description": ""}]


def test_load_new_schema_with_description(tmp_path) -> None:
    """New {description, frames} schema loads description through."""
    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "sequences": {
                    "wave": {
                        "description": "Wave hi when greeted.",
                        "frames": [_frame()],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    m = ActionsManager(str(yaml_path))
    assert m.get_description("wave") == "Wave hi when greeted."
    descs = m.list_with_descriptions()
    assert descs == [{"name": "wave", "description": "Wave hi when greeted."}]
    # get_all() now includes description per entry.
    body = m.get_all()
    assert body["actions"][0]["description"] == "Wave hi when greeted."


def test_save_with_description_persists(tmp_path) -> None:
    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text("sequences: {}\n", encoding="utf-8")
    m1 = ActionsManager(str(yaml_path))
    m1.save("hi5", [_frame()], description="High-five gesture.")
    # Fresh load picks it up.
    m2 = ActionsManager(str(yaml_path))
    assert m2.get_description("hi5") == "High-five gesture."


def test_save_preserves_existing_description_when_not_provided(tmp_path) -> None:
    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text("sequences: {}\n", encoding="utf-8")
    m = ActionsManager(str(yaml_path))
    m.save("foo", [_frame()], description="Original desc.")
    # Update frames only — description must survive.
    m.save("foo", [_frame(), _frame(delay=0.5)])
    assert m.get_description("foo") == "Original desc."


# NOTE: the former ``test_build_tools_spec_shape`` exercised a container-side
# ``llm.build_tools_spec`` helper that no longer exists — tool registration is
# now done by the agent framework's ToolRegistry via
# ``tools.action_tools.register_arm_tools``. The test was dropped during the
# migration into seeed-local-voice (the helper it covered was not migrated).


# ── name validation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_name",
    ["", "1home", "Home", "high five", "../etc/passwd", "my-pose", "高五"],
)
def test_save_rejects_bad_names(mgr: ActionsManager, bad_name: str) -> None:
    with pytest.raises(ActionsError) as exc:
        mgr.save(bad_name, [_frame()])
    assert exc.value.code == "bad_name"


def test_save_accepts_good_name(mgr: ActionsManager) -> None:
    result = mgr.save("high_five", [_frame(shoulder_pan_pos=0.5)])
    assert result["ok"] is True
    assert result["name"] == "high_five"
    assert result["frames_count"] == 1
    assert result["replaced"] is False
    assert result["etag"].startswith("sha256:")


def test_save_replaced_flag(mgr: ActionsManager) -> None:
    mgr.save("foo", [_frame()])
    second = mgr.save("foo", [_frame(), _frame(delay=0.6)])
    assert second["replaced"] is True
    assert second["frames_count"] == 2


# ── frame validation ───────────────────────────────────────────────


def test_save_rejects_empty_frames(mgr: ActionsManager) -> None:
    with pytest.raises(ActionsError) as exc:
        mgr.save("foo", [])
    assert exc.value.code == "empty_frames"


def test_save_rejects_too_many_frames(mgr: ActionsManager) -> None:
    frames = [_frame()] * (MAX_FRAMES + 1)
    with pytest.raises(ActionsError) as exc:
        mgr.save("foo", frames)
    assert exc.value.code == "too_many_frames"


def test_save_rejects_missing_joint(mgr: ActionsManager) -> None:
    frame = _frame()
    del frame["joints"]["gripper.pos"]
    with pytest.raises(ActionsError) as exc:
        mgr.save("foo", [frame])
    assert exc.value.code == "missing_joint"


def test_save_rejects_delay_out_of_range(mgr: ActionsManager) -> None:
    for bad in (0.0, 0.01, 5.5, -1.0):
        with pytest.raises(ActionsError) as exc:
            mgr.save("foo", [_frame(delay=bad)])
        assert exc.value.code == "bad_delay", f"expected bad_delay for {bad}"


def test_save_rejects_non_numeric_joint(mgr: ActionsManager) -> None:
    frame = _frame()
    frame["joints"]["gripper.pos"] = "not-a-number"
    with pytest.raises(ActionsError) as exc:
        mgr.save("foo", [frame])
    assert exc.value.code == "bad_frame"


# ── atomic write ───────────────────────────────────────────────────


def test_save_atomic_failure_preserves_state(mgr: ActionsManager, monkeypatch) -> None:
    # Snapshot the in-memory state pre-failure.
    before = mgr.get_all()
    pre_etag = before["etag"]
    pre_names = {a["name"] for a in before["actions"]}

    def boom(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(
        "ovs_agent.actuators.actions.os.replace", boom
    )
    with pytest.raises(OSError):
        mgr.save("should_not_persist", [_frame()])

    after = mgr.get_all()
    assert after["etag"] == pre_etag, "etag must not change on failed write"
    assert {a["name"] for a in after["actions"]} == pre_names
    assert mgr.get_sequence("should_not_persist") is None


def test_save_writes_yaml_and_reloads(tmp_path) -> None:
    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text("sequences: {}\n", encoding="utf-8")
    m1 = ActionsManager(str(yaml_path))
    m1.save("wave", [_frame(shoulder_pan_pos=0.6), _frame(shoulder_pan_pos=-0.6)])
    # Fresh manager reads from disk → must see the persisted sequence.
    m2 = ActionsManager(str(yaml_path))
    frames = m2.get_sequence("wave")
    assert frames is not None
    assert len(frames) == 2
    assert frames[0]["joints"]["shoulder_pan.pos"] == pytest.approx(0.6)


# ── etag behavior ──────────────────────────────────────────────────


def test_etag_double_factor_invalidation(mgr: ActionsManager) -> None:
    etag1 = mgr.get_etag()
    # Cached path: nothing changed → identical
    assert mgr.get_etag() == etag1

    # External rewrite (no API call) — size + mtime both change
    path = mgr._path
    raw = path.read_text(encoding="utf-8") + "\n# touched\n"
    path.write_text(raw, encoding="utf-8")
    etag2 = mgr.get_etag()
    assert etag2 != etag1


def test_etag_cached_when_stat_unchanged(mgr: ActionsManager, monkeypatch) -> None:
    """If neither size nor mtime change, we MUST NOT re-read the file."""
    etag1 = mgr.get_etag()  # primes cache
    real_open = Path.open
    open_calls = {"count": 0}

    def counting_open(self, *args, **kwargs):  # noqa: ANN001
        if str(self).endswith("actions.yaml") and (args and args[0] == "rb"):
            open_calls["count"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    etag2 = mgr.get_etag()
    assert etag2 == etag1
    assert open_calls["count"] == 0


def test_save_updates_etag(mgr: ActionsManager) -> None:
    etag1 = mgr.get_etag()
    result = mgr.save("new_act", [_frame()])
    assert result["etag"] != etag1
    assert mgr.get_etag() == result["etag"]


# ── If-Match (optimistic concurrency) ──────────────────────────────


def test_if_match_mismatch_raises_412(mgr: ActionsManager) -> None:
    with pytest.raises(ActionsError) as exc:
        mgr.save("foo", [_frame()], if_match="sha256:wrongvalue00000")
    assert exc.value.code == "etag_mismatch"
    assert exc.value.http_status == 412


def test_if_match_correct_succeeds(mgr: ActionsManager) -> None:
    current = mgr.get_etag()
    result = mgr.save("foo", [_frame()], if_match=current)
    assert result["ok"] is True


def test_if_match_none_skips_check(mgr: ActionsManager) -> None:
    # if_match=None → no optimistic check; always saves
    assert mgr.save("foo", [_frame()], if_match=None)["ok"] is True


# ── get_sequence safety ────────────────────────────────────────────


def test_get_sequence_returns_none_for_bad_name(mgr: ActionsManager) -> None:
    # Path-traversal-style names — validated at read time too.
    for bad in ("../etc/passwd", "Home", ""):
        assert mgr.get_sequence(bad) is None


def test_get_sequence_returns_copy(mgr: ActionsManager) -> None:
    """Mutating the returned list MUST NOT mutate internal state."""
    mgr.save("foo", [_frame(), _frame()])
    a = mgr.get_sequence("foo")
    assert a is not None
    a.clear()
    b = mgr.get_sequence("foo")
    assert b is not None and len(b) == 2


# ── observation_server endpoint integration ────────────────────────


class _FakeRobot:
    def __init__(self) -> None:
        self.obs = {f"j{i}.pos": 0.0 for i in range(6)}
        self.torque_calls: List[str] = []
        self.sent_frames: List[Dict[str, float]] = []
        self.fail_update = False
        # Mirrors the Actuator ABC's public torque state. set_torque
        # updates this in lockstep with the (faked) bus, so the
        # observation server reads it via ``torque_enabled`` instead of
        # poking a private field.
        self._torque_on = True

    @property
    def torque_enabled(self) -> bool:
        return self._torque_on

    def update_cache(self) -> Dict[str, float]:
        if self.fail_update:
            raise RuntimeError("bus busy")
        return dict(self.obs)

    def get_cached_observation(self) -> Dict[str, float]:
        return dict(self.obs)

    def observation_features(self) -> Dict[str, Any]:
        return {k: {"type": "float"} for k in self.obs}

    def set_torque(self, enable: bool) -> None:
        self.torque_calls.append("on" if enable else "off")
        self._torque_on = bool(enable)

    def execute_sequence(self, frames: List[Dict[str, Any]], *, cancel_event=None) -> bool:
        for f in frames:
            if cancel_event is not None and cancel_event.is_set():
                break
            self.sent_frames.append(f.get("joints", {}))
            # Honor the frame delay so concurrency tests can observe
            # an in-progress test_running window.
            try:
                d = float(f.get("delay", 0.0))
            except (TypeError, ValueError):
                d = 0.0
            if d > 0:
                if cancel_event is not None:
                    # Sleep in small chunks to remain cancellable.
                    end = time.monotonic() + d
                    while time.monotonic() < end:
                        if cancel_event.is_set():
                            return True
                        time.sleep(min(0.05, end - time.monotonic()))
                else:
                    time.sleep(d)
        return True


@pytest.fixture
def http(tmp_path):
    from fastapi.testclient import TestClient

    from ovs_agent.plugins.actuator_observation_server import _build_app

    yaml_path = tmp_path / "actions.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"sequences": {"home": [_frame(delay=1.5)]}}, sort_keys=False),
        encoding="utf-8",
    )
    am = ActionsManager(str(yaml_path))
    robot = _FakeRobot()
    app = _build_app(robot, am)
    client = TestClient(app)
    return client, robot, am


def test_endpoint_list_actions(http) -> None:
    client, _robot, _am = http
    r = client.get("/actions")
    assert r.status_code == 200
    body = r.json()
    assert "etag" in body
    assert any(a["name"] == "home" for a in body["actions"])


def test_endpoint_etag_only(http) -> None:
    client, _robot, am = http
    r = client.get("/actions/etag")
    assert r.status_code == 200
    assert r.json()["etag"] == am.get_etag()


def test_endpoint_save_and_test_full_flow(http) -> None:
    client, robot, am = http
    etag = am.get_etag()

    # Save with correct If-Match.
    r = client.post(
        "/actions",
        json={"name": "high_five", "frames": [_frame(shoulder_pan_pos=0.5)]},
        headers={"If-Match": etag},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["replaced"] is False
    new_etag = body["etag"]
    assert new_etag != etag

    # If-Match with stale etag → 412.
    r = client.post(
        "/actions",
        json={"name": "again", "frames": [_frame()]},
        headers={"If-Match": etag},
    )
    assert r.status_code == 412
    assert r.json()["detail"]["error"] == "etag_mismatch"

    # Trigger test endpoint — debounce: we just started, torque_on_since_ns
    # was set at app construction → likely < 500ms, expect 409 debounce.
    r = client.post("/actions/high_five/test")
    assert r.status_code == 409
    assert r.json()["detail"]["error"] in ("debounce", "torque_off")

    # Advance the clock by re-posting torque/on; wait > 500ms.
    r = client.post("/torque/on")
    assert r.status_code == 200
    time.sleep(0.6)

    r = client.post("/actions/high_five/test")
    assert r.status_code == 200, r.text
    # Give the background thread a moment to send frames.
    time.sleep(0.6)
    assert robot.sent_frames, "expected at least one frame sent"


def test_endpoint_save_with_description_round_trips(http) -> None:
    """POST /actions {description} → persisted → GET /actions reflects it.

    Regression guard: the HTTP layer used to drop the description field on
    the way to actions_manager.save(), which silently broke function-calling
    (the recorded action would never trigger by voice).
    """
    client, _robot, am = http
    etag = am.get_etag()
    r = client.post(
        "/actions",
        json={
            "name": "hi5",
            "description": "Wave hi when greeted.",
            "frames": [_frame(shoulder_pan_pos=0.5)],
        },
        headers={"If-Match": etag},
    )
    assert r.status_code == 200, r.text
    assert am.get_description("hi5") == "Wave hi when greeted."

    listing = client.get("/actions").json()
    hi5 = next(a for a in listing["actions"] if a["name"] == "hi5")
    assert hi5["description"] == "Wave hi when greeted."


def test_endpoint_save_without_description_preserves_existing(http) -> None:
    """Updating frames only must not erase the existing description.

    The frontend sends `description` only when the user actually typed
    something; an empty textarea omits the key. The container must treat
    a missing key as "leave it alone".
    """
    client, _robot, am = http
    am.save("hi5", [_frame()], description="Original desc.")
    etag = am.get_etag()

    r = client.post(
        "/actions",
        json={"name": "hi5", "frames": [_frame(shoulder_pan_pos=0.3)]},
        headers={"If-Match": etag},
    )
    assert r.status_code == 200, r.text
    assert am.get_description("hi5") == "Original desc."


def test_endpoint_test_torque_off(http) -> None:
    client, _robot, _am = http
    r = client.post("/torque/off")
    assert r.status_code == 200
    # With torque off, /test must 409 torque_off (not debounce — torque check
    # comes first in the precondition chain).
    r = client.post("/actions/home/test")
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "torque_off"


def test_endpoint_test_path_traversal_blocked(http) -> None:
    client, _robot, _am = http
    # Even though FastAPI's path routing would normalize this, the handler
    # re-validates the name against NAME_RE. Use a name that contains an
    # invalid character that *survives* routing.
    r = client.post("/actions/Home/test")
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "bad_name"


def test_endpoint_test_unknown_action_404(http) -> None:
    client, _robot, _am = http
    # Wait past the debounce window so we reach the lookup step.
    time.sleep(0.6)
    r = client.post("/actions/does_not_exist/test")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "not_found"


def test_endpoint_preview_plays_ad_hoc_frames_without_saving(http) -> None:
    """POST /actions/preview accepts frames in the body, executes them on
    the arm, and does NOT write to actions.yaml — gives the verify-panel
    recorder a cheap try-before-commit loop.
    """
    client, robot, am = http
    actions_before = am.get_all()

    # Advance past the torque debounce window.
    r = client.post("/torque/on")
    assert r.status_code == 200
    time.sleep(0.6)

    r = client.post(
        "/actions/preview",
        json={"frames": [_frame(shoulder_pan_pos=0.2), _frame(shoulder_pan_pos=-0.2)]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["preview"] is True
    assert body["frames"] == 2

    # Give the runner a beat to execute.
    time.sleep(0.6)
    assert len(robot.sent_frames) >= 1
    # actions.yaml is untouched: list+etag identical to before the call.
    actions_after = am.get_all()
    assert actions_after["actions"] == actions_before["actions"]
    assert actions_after["etag"] == actions_before["etag"]


def test_endpoint_preview_validates_frames(http) -> None:
    """Invalid frames (empty / missing joints) come back as 4xx so the UI
    surfaces the same error code as a /test call would.
    """
    client, _robot, _am = http
    r = client.post("/torque/on")
    assert r.status_code == 200
    time.sleep(0.6)

    r = client.post("/actions/preview", json={"frames": []})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["error"] == "empty_frames"


def test_endpoint_preview_respects_torque_off(http) -> None:
    """Torque off → preview rejected the same way /test is rejected."""
    client, _robot, _am = http
    r = client.post("/torque/off")
    assert r.status_code == 200

    r = client.post(
        "/actions/preview",
        json={"frames": [_frame()]},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "torque_off"


def test_endpoint_cancel_acks(http) -> None:
    client, _robot, _am = http
    r = client.post("/actions/cancel")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_endpoint_observation_fallback_to_cache(http) -> None:
    client, robot, _am = http
    # Prime the cache via a live read.
    r = client.get("/observation")
    assert r.status_code == 200

    # Now make live reads fail; cache fallback should kick in.
    robot.fail_update = True
    r = client.get("/observation")
    assert r.status_code == 200
    assert r.headers.get("X-Observation-Source") == "cache"


def test_endpoint_torque_endpoints_record(http) -> None:
    client, robot, _am = http
    assert client.post("/torque/off").status_code == 200
    assert client.post("/torque/on").status_code == 200
    assert robot.torque_calls == ["off", "on"]


# ── B3 regression: torque enforcement on execute_action ────────────


def test_robot_execute_action_refuses_when_torque_off() -> None:
    """SOArmActuator.execute_action must refuse to dispatch when torque is off.

    This guards the voice pipeline path: pipeline.py calls
    actuator.execute_action() directly (not through HTTP), so the HTTP-side
    torque check would otherwise be bypassed.
    """
    from ovs_agent.apps.voice_arm.so_arm import SOArmActuator

    arm = SOArmActuator(port="/dev/null")

    # Inject a fake lerobot client so execute_sequence would otherwise run.
    class _FakeLeRobot:
        def __init__(self) -> None:
            self.sent: list = []

        def send_action(self, joints: dict) -> None:
            self.sent.append(joints)

        def get_observation(self) -> dict:
            return {}

    fake = _FakeLeRobot()
    arm._robot = fake  # noqa: SLF001
    arm._torque_state = "off"  # noqa: SLF001

    actions_map = {"sequences": {"wave": [_frame()]}}
    assert arm.execute_action("wave", actions_map) is False
    assert fake.sent == [], "no frames should be sent with torque off"

    # Re-enabling torque allows execution.
    arm._torque_state = "on"  # noqa: SLF001
    assert arm.execute_action("wave", actions_map) is True
    assert len(fake.sent) == 1


def test_torque_endpoints_sync_robot_state(http) -> None:
    """observation_server's /torque/off|on must flip the actuator's public
    ``torque_enabled`` state.

    Otherwise the voice pipeline (which reads ``torque_enabled``) would
    see a stale "on" value after a user toggled torque off via the verify
    panel. This is the linchpin of B3 — same source of truth for both
    callers. The server drives this purely through the public
    ``set_torque`` call (no private ``_torque_state`` poke).
    """
    client, robot, _am = http
    assert robot.torque_enabled is True
    assert client.post("/torque/off").status_code == 200
    assert robot.torque_enabled is False
    assert client.post("/torque/on").status_code == 200
    assert robot.torque_enabled is True


# ── W2 regression: /test response includes duration_ms ─────────────


def test_endpoint_test_returns_duration_ms(http) -> None:
    client, _robot, am = http
    # Make sure debounce window has passed.
    client.post("/torque/on")
    time.sleep(0.6)

    # Save a 3-frame sequence with known delays.
    am.save("three_frame", [_frame(delay=0.5), _frame(delay=0.4), _frame(delay=0.3)])
    r = client.post("/actions/three_frame/test")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frames"] == 3
    # duration_ms should be sum(delays)*1000 + frames * ESTIMATED_MOVE_MS_PER_FRAME
    # = 1200 + 3*500 = 2700ms. Allow a small tolerance for future tuning.
    assert "duration_ms" in body, body
    dur = int(body["duration_ms"])
    assert dur >= 1200, f"duration_ms {dur} too small for sum(delays)=1.2s"
    # Wait for the test thread to clear test_running before next test.
    time.sleep(0.2)


# ── B2 regression: atomic busy check rejects concurrent /test ──────


def test_endpoint_concurrent_test_409_busy(http) -> None:
    """Two concurrent /test calls: the second MUST get 409 busy.

    The previous implementation had a window between "check busy" and
    "set busy" where two requests could both slip through. After the
    fix, the check+set are inside one state_lock critical section.
    """
    import threading as _t

    client, _robot, am = http
    client.post("/torque/on")
    time.sleep(0.6)

    # Save a slow sequence so the runner stays "busy" long enough to
    # race a second request against it.
    am.save("slow_seq", [_frame(delay=1.0), _frame(delay=1.0)])

    results: list = []

    def _hit() -> None:
        r = client.post("/actions/slow_seq/test")
        results.append(r.status_code)

    # Fire two requests back-to-back. TestClient is synchronous per call
    # but threads make the dispatch concurrent at the server's lock.
    threads = [_t.Thread(target=_hit) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5)

    assert sorted(results) == sorted([200, 409]), (
        f"expected exactly one 200 and one 409, got {results}"
    )
    # Cleanup: wait for slow_seq runner to finish so it doesn't bleed
    # state_running=True into other tests.
    time.sleep(2.5)


def test_endpoint_save_409_during_test(http) -> None:
    """POST /actions must return 409 while a test is running."""
    import threading as _t

    client, _robot, am = http
    client.post("/torque/on")
    time.sleep(0.6)

    am.save("slow2", [_frame(delay=1.0)])
    started = _t.Event()

    def _start_test() -> None:
        client.post("/actions/slow2/test")
        started.set()

    t = _t.Thread(target=_start_test)
    t.start()
    started.wait(timeout=2)
    # While the runner is still going, a save attempt should be rejected.
    r = client.post("/actions", json={"name": "new_one", "frames": [_frame()]})
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "busy"
    time.sleep(1.5)
