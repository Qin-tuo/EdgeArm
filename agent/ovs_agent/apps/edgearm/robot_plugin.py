"""SO-101 semantic tools and local operator control using OVS Plugin.

One serial worker owns the actuator. Perception/MCU adapters publish on the app
loop via SensorStore; there is no competing camera or serial owner here.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web
from ovs_agent.actuators.factory import create_actuator
from ovs_agent.plugin import Plugin
from . import so_arm  # Registers the existing SO driver through the OVS factory.
from .contracts import SensorStore


class SO101Plugin(Plugin):
    name = "so101"

    def __init__(self, app, config: dict):
        super().__init__(app)
        self.cfg = config
        self.sensors = SensorStore(float(config.get("sensor_max_age_s", 0.5)))
        self.arm = None
        self.grasp_backend = None  # Installed by the verified hardware adapter, not YAML imports.
        self.boot_id = str(uuid.uuid4())
        self.state = "idle"
        self.records: dict[str, dict] = {}
        self._active = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-bus")
        self._feedback = {}
        self._poll = None
        self._runner = None
        self._socket = None
        self._shutdown = False

    async def _serial(self, fn, *args):
        future = asyncio.get_running_loop().run_in_executor(self._worker, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A cancelled caller must not discard a queued hold/disconnect,
            # nor pretend a running serial operation has stopped.
            try:
                await asyncio.shield(future)
            finally:
                raise

    def setup(self) -> bool:
        registry = self.app.tool_registry
        registry.tool(name="robot_query", description="查询机械臂健康、物体/放置区ID与当前请求令牌。")(self.query)
        registry.tool(name="robot_grasp_place", description="请求抓放已验证物体到已配置放置区。接受不等于完成；仅 completed 表示已验证成功。")(self.submit)
        registry.tool(name="robot_cancel", description="取消抓放并请求受控停止；保持夹爪，不自动释放物体。")(self.cancel)
        for name in ("robot_query", "robot_grasp_place", "robot_cancel"):
            registry._tools[name].parameters["additionalProperties"] = False
        if self.cfg.get("enabled", False) is True:
            self.arm = create_actuator("so_arm", dict(self.cfg.get("actuator", {})))
        return True

    async def start(self) -> None:
        await super().start()
        if self.arm is not None:
            try:
                await self._serial(self.arm.connect)
                self._feedback = self.arm.feedback()
            except Exception as exc:
                self.state = "fault"
                self._feedback = {"error": str(exc), "motion_enabled": False}
            self._poll = asyncio.create_task(self._observe(), name="so101-feedback")
        socket_path = Path(self.cfg.get("control_socket", "/run/edgearm/control.sock"))
        socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Existing socket may belong to another live owner: never unlink it on startup.
        if socket_path.exists():
            raise RuntimeError(f"operator socket already exists: {socket_path}")
        api = web.Application(client_max_size=4096)
        api.router.add_get("/status", self._status_http)
        api.router.add_post("/enable", self._enable_http)
        api.router.add_post("/stop", self._stop_http)
        self._runner = web.AppRunner(api)
        await self._runner.setup()
        await web.UnixSite(self._runner, str(socket_path)).start()
        os.chmod(socket_path, 0o600)
        self._socket = socket_path

    async def _observe(self):
        while self._running:
            try:
                await self._serial(self.arm.update_cache)
                self._feedback = self.arm.feedback()
            except Exception as exc:
                self.arm.request_stop()
                self.state = "fault"
                self._feedback = {"error": str(exc), "motion_enabled": False}
            await asyncio.sleep(0.1)

    def query(self) -> dict:
        feedback = dict(self._feedback)
        if feedback.get("sampled_at"):
            feedback["age_s"] = time.monotonic() - feedback["sampled_at"]
        return {"state": self.state, "boot_id": self.boot_id, "now": time.monotonic(),
                "hardware_validated": False, "feedback": feedback,
                "sensors": self.sensors.snapshot(), "zones": list(self.cfg.get("zones", [])),
                "grasp_available": self.grasp_backend is not None,
                "tasks": list(self.records.values())[-16:]}

    async def submit(self, object_id: str, zone_id: str, request_id: str,
                     boot_id: str, expires_at: float) -> dict:
        if any(not isinstance(v, str) or not v or len(v) > 128 for v in (object_id, zone_id, request_id, boot_id)):
            return {"status": "rejected", "error": "invalid IDs"}
        now = time.monotonic()
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)) or not math.isfinite(expires_at) or not now < expires_at <= now + 5:
            return {"status": "rejected", "error": "request must expire within five host-monotonic seconds"}
        if boot_id != self.boot_id or self._shutdown:
            return {"status": "rejected", "error": "stale boot/cancel token"}
        if request_id in self.records:
            record = self.records[request_id]
            if (record["object_id"], record["zone_id"]) != (object_id, zone_id):
                return {"status": "rejected", "error": "request ID reused for another task"}
            return dict(record)
        if len(self.records) >= 256:
            return {"status": "rejected", "error": "request ledger full; no unknown action will be replayed"}
        record = {"request_id": request_id, "object_id": object_id, "zone_id": zone_id,
                  "status": "rejected", "error": ""}
        self.records[request_id] = record
        if self._active is not None and not self._active.done():
            record["error"] = "another task owns the arm"
        elif self.state in ("fault", "stopping") or not self._feedback.get("motion_enabled"):
            record["error"] = "motion disabled; local operator enable required"
        elif not self.arm or not self.grasp_backend or not self.sensors.ready():
            record["error"] = "grasp/perception/contact backend unavailable or stale"
        elif zone_id not in self.cfg.get("zones", []):
            record["error"] = "unknown configured zone"
        else:
            objects = {o.object_id: o for o in self.sensors.scene.objects}
            obj = objects.get(object_id)
            if obj is None or not obj.depth_valid or not obj.reachable:
                record["error"] = "unknown/invalid/unreachable object"
            else:
                record.update(status="accepted", error="")
                self.state = "approaching"
                self._active = asyncio.create_task(
                    self._execute(record, boot_id, expires_at), name=f"so101-{request_id}",
                )
        return dict(record)

    async def _execute(self, record, boot_id, expires_at):
        try:
            scene, contact = self.sensors.scene, self.sensors.contact
            frames = await asyncio.wait_for(
                asyncio.to_thread(self.grasp_backend.plan, scene, contact,
                                  record["object_id"], record["zone_id"]),
                timeout=max(0.001, expires_at - time.monotonic()),
            )
            if boot_id != self.boot_id:
                record["status"] = "cancelled"
                return
            if time.monotonic() >= expires_at or not self.sensors.ready():
                record.update(status="rejected", error="expired or stale before dispatch")
                return
            dispatched = await self._serial(self.arm.execute_sequence, frames)
            self._feedback = self.arm.feedback()
            if boot_id != self.boot_id or not dispatched:
                record["status"] = "cancelled"
                return
            self.state = "verifying"
            # Samples collected during planning/motion cannot prove the final
            # result. Give the device owners a bounded window to observe it.
            finished_at = time.monotonic()
            deadline = finished_at + 2
            while True:
                if boot_id != self.boot_id:
                    record['status'] = 'cancelled'
                    return
                new_scene, new_contact = self.sensors.scene, self.sensors.contact
                if (self.sensors.ready() and new_scene.sequence > scene.sequence
                        and new_contact.sequence > contact.sequence
                        and new_scene.captured_at >= finished_at
                        and new_contact.sampled_at >= finished_at):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("new scene/contact observations required to verify task")
                await asyncio.sleep(0.02)
            await self._serial(self.arm.update_cache)
            self._feedback = self.arm.feedback()
            verified = await asyncio.wait_for(
                asyncio.to_thread(self.grasp_backend.verify, new_scene, new_contact,
                                  self._feedback, record["object_id"], record["zone_id"]), timeout=2,
            )
            if boot_id != self.boot_id:
                record["status"] = "cancelled"
            elif verified is True:
                record["status"] = "completed"
            else:
                raise RuntimeError("grasp/place result not verified")
        except Exception as exc:
            if boot_id != self.boot_id:
                record['status'] = 'cancelled'
                return
            record.update(status="fault", error=str(exc))
            self.state = "fault"
            self.arm.request_stop()
            try:
                record["stop"] = await self._serial(self.arm.stop_motion)
            except Exception as stop_exc:
                record["stop"] = {"confirmed": False, "error": str(stop_exc)}
        finally:
            if self.state not in ("fault", "stopping"):
                self.state = "idle"

    async def cancel(self) -> dict:
        self.boot_id = str(uuid.uuid4())  # Discards late tools from the cancelled turn.
        self.state = "stopping"
        if self.arm is None:
            self.state = "idle"
            return {"status": "motion_disabled", "confirmed": False}
        self.arm.request_stop()  # Thread-safe signal before queued serial work.
        try:
            result = await self._serial(self.arm.stop_motion)
            self._feedback = self.arm.feedback()
            self.state = "idle" if result.get("confirmed") or result.get("status") == "motion_disabled" else "fault"
            return result
        except Exception as exc:
            self.state = "fault"
            return {"status": "stop_failed", "confirmed": False, "error": str(exc)}

    async def on_user_stop_intent(self, data: str) -> None:
        await self.cancel()

    async def on_sleep(self, data) -> None:
        await self.cancel()

    async def on_error(self, exc) -> None:
        await self.cancel()

    async def _status_http(self, request):
        return web.json_response(self.query())

    async def _enable_http(self, request):
        if not self.arm or self._shutdown or self.state in ("fault", "stopping") or (self._active and not self._active.done()):
            return web.json_response({"error": "arm unavailable, faulted or busy"}, status=409)
        try:
            await self._serial(self.arm.set_torque, True)
            self._feedback = self.arm.feedback()
            enabled = self._feedback.get("motion_enabled", False)
            return web.json_response({"motion_enabled": enabled, "hardware_validated": False},
                                     status=200 if enabled else 409)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=409)

    async def _stop_http(self, request):
        return web.json_response(await self.cancel())

    async def stop(self) -> None:
        self._shutdown = True
        self._running = False
        await self.cancel()
        if self._active:
            await self._active  # Do not abandon a live serial worker on coroutine cancellation.
        if self._poll:
            await self._poll
        if self.arm:
            await self._serial(self.arm.disconnect)
        if self._runner:
            await self._runner.cleanup()
        if self._socket:
            self._socket.unlink(missing_ok=True)
        self._worker.shutdown(wait=True)
        await super().stop()
