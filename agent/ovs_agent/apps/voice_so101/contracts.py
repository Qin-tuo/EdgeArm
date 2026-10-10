"""Bounded, structured sensor contracts. Device owners publish; tools read.

Times below are host monotonic seconds after explicit source-clock conversion.
Raw camera/depth data stay with the perception owner, not the audio event loop.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class ObjectCandidate:
    object_id: str
    label: str
    base_xyz_m: tuple[float, float, float]
    confidence: float
    depth_valid: bool
    reachable: bool


@dataclass(frozen=True)
class Scene:
    sequence: int
    captured_at: float
    arrived_at: float
    calibration_version: str  # Camera intrinsics and fixed camera-to-base transform.
    objects: tuple[ObjectCandidate, ...]
    source_time: float | None = None
    source_clock: str = "host_monotonic"
    clock_offset_s: float = 0.0


@dataclass(frozen=True)
class Contact:
    sequence: int
    sampled_at: float
    arrived_at: float
    calibration_version: str
    pressure: float
    unit: str
    touching: bool
    healthy: bool
    source_time: float | None = None
    source_clock: str = "host_monotonic"
    clock_offset_s: float = 0.0


class SensorStore:
    """One-slot queues; app-loop publication only, no unbounded frame backlog."""
    def __init__(self, max_age_s: float = 0.5):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.max_age_s = max_age_s
        self.scene: Scene | None = None
        self.contact: Contact | None = None

    def _validate(self, sample, sampled_at, previous):
        now = time.monotonic()
        if type(sample.sequence) is not int or sample.sequence < 0 or (previous and sample.sequence <= previous.sequence):
            raise ValueError("sensor sequence must increase")
        if not sample.calibration_version or not sample.source_clock:
            raise ValueError("calibration version and clock domain are required")
        if not all(math.isfinite(v) for v in (sampled_at, sample.arrived_at, sample.clock_offset_s)):
            raise ValueError("invalid sensor timestamps")
        if not 0 <= sampled_at <= sample.arrived_at <= now or now - sampled_at > self.max_age_s:
            raise ValueError("stale or future sensor sample")

    def publish_scene(self, sample: Scene) -> None:
        self._validate(sample, sample.captured_at, self.scene)
        if len(sample.objects) > 32 or len({o.object_id for o in sample.objects}) != len(sample.objects):
            raise ValueError("scene requires <=32 unique object IDs")
        for obj in sample.objects:
            if not obj.object_id or not obj.label or len(obj.base_xyz_m) != 3:
                raise ValueError("invalid object candidate")
            if type(obj.depth_valid) is not bool or type(obj.reachable) is not bool:
                raise ValueError("depth/reachability flags must be booleans")
            if not all(math.isfinite(v) for v in obj.base_xyz_m) or not math.isfinite(obj.confidence) or not 0 <= obj.confidence <= 1:
                raise ValueError("invalid object geometry/confidence")
        self.scene = sample

    def publish_contact(self, sample: Contact) -> None:
        self._validate(sample, sample.sampled_at, self.contact)
        if type(sample.touching) is not bool or type(sample.healthy) is not bool:
            raise ValueError("contact health flags must be booleans")
        if not sample.unit or not math.isfinite(sample.pressure):
            raise ValueError("contact requires finite, calibrated measurement and unit")
        self.contact = sample

    def ready(self) -> bool:
        now = time.monotonic()
        return bool(self.scene and self.contact and self.contact.healthy
                    and now - self.scene.captured_at <= self.max_age_s
                    and now - self.contact.sampled_at <= self.max_age_s)

    def snapshot(self) -> dict:
        return {"ready": self.ready(), "scene": asdict(self.scene) if self.scene else None,
                "contact": asdict(self.contact) if self.contact else None}


class GraspBackend(Protocol):
    """A hardware-adapted planner/verifier; deliberately no default fake.

Build bounded, low-speed frames for one reachable object and a configured zone.
Verification must inspect NEW scene/contact plus joint feedback, not sleep or
the dispatch result. Finite recovery must reuse the same controller/bus owner.
"""
    def plan(self, scene: Scene, contact: Contact, object_id: str, zone_id: str) -> list[dict]: ...
    def verify(self, scene: Scene, contact: Contact, feedback: dict,
               object_id: str, zone_id: str) -> bool: ...
