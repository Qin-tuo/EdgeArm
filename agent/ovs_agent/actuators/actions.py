"""actions_manager.py — manages the in-memory ACTION_MAP + actions.yaml on disk.

Sequences-only schema (see DESIGN_action_modules.md §4):

  sequences:
    <action_name>:
      description: "Short natural-language description of when to trigger."  # optional
      frames:
        - joints: {shoulder_pan.pos: 0.0, ...}
          delay: 0.4

Legacy schema (still loaded for backward compat — description defaults to ""):

  sequences:
    <action_name>:
      - joints: {...}
        delay: 0.4

The description, when present, is forwarded to the LLM tools spec so
function calling can pick the right action without prompt-engineered
phrase rules. See llm.py:build_tools_spec.

ActionsManager is the **single source of truth** for action state in the
voice-arm container. All readers and writers must go through it; do not
parse actions.yaml elsewhere.

Concurrency:
  * A single threading.Lock (`_lock`) guards reads, writes, and etag cache.
  * `save()` uses tempfile + os.replace so a crash mid-write cannot leave
    the yaml file corrupted; in-memory ACTION_MAP is only mutated AFTER
    os.replace succeeds, preserving the "file is source of truth" invariant.

Etag:
  * Returned in `GET /actions` and `GET /actions/etag`.
  * Algorithm: sha256(file_bytes).hexdigest()[:16] prefixed with "sha256:"
  * Double-factor invalidation: (st_size, st_mtime_ns) — if either differs
    from the cached tuple we rehash. mtime alone is unreliable at sub-ns
    boundaries; size+mtime collisions for different content are astronomically
    unlikely in practice.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# Validation constants — kept module-level so tests can import them.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_FRAMES = 50
MIN_DELAY = 0.05
MAX_DELAY = 5.0
MAX_YAML_BYTES = 100 * 1024  # 100 KB sanity cap (matches yaml_file P-2 limit)

# Default required joint set — used only when the caller does not supply
# an actuator-derived ``required_fields``. Every saved frame must specify
# all of these (even if a value is 0.0) so execution is deterministic
# (no "hold last" surprises).
#
# This module-level tuple is NOT the source of truth: ``ActionsManager``
# takes a ``required_fields`` ctor arg (wired from the actuator's
# ``observation_features()`` by ArmPlugin) so a different motor with a
# different joint set works without editing this file. The SO-ARM 6-joint
# default is kept here purely for backward-compat with callers that
# construct an ``ActionsManager`` without specifying the field set.
REQUIRED_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class ActionsError(Exception):
    """Raised for any user-input validation or persistence failure.

    `code` is a short machine-readable string (e.g. "bad_name", "too_many_frames",
    "etag_mismatch") so callers can map to HTTP status without string-matching.
    """

    def __init__(self, code: str, message: str, *, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass
class _EtagCache:
    etag: str
    size: int
    mtime_ns: int


class ActionsManager:
    def __init__(
        self,
        yaml_path: str | os.PathLike[str],
        required_fields: Optional[List[str]] = None,
    ) -> None:
        self._path = Path(yaml_path)
        self._lock = threading.Lock()
        # name -> {"description": str, "frames": [frame, ...]}
        self._action_map: Dict[str, Dict[str, Any]] = {}
        self._etag_cache: Optional[_EtagCache] = None
        # The joint fields every saved frame must specify. Defaults to the
        # SO-ARM 6-joint set (REQUIRED_JOINTS) for backward compat, but
        # ArmPlugin wires the actuator's ``observation_features()`` keys
        # here so a different motor with a different joint set validates
        # against ITS fields, not the hard-coded SO-ARM ones.
        self._required_fields: Tuple[str, ...] = (
            tuple(required_fields) if required_fields else REQUIRED_JOINTS
        )
        self._load_from_disk()

    # ── public read API ─────────────────────────────────────────────

    def get_all(self) -> Dict[str, Any]:
        """Snapshot of actions + etag for GET /actions."""
        with self._lock:
            etag = self._compute_etag_locked()
            return {
                "etag": etag,
                "actions": [
                    {
                        "name": name,
                        "frames": len(entry["frames"]),
                        "description": entry.get("description", ""),
                    }
                    for name, entry in sorted(self._action_map.items())
                ],
            }

    def get_etag(self) -> str:
        """Cheap etag for GET /actions/etag (double-factor invalidation)."""
        with self._lock:
            return self._compute_etag_locked()

    def get_sequence(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Return a deep-enough copy of a sequence for the main pipeline."""
        if not isinstance(name, str) or not NAME_RE.match(name):
            return None
        with self._lock:
            entry = self._action_map.get(name)
            if entry is None:
                return None
            # Shallow copy of list is fine — frames themselves are
            # never mutated by the executor (it reads joints / delay).
            return [dict(f) for f in entry["frames"]]

    def get_description(self, name: str) -> Optional[str]:
        if not isinstance(name, str) or not NAME_RE.match(name):
            return None
        with self._lock:
            entry = self._action_map.get(name)
            if entry is None:
                return None
            return entry.get("description", "")

    def list_with_descriptions(self) -> List[Dict[str, str]]:
        """Returns [{name, description, response_mode?, completion_text?}, ...].

        Used by arm_tools.register_arm_tools to thread per-action response
        mode metadata into the ovs-agent ToolRegistry. Keys other than
        ``name``/``description`` are only present when set in actions.yaml,
        so the upstream framework's default (``response_mode="await"``,
        empty completion_text) still applies when an action omits them.
        """
        with self._lock:
            out: List[Dict[str, Any]] = []
            for name, entry in sorted(self._action_map.items()):
                item: Dict[str, Any] = {
                    "name": name,
                    "description": entry.get("description", ""),
                }
                rmode = entry.get("response_mode")
                if isinstance(rmode, str) and rmode:
                    item["response_mode"] = rmode
                ctext = entry.get("completion_text")
                if isinstance(ctext, str) and ctext:
                    item["completion_text"] = ctext
                out.append(item)
            return out

    def list_names(self) -> List[str]:
        """Snapshot of action names (used by LLM 'valid_actions' set)."""
        with self._lock:
            return sorted(self._action_map.keys())

    def as_actions_map(self) -> Dict[str, Any]:
        """Legacy compatibility view: returns {"sequences": {name: [frames]}}.

        Some callers (e.g. RobotArm.execute_action) still take the
        old actions_map dict. This adapter keeps the call sites stable.
        """
        with self._lock:
            return {
                "sequences": {
                    k: [dict(f) for f in v["frames"]]
                    for k, v in self._action_map.items()
                }
            }

    # ── public write API ────────────────────────────────────────────

    def save(
        self,
        name: str,
        frames: List[Dict[str, Any]],
        *,
        if_match: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save (insert or overwrite) `name` with `frames` (+ optional description).

        Atomic: tempfile + os.replace. In-memory state updates only AFTER
        the replace succeeds.

        Raises ActionsError on validation failure or etag mismatch.
        """
        self._validate_name(name)
        normalized = self._validate_frames(frames, self._required_fields)
        desc = "" if description is None else str(description).strip()

        with self._lock:
            current_etag = self._compute_etag_locked()
            if if_match is not None and if_match != current_etag:
                raise ActionsError(
                    "etag_mismatch",
                    f"If-Match {if_match!r} does not match current etag {current_etag!r}",
                    http_status=412,
                )

            replaced = name in self._action_map
            # Preserve prior description if caller did not provide one on update.
            if description is None and replaced:
                desc = self._action_map[name].get("description", "")

            # Build new full file content first; only mutate memory once
            # the replace succeeds. Preserve any prior response_mode /
            # completion_text on the existing entry so a save() that
            # only changes frames doesn't silently revert custom modes.
            new_map = dict(self._action_map)
            new_entry: Dict[str, Any] = {"description": desc, "frames": normalized}
            if replaced:
                prior = self._action_map.get(name, {})
                prior_rmode = prior.get("response_mode")
                if isinstance(prior_rmode, str) and prior_rmode:
                    new_entry["response_mode"] = prior_rmode
                prior_ctext = prior.get("completion_text")
                if isinstance(prior_ctext, str) and prior_ctext:
                    new_entry["completion_text"] = prior_ctext
            new_map[name] = new_entry
            payload = {"sequences": {k: self._serialize_entry(new_map[k]) for k in sorted(new_map.keys())}}
            serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

            if len(serialized.encode("utf-8")) > MAX_YAML_BYTES:
                raise ActionsError(
                    "yaml_too_large",
                    f"Serialized actions.yaml exceeds {MAX_YAML_BYTES} bytes",
                    http_status=413,
                )

            self._atomic_write(serialized)
            # Only now update in-memory state.
            self._action_map = new_map
            new_etag = self._refresh_etag_locked()

            return {
                "ok": True,
                "name": name,
                "frames_count": len(normalized),
                "etag": new_etag,
                "replaced": replaced,
            }

    # ── internals ───────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """Read actions.yaml at startup (best-effort; missing file = empty)."""
        with self._lock:
            self._action_map = {}
            if not self._path.exists():
                self._etag_cache = None
                return
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except (OSError, yaml.YAMLError) as exc:
                print(f"[ActionsManager] WARN: failed to load {self._path}: {exc}")
                self._etag_cache = None
                return
            if not isinstance(data, dict):
                self._etag_cache = None
                return
            sequences = data.get("sequences") or {}
            # Legacy: collapse any leftover `poses` into single-frame sequences.
            poses = data.get("poses") or {}
            for pname, pdata in (poses.items() if isinstance(poses, dict) else []):
                if pname in sequences:
                    continue
                if isinstance(pdata, dict) and isinstance(pdata.get("joints"), dict):
                    sequences[pname] = [{"joints": pdata["joints"], "delay": 1.5}]
            for sname, sdata in (sequences.items() if isinstance(sequences, dict) else []):
                frames = self._coerce_frames(sdata)
                if frames is None or not isinstance(sname, str) or not NAME_RE.match(sname):
                    continue
                description = ""
                response_mode = ""
                completion_text = ""
                if isinstance(sdata, dict):
                    raw_desc = sdata.get("description", "")
                    if isinstance(raw_desc, str):
                        description = raw_desc.strip()
                    rmode = sdata.get("response_mode", "")
                    if isinstance(rmode, str):
                        response_mode = rmode.strip()
                    ctext = sdata.get("completion_text", "")
                    if isinstance(ctext, str):
                        completion_text = ctext.strip()
                entry: Dict[str, Any] = {
                    "description": description,
                    "frames": frames,
                }
                if response_mode:
                    entry["response_mode"] = response_mode
                if completion_text:
                    entry["completion_text"] = completion_text
                self._action_map[sname] = entry
            self._refresh_etag_locked()

    @staticmethod
    def _serialize_entry(entry: Dict[str, Any]) -> Any:
        """Serialize an in-memory entry back to YAML-friendly shape.

        Mapping shape: {description?, response_mode?, completion_text?, frames}.
        Optional keys are only emitted when set (truthy non-empty string),
        so default-only entries round-trip cleanly without polluting the
        YAML with empty fields. When no metadata at all is set we fall
        back to the legacy bare-list shape that external tooling and
        old fixtures still understand.
        """
        desc = entry.get("description", "")
        rmode = entry.get("response_mode", "")
        ctext = entry.get("completion_text", "")
        frames = entry["frames"]
        has_meta = bool(desc) or (isinstance(rmode, str) and rmode) or (
            isinstance(ctext, str) and ctext
        )
        if not has_meta:
            return frames
        out: Dict[str, Any] = {}
        if desc:
            out["description"] = desc
        if isinstance(rmode, str) and rmode:
            out["response_mode"] = rmode
        if isinstance(ctext, str) and ctext:
            out["completion_text"] = ctext
        out["frames"] = frames
        return out

    @staticmethod
    def _coerce_frames(sdata: Any) -> Optional[List[Dict[str, Any]]]:
        """Accept either a list-of-frames (new) or legacy {frames: [...]}."""
        if isinstance(sdata, list):
            frame_list = sdata
        elif isinstance(sdata, dict) and isinstance(sdata.get("frames"), list):
            frame_list = sdata["frames"]
        else:
            return None
        out: List[Dict[str, Any]] = []
        for frame in frame_list:
            if not isinstance(frame, dict):
                continue
            # Legacy frames could be a flat dict of joint:value with no
            # explicit delay — normalize.
            if "joints" in frame and isinstance(frame["joints"], dict):
                joints = {str(k): float(v) for k, v in frame["joints"].items()}
                delay = float(frame.get("delay", 0.4))
            else:
                joints = {str(k): float(v) for k, v in frame.items() if k != "delay"}
                delay = float(frame.get("delay", 0.4))
            out.append({"joints": joints, "delay": delay})
        return out

    @staticmethod
    def _validate_name(name: Any) -> None:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ActionsError(
                "bad_name",
                f"name must match {NAME_RE.pattern!r}, got {name!r}",
            )

    @staticmethod
    def _validate_frames(
        frames: Any,
        required_fields: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        # ``required_fields`` lets a caller validate against the active
        # actuator's joint set instead of the SO-ARM default. Static
        # callers (observation_server's /actions/preview) may pass the
        # actuator-derived set; when omitted we fall back to the
        # module-level SO-ARM REQUIRED_JOINTS for backward compat.
        required = tuple(required_fields) if required_fields else REQUIRED_JOINTS
        if not isinstance(frames, list) or not frames:
            raise ActionsError("empty_frames", "frames must be a non-empty list")
        if len(frames) > MAX_FRAMES:
            raise ActionsError(
                "too_many_frames",
                f"frames must be <= {MAX_FRAMES}, got {len(frames)}",
            )
        normalized: List[Dict[str, Any]] = []
        for idx, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ActionsError("bad_frame", f"frame[{idx}] is not an object")
            joints = frame.get("joints")
            if not isinstance(joints, dict):
                raise ActionsError("bad_frame", f"frame[{idx}].joints missing/invalid")
            for req in required:
                if req not in joints:
                    raise ActionsError(
                        "missing_joint",
                        f"frame[{idx}] missing joint {req!r}",
                    )
            try:
                joint_values = {str(k): float(joints[k]) for k in required}
            except (TypeError, ValueError) as exc:
                raise ActionsError("bad_frame", f"frame[{idx}] joint not numeric: {exc}") from exc
            try:
                delay = float(frame.get("delay", 0.4))
            except (TypeError, ValueError) as exc:
                raise ActionsError("bad_frame", f"frame[{idx}].delay not numeric: {exc}") from exc
            if not (MIN_DELAY <= delay <= MAX_DELAY):
                raise ActionsError(
                    "bad_delay",
                    f"frame[{idx}].delay {delay} outside [{MIN_DELAY}, {MAX_DELAY}]",
                )
            normalized.append({"joints": joint_values, "delay": delay})
        return normalized

    def _atomic_write(self, serialized: str) -> None:
        """tempfile + os.replace. Raises on any IO failure."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile in the same dir so os.replace is atomic on
        # POSIX (cross-filesystem replace would degrade to copy+unlink).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".actions.",
            suffix=".yaml.tmp",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            # Best-effort cleanup; the in-memory state is untouched
            # because we haven't returned successfully yet.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _compute_etag_locked(self) -> str:
        """Double-factor (size, mtime_ns) check; rehash file only on mismatch."""
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            self._etag_cache = None
            return "sha256:empty"
        cache = self._etag_cache
        if cache is not None and cache.size == st.st_size and cache.mtime_ns == st.st_mtime_ns:
            return cache.etag
        return self._refresh_etag_locked()

    def _refresh_etag_locked(self) -> str:
        try:
            with self._path.open("rb") as fh:
                raw = fh.read()
            st = os.stat(self._path)
        except FileNotFoundError:
            self._etag_cache = None
            return "sha256:empty"
        digest = hashlib.sha256(raw).hexdigest()[:16]
        etag = f"sha256:{digest}"
        self._etag_cache = _EtagCache(etag=etag, size=st.st_size, mtime_ns=st.st_mtime_ns)
        return etag
