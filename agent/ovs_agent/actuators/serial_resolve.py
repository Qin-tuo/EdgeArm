"""Resolve a serial actuator's ``/dev/tty*`` port by *stable USB identity*.

Why this exists
---------------
Picking the right serial port for a robot arm by ``/dev/ttyACMx`` number is
unsafe on a box that hosts more than one arm: the kernel's enumeration order is
not stable across reboots / hotplug. We confirmed this in the field — a
B601-DM that the runbook documented as ``ttyACM1`` enumerated as ``ttyACM0``
when it was the only arm plugged in. Selecting by number can therefore grab the
*wrong* arm, which for a robot is a safety incident, not a nuisance.

The fix: match on the device's USB identity (VID:PID primarily, with
vendor/product/serial string fallbacks), which is stable across reboots and
ports. Each actuator declares its own match spec, so two arms on the same host
never collide. Identity is read from sysfs (``/sys/class/tty/<dev>/device`` →
walk up to the USB node's ``idVendor``/``idProduct``/...), which needs no root.

Safety contract
---------------
- An explicit path (``/dev/ttyACM1`` or a ``/dev/serial/by-id/...`` symlink)
  ALWAYS wins and is returned as a realpath — the operator's escape hatch.
- ``channel='auto'`` requires a ``match`` spec; with none we refuse rather than
  guess.
- **0 or >1 matches → raise.** We never fall back to "the first serial port"
  (that is exactly how you grab the neighbouring arm). On ambiguity the operator
  must tighten the spec (add the per-unit ``serial``) or pin an explicit path.
- Output is always a realpath ``/dev/ttyACMx`` so SDKs that branch on
  ``channel.startswith('/dev/tty')`` (serial vs SocketCAN) behave correctly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Match-spec keys. Each maps to a list of wanted values:
#   usb_id   — "vid:pid" exact (case-insensitive), the most stable key.
#   vendor   — substring of the USB manufacturer / ID_VENDOR string.
#   product  — substring of the USB product / ID_MODEL string.
#   serial   — exact per-unit serial (pin one specific physical device).
_SPEC_KEYS = ("usb_id", "vendor", "product", "serial")


def _default_log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _list_serial_ttys() -> list[str]:
    """All candidate serial device basenames currently present under /dev."""
    out: list[str] = []
    for pat in ("ttyACM*", "ttyUSB*"):
        out.extend(p.name for p in Path("/dev").glob(pat))
    return sorted(out)


def _usb_identity(tty_name: str) -> dict[str, str]:
    """Read the USB identity of a serial tty from sysfs (no root needed).

    Walks up from ``/sys/class/tty/<tty>/device`` (the USB *interface*) to the
    parent USB *device* node that carries ``idVendor``/``idProduct``.
    """
    base = Path(f"/sys/class/tty/{tty_name}/device")
    try:
        node = base.resolve(strict=True)
    except OSError:
        return {}

    def _read(n: Path, name: str) -> str:
        try:
            return (n / name).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    for _ in range(6):  # interface → device is 1-2 hops; bound the walk
        if (node / "idVendor").exists():
            return {
                "vid": _read(node, "idVendor").lower(),
                "pid": _read(node, "idProduct").lower(),
                "vendor": _read(node, "manufacturer"),
                "product": _read(node, "product"),
                "serial": _read(node, "serial"),
            }
        if node.parent == node:
            break
        node = node.parent
    return {}


def _spec_matches(ident: Mapping[str, str], spec: Mapping[str, Sequence[str]] | None) -> bool:
    """True if *ident* satisfies ANY rule in *spec* (empty/None spec → False)."""
    if not spec:
        return False
    usb_id = f"{ident.get('vid', '')}:{ident.get('pid', '')}"
    for want in spec.get("usb_id", []) or []:
        if str(want).lower() == usb_id:
            return True
    vendor = ident.get("vendor", "").lower()
    for want in spec.get("vendor", []) or []:
        if str(want).lower() in vendor:
            return True
    product = ident.get("product", "").lower()
    for want in spec.get("product", []) or []:
        if str(want).lower() in product:
            return True
    for want in spec.get("serial", []) or []:
        if str(want) == ident.get("serial", ""):
            return True
    return False


def resolve_serial_port(
    channel: str,
    *,
    match: Mapping[str, Sequence[str]] | None = None,
    exclude: Mapping[str, Sequence[str]] | None = None,
    ambiguous: str = "error",
    retry_s: float = 10.0,
    log: Callable[[str], None] = _default_log,
) -> str:
    """Resolve *channel* to a realpath ``/dev/ttyACMx``.

    ``channel``: an explicit ``/dev/tty*`` path, a ``/dev/serial/by-id/...``
    symlink, or ``'auto'`` (match by USB identity).

    See the module docstring for the safety contract. Raises ``ValueError`` for
    a malformed request (``auto`` without a ``match`` spec) and ``RuntimeError``
    when ``auto`` finds zero or — under ``ambiguous='error'`` — multiple
    matches.
    """
    ch = (channel or "").strip()
    if ch.lower() != "auto":
        # Explicit path / by-id symlink: operator escape hatch, always wins.
        return os.path.realpath(ch)

    if not match:
        raise ValueError(
            "channel='auto' requires a 'match' spec "
            f"({'/'.join(_SPEC_KEYS)}); none provided. "
            "Set an explicit channel path or add channel_match in config."
        )

    attempts = max(1, int(retry_s))
    seen: list[str] = []
    for i in range(attempts):
        seen = _list_serial_ttys()
        candidates: list[tuple[str, dict[str, str]]] = []
        for tty in seen:
            ident = _usb_identity(tty)
            if _spec_matches(ident, match) and not _spec_matches(ident, exclude):
                candidates.append((tty, ident))

        if len(candidates) == 1:
            tty, ident = candidates[0]
            path = os.path.realpath(f"/dev/{tty}")
            log(
                f"[serial] matched {tty} "
                f"({ident.get('vid')}:{ident.get('pid')} {ident.get('product')!r} "
                f"serial={ident.get('serial')!r}) -> {path}"
            )
            return path

        if len(candidates) > 1:
            listing = ", ".join(
                f"{t}({d.get('vid')}:{d.get('pid')} {d.get('product')!r})"
                for t, d in candidates
            )
            if ambiguous == "first":
                tty, ident = sorted(candidates)[0]
                path = os.path.realpath(f"/dev/{tty}")
                log(
                    f"[serial] ⚠ {len(candidates)} devices match "
                    f"({listing}); ambiguous=first -> {path}"
                )
                return path
            raise RuntimeError(
                f"channel='auto' is ambiguous: {len(candidates)} devices match "
                f"[{listing}]. Tighten the match spec (add the per-unit 'serial') "
                "or set an explicit channel path."
            )

        # No match yet — USB may still be enumerating after a cold boot/recreate.
        if i < attempts - 1:
            time.sleep(1.0)

    raise RuntimeError(
        f"channel='auto' found no serial device matching {dict(match)} after "
        f"{retry_s:.0f}s. Serial ttys seen: {seen or '(none)'}. "
        "Set an explicit channel path (e.g. /dev/ttyACM1)."
    )
