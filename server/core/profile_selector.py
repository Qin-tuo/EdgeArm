"""Single product preset selector; unknown hardware requires explicit selection."""
import os
TIER_RK3588 = "rk3588"
TIER_UNKNOWN = "unknown"
PRESET_TABLE = {(TIER_RK3588, "so101"): "rk3588-default"}
KNOWN_PRESETS = ["so101"]
class UnsupportedPreset(ValueError):
    pass

def detect_device_tier():
    value = os.environ.get("OVS_DEVICE_TIER")
    if value:
        return value
    try:
        from pathlib import Path
        if "rk3588" in Path("/proc/device-tree/compatible").read_text().lower():
            return TIER_RK3588
    except OSError:
        pass
    return TIER_UNKNOWN

def resolve_profile_name(preset, device_tier=None):
    key = (device_tier or detect_device_tier(), preset)
    if key not in PRESET_TABLE:
        raise UnsupportedPreset(f"SO-101 requires RK3588 and preset so101: {key}")
    return PRESET_TABLE[key]
