"""Selected-profile drift checks; no native engine/model/device initialization."""
import copy
import json
from pathlib import Path

import pytest

from server.core import rk_profile_contract as contract
from server.core import profile_selector


def selected():
    return json.loads((Path(__file__).resolve().parents[2] / 'configs/profiles/rk3588-default.json').read_text())


def test_selected_profile_matches_its_runtime_contract():
    profile = selected()
    status = contract.runtime_status(profile, profile['env'])
    assert status['required'] and status['verified']
    assert status['device'] == 'rk3588'


@pytest.mark.parametrize('key', ['RK_PLATFORM', 'MATCHA_MIN_MEL_FRAMES', 'VOCOS_FRAMES', 'ASR_NPU_CORE_MASK'])
def test_missing_runtime_settings_are_not_verified(key):
    profile = selected()
    runtime = dict(profile['env'])
    del runtime[key]
    status = contract.runtime_status(profile, runtime)
    assert not status['verified'] and key in status['missing_runtime']


def test_wrong_platform_is_drift_even_with_complete_settings():
    profile = selected()
    runtime = dict(profile['env'], RK_PLATFORM='rk3576')
    status = contract.runtime_status(profile, runtime)
    assert not status['verified'] and 'RK_PLATFORM' in status['mismatches']
    altered = copy.deepcopy(profile)
    altered['env']['RK_PLATFORM'] = 'rk3576'
    assert not contract.runtime_status(altered, altered['env'])['verified']


def test_only_so101_rk3588_preset_is_selectable():
    assert profile_selector.resolve_profile_name('so101', 'rk3588') == 'rk3588-default'
    for preset, board in [('default', 'rk3588'), ('so101', 'rk3576'), ('so101', 'jetson'), ('so101', 'unknown')]:
        with pytest.raises(profile_selector.UnsupportedPreset):
            profile_selector.resolve_profile_name(preset, board)
