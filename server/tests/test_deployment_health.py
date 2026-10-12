"""Compose must reject HTTP 200 from uninitialised model backends."""
import json
from pathlib import Path
from unittest.mock import patch
import urllib.request

import pytest
import yaml


COMPOSE = yaml.safe_load((Path(__file__).resolve().parents[2] / 'deploy/compose.yaml').read_text())


@pytest.mark.parametrize('service,body,expected', [
    ('speech', {'asr': True, 'tts': True}, 0),
    ('speech', {'asr': False, 'tts': True}, 1),
    ('speech', {'asr': True, 'tts': False}, 1),
    ('speech', {}, 1),
    ('llm', {'status': 'ok'}, 0),
    ('llm', {'status': 'unavailable'}, 1),
])
def test_backend_state_controls_compose_health(service, body, expected):
    from io import BytesIO
    command = COMPOSE['services'][service]['healthcheck']['test']
    assert command[:3] == ['CMD', 'python3', '-c']
    with patch.object(urllib.request, 'urlopen', return_value=BytesIO(json.dumps(body).encode())):
        with pytest.raises(SystemExit) as result:
            exec(command[3], {'exit': lambda code: (_ for _ in ()).throw(SystemExit(code))})
    assert result.value.code == expected
