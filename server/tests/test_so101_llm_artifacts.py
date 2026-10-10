"""Offline integrity checks; tiny fixtures, no model downloads or native worker."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'so101_artifacts', Path(__file__).resolve().parents[2] / 'services/rk1828-llm/verify_artifacts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def artifacts(tmp_path):
    entries = []
    for i in range(4):
        data = f'fixture{i}'.encode()
        path = tmp_path / f'model{i}'
        path.write_bytes(data)
        entries.append(dict(name=path.name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(files=entries)))
    return manifest, tmp_path, entries


def test_cached_equal_size_corruption_fails_integrity(artifacts):
    manifest, root, entries = artifacts
    module.verify(manifest, root)
    (root / entries[0]['name']).write_bytes(b'x' * entries[0]['size_bytes'])
    with pytest.raises(ValueError, match='SHA256'):
        module.verify(manifest, root)


def test_missing_or_partial_export_cannot_load(artifacts):
    manifest, root, entries = artifacts
    (root / entries[1]['name']).unlink()
    with pytest.raises(FileNotFoundError):
        module.verify(manifest, root)
    manifest.write_text(json.dumps(dict(files=entries[:3])))
    with pytest.raises(ValueError, match='matched four-file'):
        module.verify(manifest, root)


@pytest.mark.parametrize('field,value', [('name', '../escape'), ('sha256', ''), ('size_bytes', 0)])
def test_invalid_locks_fail_closed(artifacts, field, value):
    manifest, root, entries = artifacts
    entries[0][field] = value
    manifest.write_text(json.dumps(dict(files=entries)))
    with pytest.raises(ValueError, match='invalid'):
        module.verify(manifest, root)
