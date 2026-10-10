"""Fail closed on missing/corrupt RK1828 models, including cached/manual files."""
import hashlib
import json
from pathlib import Path
import re
import sys


def verify(manifest_path, model_dir):
    files = json.loads(Path(manifest_path).read_text())['files']
    if len(files) != 4 or len({f['name'] for f in files}) != 4:
        raise ValueError('RK1828 requires one matched four-file export')
    for entry in files:
        name, size, expected = entry['name'], entry['size_bytes'], entry['sha256']
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid artifact filename')
        if type(size) is not int or size <= 0 or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError(f'invalid integrity lock: {name}')
        path = Path(model_dir) / name
        if path.stat().st_size != size:
            raise ValueError(f'artifact size mismatch: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'artifact SHA256 mismatch: {name}')


if __name__ == '__main__':
    verify(sys.argv[1], sys.argv[2])
    print('RK1828 artifact sizes and SHA256: OK (inference not tested)', flush=True)
