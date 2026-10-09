#!/usr/bin/env python3
"""Register local model, dataset or artifact files without copying their contents."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_record(file):
    path = Path(file).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--kind", choices=["model", "dataset", "artifact"], required=True)
    parser.add_argument("--source", required=True, help="Declared public source URL or independent origin")
    parser.add_argument("--license", required=True, help="Declared license; this tool does not validate permission")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        if not all(value.strip() for value in [args.id, args.source, args.license]):
            raise ValueError("id, source and license must not be empty")
        paths = [file.resolve(strict=True) for file in args.files]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate input file")
        if args.output.resolve() in paths:
            raise ValueError("manifest must not overwrite an input file")
        data = {
            "schema_version": 1, "id": args.id, "kind": args.kind,
            "source": args.source, "declared_license": args.license,
            "permission_verified": False,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": [file_record(file) for file in paths],
        }
        # Exclusive creation prevents an accidental replacement of a frozen manifest.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

