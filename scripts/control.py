#!/usr/bin/env python3
"""Local operator client; no enable tool is exposed to the LLM."""
import argparse
import asyncio
import json
from pathlib import Path
import aiohttp

async def run(args):
    connector = aiohttp.UnixConnector(path=args.socket)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=5)) as session:
        method = session.get if args.action == 'status' else session.post
        async with method('http://localhost/' + args.action) as response:
            data = await response.json()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if response.ok else 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['status', 'stop', 'enable'])
parser.add_argument('--socket', default=str(Path(__file__).resolve().parents[1] / '.local/run/edgearm/control.sock'))
args = parser.parse_args()
try:
    raise SystemExit(asyncio.run(run(args)))
except (OSError, aiohttp.ClientError) as exc:
    parser.exit(1, f'Operator control unavailable: {exc}\n')
