#!/usr/bin/env python3
"""Wait for both native model services before starting the agent."""
import argparse
import json
import math
import socket
import time
import urllib.error
import urllib.request


SERVICES = {
    'speech': 'http://127.0.0.1:8621/health',
    'llm': 'http://127.0.0.1:1828/health',
}


def agent_port_available():
    try:
        connection = socket.create_connection(('127.0.0.1', 18000), timeout=1)
    except ConnectionRefusedError:
        return True
    except OSError as error:
        print(f'Cannot check agent port 18000: {error}', flush=True)
        return False
    connection.close()
    print('Port 18000 is already in use. The agent may already be running; do not start a second instance.', flush=True)
    return False


def probe_service(name, url, request_timeout):
    try:
        with urllib.request.urlopen(url, timeout=request_timeout) as response:
            health = json.load(response)
        if not isinstance(health, dict):
            return False, 'invalid health response'
        if name == 'speech':
            ready = health.get('asr') is True and health.get('tts') is True
            detail = f"ASR={health.get('asr')}, TTS={health.get('tts')}"
        else:
            ready = health.get('status') == 'ok'
            detail = f"status={health.get('status')}"
        return ready, detail
    except urllib.error.HTTPError as error:
        return False, f'HTTP {error.code}'
    except (OSError, ValueError, urllib.error.URLError) as error:
        return False, str(error)


def wait_for_services(timeout, interval=1.0):
    deadline = time.monotonic() + timeout
    details = {name: 'not checked' for name in SERVICES}
    previous = {}
    while time.monotonic() < deadline:
        ready = {}
        for name, url in SERVICES.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready[name], details[name] = probe_service(name, url, min(5.0, remaining))
            state = (ready[name], details[name])
            if previous.get(name) != state:
                print(f"{'Ready' if ready[name] else 'Waiting'}: {name} ({url}): {details[name]}", flush=True)
                previous[name] = state
        if all(ready.get(name, False) for name in SERVICES):
            return True
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))
    print(f'Model services did not become ready within {timeout:g}s. Keep both service terminals running.', flush=True)
    for name, url in SERVICES.items():
        print(f'  {name} ({url}): {details[name]}', flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--check-agent-port', action='store_true', help='Only check for an existing agent listener.')
    args = parser.parse_args()
    if args.check_agent_port:
        return 0 if agent_port_available() else 1
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be a positive finite number')
    return 0 if wait_for_services(args.timeout) else 1


if __name__ == '__main__':
    raise SystemExit(main())
