#!/usr/bin/env python3
"""Run in-process software tests. Report restricted IPC explicitly.

The managed execution sandbox may permit socketpair creation but deny send().
Asyncio's thread wakeup pipe then cannot wake select(). Bound selector waits in
THIS test runner only; no fake device, socket, backend result or assertion is
injected. Hardware/latency acceptance must run outside this runner.
"""
import asyncio
import os
from pathlib import Path
import selectors
import socket
import sys

# Use only the project's test plugins. Host ROS/colcon plugins may pull in
# unrelated dependencies during discovery even in a voice-only environment.
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / 'agent')]

x, y = socket.socketpair()
try:
    x.send(b'x')
except PermissionError:
    print('Restricted IPC: test-only selector polling enabled; hardware/timing checks excluded.', flush=True)
    class BoundedSelector(selectors.DefaultSelector):
        def select(self, timeout=None):
            return super().select(0.01 if timeout is None else min(0.01, timeout))
    class TestPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            return asyncio.SelectorEventLoop(BoundedSelector())
    asyncio.set_event_loop_policy(TestPolicy())
finally:
    x.close()
    y.close()

import pytest

class EnvironmentGates:
    def pytest_configure(self, config):
        config.addinivalue_line("markers", "network: real loopback sockets; requires a socket-enabled host")
    def pytest_collection_modifyitems(self, config, items):
        probe = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            for item in items:
                if item.get_closest_marker("network"):
                    item.add_marker(pytest.mark.skip(reason="sandbox denies TCP bind; rerun on target host"))
        finally:
            if probe is not None:
                probe.close()

args = sys.argv[1:] or ['agent/tests', 'server/tests', 'tests']
result = pytest.main(['--import-mode=importlib', '-q', *args], plugins=['pytest_asyncio.plugin', EnvironmentGates()])
# If an upstream test leaks a blocking executor, show its stack during
# interpreter shutdown rather than reporting a silent successful hang.
import faulthandler
faulthandler.dump_traceback_later(15)
raise SystemExit(result)
