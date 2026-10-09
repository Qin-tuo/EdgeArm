"""Dashboard /api/translator/runtime endpoints (Phase 2a).

Verifies:
  - GET returns backend / src_lang / tgt_lang / supported_targets
  - PATCH with a legal tgt_lang updates ``config.translator_tgt_lang``
  - PATCH with malformed / unsupported / missing body returns 400
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from ovs_agent.app_mode import AppMode, ModeManager
from ovs_agent.config import Config
from ovs_agent.event_bus import EventBus
from ovs_agent.plugins.debug_dashboard import DebugDashboardPlugin


class _Chat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test chat"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, config: Config):
    app = MagicMock()
    app.config = config
    app.events = EventBus()
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(
        _ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock()
    )
    app.session = SimpleNamespace(history=[])
    app.broadcast = AsyncMock()
    mgr = ModeManager(lambda: None)
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    return app


def _cfg(port: int, **overrides) -> Config:
    base = dict(
        metadata={"dashboard_port": port},
        translator_backend="ctranslate2",
        translator_url="http://localhost:9001",
        translator_src_lang="zho_Hans",
        translator_tgt_lang="eng_Latn",
        translator_timeout_s=5.0,
    )
    base.update(overrides)
    return Config(**base)


@pytest.mark.asyncio
async def test_translator_runtime_get_returns_full_payload(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/translator/runtime")
            assert r.status == 200
            data = await r.json()
        assert data["backend"] == "ctranslate2"
        assert data["src_lang"] == "zho_Hans"
        assert data["tgt_lang"] == "eng_Latn"
        assert data["url"] == "http://localhost:9001"
        assert data["timeout_s"] == 5.0
        # supported_targets covers at least the 11 NLLB targets we ship.
        targets = data["supported_targets"]
        assert isinstance(targets, list)
        assert len(targets) >= 11
        codes = {t["code"] for t in targets}
        assert {"eng_Latn", "zho_Hans", "jpn_Jpan"}.issubset(codes)
        for t in targets:
            assert set(t.keys()) == {"code", "name"}
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_updates_target_language(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "jpn_Jpan"},
            )
            assert r.status == 200
            data = await r.json()
        assert data["ok"] is True
        assert data["tgt_lang"] == "jpn_Jpan"
        # No _source_path on this in-memory config → persistence skipped.
        assert data["persisted"] is False
        # Live config mutated.
        assert cfg.translator_tgt_lang == "jpn_Jpan"
        # Broadcast fired with the runtime payload.
        app.broadcast.assert_awaited()
        evt_name, evt_payload = app.broadcast.await_args.args
        assert evt_name == "on_translator_runtime_change"
        assert evt_payload["tgt_lang"] == "jpn_Jpan"
        assert evt_payload["persisted"] is False
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_persists_to_yaml(unused_tcp_port, tmp_path):
    """When config has a _source_path, PATCH writes the new tgt_lang back."""
    import yaml

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "translator_backend": "ctranslate2",
                "translator_url": "http://localhost:9001",
                "translator_src_lang": "zho_Hans",
                "translator_tgt_lang": "eng_Latn",
                "translator_timeout_s": 5.0,
                "pipeline_mode": "always_on",
            },
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg = _cfg(unused_tcp_port)
    cfg._source_path = str(yaml_path)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "jpn_Jpan"},
            )
            assert r.status == 200
            data = await r.json()
        assert data["ok"] is True
        assert data["persisted"] is True
        # YAML file actually updated on disk.
        on_disk = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert on_disk["translator_tgt_lang"] == "jpn_Jpan"
        # Other keys preserved.
        assert on_disk["translator_backend"] == "ctranslate2"
        assert on_disk["pipeline_mode"] == "always_on"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_malformed_code(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "english"},  # wrong format
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "NLLB format" in data["error"]
        # Config NOT mutated.
        assert cfg.translator_tgt_lang == "eng_Latn"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_unsupported_code(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # Well-formed FLORES code but not in our supported_targets.
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "vie_Latn"},
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "supported" in data["error"]
        assert cfg.translator_tgt_lang == "eng_Latn"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_rejects_missing_field(unused_tcp_port):
    cfg = _cfg(unused_tcp_port)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={},
            )
            assert r.status == 400
            data = await r.json()
        assert data["ok"] is False
        assert "tgt_lang" in data["error"]
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_unwritable_yaml_returns_persist_error(
    unused_tcp_port, tmp_path,
):
    """When the yaml file can't be written, runtime change still succeeds
    but persisted=False + persist_error is surfaced in both HTTP response
    and the broadcast payload."""
    bad_path = tmp_path / "does" / "not" / "exist" / "config.yaml"
    cfg = _cfg(unused_tcp_port)
    cfg._source_path = str(bad_path)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "jpn_Jpan"},
            )
            assert r.status == 200
            data = await r.json()
        assert data["ok"] is True
        assert data["persisted"] is False
        assert "persist_error" in data
        # Runtime config still mutated despite persist failure.
        assert cfg.translator_tgt_lang == "jpn_Jpan"
        # Broadcast payload also carries persist_error so WS subscribers
        # can react to the failure without polling the HTTP endpoint.
        evt_name, evt_payload = app.broadcast.await_args.args
        assert evt_name == "on_translator_runtime_change"
        assert evt_payload["persisted"] is False
        assert "persist_error" in evt_payload
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_translator_runtime_patch_same_value_skips_persist(
    unused_tcp_port, tmp_path,
):
    """PATCHing the current value is not a noop (still broadcasts) but
    skips the disk write to avoid spurious yaml churn."""
    import yaml

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"translator_tgt_lang": "eng_Latn"},
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    mtime_before = yaml_path.stat().st_mtime_ns
    cfg = _cfg(unused_tcp_port)  # already has translator_tgt_lang="eng_Latn"
    cfg._source_path = str(yaml_path)
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.patch(
                base + "/api/translator/runtime",
                json={"tgt_lang": "eng_Latn"},
            )
            assert r.status == 200
            data = await r.json()
        assert data["ok"] is True
        # Same value → not persisted (avoid spurious yaml churn).
        assert data["persisted"] is False
        # File on disk was NOT touched (mtime unchanged).
        assert yaml_path.stat().st_mtime_ns == mtime_before
        # Broadcast still fired so other tabs reflect the (no-op) state.
        app.broadcast.assert_awaited()
    finally:
        await plugin.stop()
