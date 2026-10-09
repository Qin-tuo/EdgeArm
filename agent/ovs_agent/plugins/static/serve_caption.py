#!/usr/bin/env python3
"""Same-origin gateway for the self-contained live-caption page.

The browser talks ONLY to this gateway (localhost), which forwards to the
backend services. This sidesteps two problems at once:

  * CORS — the /translate HTTP call is same-origin, so no preflight.
  * macOS system proxy — a host-level HTTP/SOCKS proxy (Clash etc.) corrupts
    WebSocket upgrades to Tailscale IPs (100.64.0.0/10 is not in the default
    proxy bypass list), so the browser can't reach the SLV ASR WS directly.
    Routing through localhost (which IS bypassed) fixes it; this Python process
    then connects to the backend directly (aiohttp ClientSession ignores the
    system proxy by default).

Routes:
  GET  /            → live_caption.html
  GET  /asr   (WS)  → proxied to SLV  /asr/stream   (binary mic ↑, JSON text ↓)
  POST /translate   → proxied to NLLB /translate
  GET  /health      → proxied to NLLB /health

Usage:
  uv run --project <repo>/agent python serve_caption.py
  uv run python serve_caption.py --port 18790 \
      --slv ws://100.82.225.102:8621/asr/stream \
      --nllb http://100.82.225.102:9001
Then open http://localhost:18790 and click 开始录音.

Requires aiohttp (already a dependency of the agent — run via `uv run`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

from aiohttp import ClientSession, WSMsgType, web

HTML_PATH = Path(__file__).with_name("live_caption.html")


async def handle_index(request: web.Request) -> web.Response:
    try:
        return web.Response(text=HTML_PATH.read_text(encoding="utf-8"),
                            content_type="text/html")
    except FileNotFoundError:
        return web.Response(status=500, text="live_caption.html not found")


async def handle_translate(request: web.Request) -> web.Response:
    body = await request.read()
    sess: ClientSession = request.app["sess"]
    try:
        async with sess.post(request.app["nllb"] + "/translate", data=body,
                             headers={"Content-Type": "application/json"}) as r:
            return web.Response(status=r.status, body=await r.read(),
                                content_type="application/json")
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"upstream: {e}"}, status=502)


async def handle_health(request: web.Request) -> web.Response:
    sess: ClientSession = request.app["sess"]
    try:
        async with sess.get(request.app["nllb"] + "/health") as r:
            return web.Response(status=r.status, body=await r.read(),
                                content_type="application/json")
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"upstream: {e}"}, status=502)


async def handle_asr_ws(request: web.Request) -> web.WebSocketResponse:
    """Proxy the browser ASR socket to the SLV **/asr/stream** upstream.

    /asr/stream is the dedicated ASR-only endpoint — no TTS/LLM machinery (a
    text-only caption doesn't need voice-out). Since the server-VAD
    multi-utterance fix it stays open across utterances with COMPLETE finals
    (prepare_finalize+finalize), so this is a plain passthrough: binary PCM up,
    ``{type: partial|final}`` JSON down (already the right shape — no translation).
    vad=silero so the server segments utterances.
    """
    client = web.WebSocketResponse(max_msg_size=0)
    await client.prepare(request)
    q = request.query
    params = {
        "language": q.get("language", "auto"),
        "sample_rate": q.get("sample_rate", "16000"),
        "vad": q.get("vad", "silero"),
        "vad_silence_ms": q.get("vad_silence_ms", "400"),
    }
    upstream = request.app["slv"] + "?" + urlencode(params)
    sess: ClientSession = request.app["sess"]
    try:
        async with sess.ws_connect(upstream, max_msg_size=0) as up:
            async def c2u() -> None:
                async for m in client:
                    if m.type == WSMsgType.BINARY:
                        await up.send_bytes(m.data)   # mic PCM
                    elif m.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break
                await up.close()

            async def u2c() -> None:
                async for m in up:
                    if m.type == WSMsgType.TEXT:
                        await client.send_str(m.data)   # already {type:partial|final}
                    elif m.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break
                if not client.closed:
                    await client.close()

            await asyncio.gather(c2u(), u2c())
    except Exception as e:  # noqa: BLE001
        if not client.closed:
            await client.close(message=str(e).encode()[:120])
    return client


async def _on_startup(app: web.Application) -> None:
    # ClientSession must be created inside the running loop. trust_env stays
    # False (default) → upstream calls ignore the macOS/shell proxy.
    app["sess"] = ClientSession()


async def _on_cleanup(app: web.Application) -> None:
    await app["sess"].close()


def main() -> None:
    ap = argparse.ArgumentParser(description="live-caption gateway")
    ap.add_argument("--port", type=int, default=18790)
    ap.add_argument("--slv", default="ws://100.82.225.102:8621/asr/stream",
                    help="SLV ASR WebSocket upstream")
    ap.add_argument("--nllb", default="http://100.82.225.102:9001",
                    help="NLLB translator base URL")
    args = ap.parse_args()

    app = web.Application()
    app["slv"] = args.slv
    app["nllb"] = args.nllb.rstrip("/")
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/caption", handle_index)
    app.router.add_get("/asr", handle_asr_ws)
    app.router.add_post("/translate", handle_translate)
    app.router.add_get("/health", handle_health)

    print(f"live-caption gateway → http://localhost:{args.port}")
    print(f"  ASR  WS  → {args.slv}")
    print(f"  translate → {args.nllb}")
    print("  open the URL, pick languages, click 开始录音")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
