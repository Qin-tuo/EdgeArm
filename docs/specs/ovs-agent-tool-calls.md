# OpenVoiceStream Agent — Tool Calling (Local Function-Call Framework)

**Status**: spec draft, pending review
**Owner**: harvest
**Date**: 2026-05-25
**Related**: `evals/mcp_llm/` in `warehouse_system` repo (reference impl), `agent/ovs_agent/llm/edge_llm.py` (existing prefix_cache fallback)

## Motivation

`ovs_agent` currently has no way to invoke tools. The LLM
backend yields plain text deltas, the session model only knows
`{role, content}`, and `app_mode.py` runs a single-pass stream→TTS
pipeline. Voice users frequently ask for actions ("what time is it",
"switch to translation mode", "turn off the kitchen light") that need
function execution, not just text completion.

This spec adds **local, in-process tool calling** (no MCP server, no
subprocess) with the option to add MCP later as an adapter.

## Goals

1. Streaming OpenAI-compatible tool_calls support (parse `delta.tool_calls`).
2. Multi-turn pump: assistant(tool_calls) → execute → role:tool → continue.
3. Local `@tool` decorator registry — write a Python function, get a tool.
4. Per-mode allowlist; tools off by default.
5. Compatible with existing `EdgeLLMBackend` prefix_cache (server-side
   responsibility — client must keep tools list stable per mode).
6. Cancel-safe: barge-in mid-tool-execution must not poison `Session.history`.

## Non-goals (for v1)

- MCP server integration (kept as future adapter; spec leaves a hook).
- Parallel tool execution (sequential is fine for voice; latency budget tight).
- Streaming tool *results* back to LLM in chunks (results are short JSON).
- Tool sandboxing / capability containment (every tool is trusted code in-process).

## Architecture

```
┌─ session.py ─────────────────────────────────────────────────────┐
│  Msg model adds tool_calls / role=tool / tool_call_id              │
│  trim_to_budget treats {assistant(tool_calls), tool*} as one turn  │
└────────────────────────────────────────────────────────────────────┘
                            ▲
                            │ history
                            │
┌─ app_mode.py ──────────────────────────────────────────────────────┐
│  run_default_dialogue_turn → runner.stream_with_tools(...)          │
│  on_assistant_token → slv.send_text   (TTS unchanged)               │
└────────────────────────────────────────────────────────────────────┘
                            │ events
                            ▼
┌─ tools/runner.py ──────────────────────────────────────────────────┐
│  multi-turn pump (lifted from warehouse_system/run.py:88-180)       │
│  accumulates tool_call deltas, dispatches to registry, appends      │
│  role:tool, continues until finish_reason!="tool_calls" or cap      │
└────────────────────────────────────────────────────────────────────┘
                            │ stream_events
                            ▼
┌─ llm/openai_compat.py + edge_llm.py ───────────────────────────────┐
│  NEW stream_events(messages, *, tools=None, session=None, **kw)    │
│      → AsyncIterator[LLMEvent]                                      │
│  delta.content → LLMEvent(kind="text", text=...)                    │
│  delta.tool_calls (per index) → tool_call_delta / tool_call_done    │
│  finish_reason → LLMEvent(kind="finish", finish_reason=...)         │
│  Existing prefix_cache + A4 fallback wraps this transparently.      │
│  Backwards-compat: old .stream() filters to text-only deltas.       │
└────────────────────────────────────────────────────────────────────┘
                            │ tools=[...]
                            ▲
┌─ tools/registry.py ────────────────────────────────────────────────┐
│  @tool decorator: builds JSON Schema from type hints                │
│  list_openai_tools(allow) → OpenAI tools[] format                   │
│  dispatch(name, args) → ToolResult dict                             │
└────────────────────────────────────────────────────────────────────┘
```

## Detailed Design

### 1. LLM backend: `stream_events` channel (`llm/base.py`)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class LLMEvent:
    kind: Literal["text", "tool_call_delta", "finish"]
    # text fields
    text: str | None = None
    # tool_call_delta fields (per OpenAI index-based accumulation)
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    name: str | None = None             # function.name fragment
    arguments: str | None = None        # function.arguments fragment (may be partial JSON)
    # finish field
    finish_reason: str | None = None    # "stop" | "tool_calls" | "length" | "error"

class LLMBackend(ABC):
    @abstractmethod
    async def stream_events(
        self, messages, *, tools=None, **kw
    ) -> AsyncIterator[LLMEvent]: ...

    async def stream(self, messages, **kw):
        """Backward-compat text-only wrapper. Existing callers unchanged."""
        async for ev in self.stream_events(messages, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text
```

### 1a. `OpenAICompatBackend.stream_events`

Refactor `_do_stream` to yield `LLMEvent` instead of `str`:

```python
async for chunk in response:
    # cache_metrics extraction unchanged.
    # finish_reason=="error" still raises LLMStreamError.
    try:
        choice0 = chunk.choices[0]
        delta = choice0.delta
        finish_reason = choice0.finish_reason
    except (IndexError, AttributeError):
        continue
    if delta.content:
        yield LLMEvent(kind="text", text=delta.content)
    for tc in (delta.tool_calls or []):
        idx = tc.index if tc.index is not None else 0
        yield LLMEvent(
            kind="tool_call_delta",
            tool_call_index=idx,
            tool_call_id=getattr(tc, "id", None),
            name=getattr(tc.function, "name", None) if tc.function else None,
            arguments=getattr(tc.function, "arguments", None) if tc.function else None,
        )
    if finish_reason:
        yield LLMEvent(kind="finish", finish_reason=finish_reason)
```

Add `tools=None` to forward as request kwarg when present. Retain
existing transient-failure retry (A3) and `_retry_disabled` plumbing.

### 1b. `EdgeLLMBackend.stream_events`

Just rename today's `stream` → `stream_events`, change the body's
`yield delta` → `yield ev` (forward all event kinds). Prefix-cache
detection and A4 fallback are unaffected — they catch exceptions,
not specific event kinds.

### 2. Session model (`session.py`)

**Extend `Msg`** beyond `{role, content}`:

```python
# Pseudo: history stays a list[dict] for JSON-trivial serialization.
# New optional keys on a single dict:
#   {"role": "assistant", "content": None,
#    "tool_calls": [{"id": "...", "type": "function",
#                    "function": {"name": "...", "arguments": "{...}"}}]}
#   {"role": "tool", "tool_call_id": "...", "content": "<json string>"}
```

**New helpers**:

```python
def add_assistant_tool_calls(self, content: str | None, tool_calls: list[dict]):
    self.history.append({
        "role": "assistant",
        "content": content,        # may be None or a "let me check..." preamble
        "tool_calls": tool_calls,
    })

def add_tool_result(self, tool_call_id: str, content: str):
    self.history.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    })

def rollback_to(self, anchor: int) -> int:
    """Truncate ``self.history`` back to ``anchor`` length.
    Returns number of messages dropped.

    Sole rollback API — replaces any per-purpose `rollback_*` helpers.
    Used on cancel / error / iteration-cap to keep history strict-valid
    for OpenAI (no orphan ``assistant(tool_calls)`` without matching
    ``role:tool`` followup, no truncated tool round)."""
```

**Trim invariant change**: `_trim_to_budget` currently groups messages
in (user, assistant) pairs. With tools, a logical *turn* is:

```
user → assistant(tool_calls)? → tool* → assistant(tool_calls)? → tool* → ... → assistant(text)
```

Replace pair grouping with **turn detection**: start at a `user` message,
consume contiguous non-user messages until the next `user` (or EOF).
A "trailing" turn is one without a terminal `assistant(text)` — pin to tail.

Drop oldest *whole turns*; never split. Same 0.75 budget rule.

**echo-recovery**: today's `_maybe_recover_from_echo` walks
`m for m in history if m.role == "assistant"` — keep that but skip
assistant messages whose `content is None` (tool-call-only messages).

### 3. Tools module (`agent/ovs_agent/tools/`)

#### 3a. `registry.py`

```python
import inspect, asyncio
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema (OpenAI-style)
    fn: Callable               # sync or async; **kwargs from JSON args
    timeout_s: float = 10.0

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def tool(self, *, name=None, description="", timeout_s=10.0):
        def deco(fn):
            sig = inspect.signature(fn)
            hints = get_type_hints(fn)
            props, required = {}, []
            for pname, param in sig.parameters.items():
                if pname == "ctx":
                    continue            # injected by registry, not LLM-visible
                t = hints.get(pname, str)
                props[pname] = _py_type_to_schema(t)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            params = {"type": "object", "properties": props}
            if required:
                params["required"] = required
            tname = name or fn.__name__
            self._tools[tname] = Tool(
                name=tname,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn, timeout_s=timeout_s,
            )
            return fn
        return deco

    def list_openai_tools(self, allow: set[str] | None = None) -> list[dict]:
        ...   # see warehouse_system/mcp_client.py:mcp_tools_to_openai

    async def dispatch(self, name: str, arguments: dict, ctx) -> dict:
        t = self._tools.get(name)
        if t is None:
            return {"success": False, "error": f"unknown tool: {name}"}
        allowed = set(t.parameters.get("properties", {}).keys())
        clean = {k: v for k, v in (arguments or {}).items() if k in allowed}
        try:
            if "ctx" in inspect.signature(t.fn).parameters:
                clean["ctx"] = ctx
            result = t.fn(**clean)
            if inspect.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=t.timeout_s)
            return result if isinstance(result, dict) else {"value": result}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"tool {name} timed out after {t.timeout_s}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

`_py_type_to_schema` handles: `str → "string"`, `int → "integer"`,
`float → "number"`, `bool → "boolean"`, `list → "array"`,
`dict → "object"`, `Literal[...] → enum`, `Optional[T] → schema(T)` (no nullable).

Global default instance `default_registry` for app-base wiring; tests
can construct dedicated `ToolRegistry()` for isolation.

#### 3b. `builtin.py`

Minimal example set:

```python
from datetime import datetime
from .registry import default_registry as r

@r.tool(description="Return the current local time as ISO 8601.")
def time_now() -> dict:
    return {"now": datetime.now().isoformat()}

@r.tool(description="Switch the agent to a different mode.")
def set_mode(mode_name: str, ctx) -> dict:
    """mode_name: target mode key from app config."""
    mm = ctx.mode_manager
    if mm is None or mode_name not in mm.available_modes():
        return {"success": False, "error": f"unknown mode: {mode_name}"}
    mm.request_switch(mode_name)
    return {"success": True, "mode": mode_name}
```

#### 3c. `runner.py`

```python
@dataclass
class ToolCallCtx:
    session: Session
    mode_manager: Any
    event_bus: Any
    config: Any

@dataclass
class _ToolCallAcc:
    id: str = ""
    name: str = ""
    arguments: str = ""

async def stream_with_tools(
    llm,
    messages: list[dict],
    *,
    session: Session,
    registry: ToolRegistry,
    allowed_tools: set[str],
    ctx: ToolCallCtx,
    max_iterations: int = 5,
    on_assistant_token,           # async (token: str) -> None
    on_tool_started=None,         # async (call: dict) -> None
    on_tool_completed=None,       # async (call: dict, result: dict, dt_ms: float) -> None
    llm_kwargs: dict | None = None,
) -> str:
    """Runs LLM ↔ tool round trips until a text-only final answer.

    Mutates ``session.history`` AND ``messages`` in lock-step. Returns
    final assistant text (also appended to session.history).

    On cancel: rolls back any (assistant_tool_calls + tool_results)
    appended during this call so session.history stays strict-valid.
    """
    tools_schema = registry.list_openai_tools(allowed_tools) or None
    iterations_done = 0
    rollback_anchor = len(session.history)
    try:
        for iter_idx in range(max_iterations):
            iterations_done += 1
            text_chunks = []
            tool_accs: dict[int, _ToolCallAcc] = {}
            finish_reason = None

            kwargs = dict(llm_kwargs or {})
            kwargs["session"] = session
            kwargs["tools"] = tools_schema
            # Must-fix #1 (codex review): on tool-loop iterations
            # *after* the first, the message list shape has changed
            # (assistant_tool_calls + tool results appended). The
            # server-side formatted_request cache lookup was warmed
            # against the original shape; sending prefix_cache=True
            # again risks a stale-hit / mismatch. Force the no-prefix
            # path for iterations >0 by injecting extra_body.
            if iter_idx > 0:
                caller_extra = dict(kwargs.get("extra_body") or {})
                caller_extra["prefix_cache"] = False
                kwargs["extra_body"] = caller_extra

            async for ev in llm.stream_events(messages, **kwargs):
                if ev.kind == "text" and ev.text:
                    text_chunks.append(ev.text)
                    await on_assistant_token(ev.text)
                elif ev.kind == "tool_call_delta":
                    slot = tool_accs.setdefault(ev.tool_call_index, _ToolCallAcc())
                    if ev.tool_call_id:
                        slot.id = ev.tool_call_id
                    if ev.name:
                        slot.name = ev.name
                    if ev.arguments:
                        slot.arguments += ev.arguments
                elif ev.kind == "finish":
                    finish_reason = ev.finish_reason

            if not tool_accs or finish_reason != "tool_calls":
                final_text = "".join(text_chunks)
                if final_text:
                    session.add_assistant(final_text)
                return final_text

            # Commit assistant(tool_calls) to messages + session.
            preamble = "".join(text_chunks) or None
            tc_payload = [
                {
                    "id": acc.id or f"call_{idx}",
                    "type": "function",
                    "function": {"name": acc.name, "arguments": acc.arguments or "{}"},
                }
                for idx, acc in sorted(tool_accs.items())
            ]
            messages.append({"role": "assistant", "content": preamble,
                             "tool_calls": tc_payload})
            session.add_assistant_tool_calls(preamble, tc_payload)

            # Execute each tool sequentially.
            for tc in tc_payload:
                if on_tool_started:
                    await on_tool_started(tc)
                t0 = time.monotonic()
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                    result = {"success": False,
                              "error": f"invalid arguments JSON: "
                                       f"{tc['function']['arguments']!r}"}
                else:
                    result = await registry.dispatch(
                        tc["function"]["name"], args, ctx
                    )
                dt_ms = (time.monotonic() - t0) * 1000
                content = json.dumps(result, ensure_ascii=False)
                messages.append({"role": "tool",
                                 "tool_call_id": tc["id"],
                                 "content": content})
                session.add_tool_result(tc["id"], content)
                if on_tool_completed:
                    await on_tool_completed(tc, result, dt_ms)
            # loop continues

        # Iteration cap hit. Must-fix #2 (codex review): the partial
        # tool round (assistant_tool_calls + tool results appended in
        # the last failing iteration) has no terminal assistant(text),
        # so the trim turn-detector would pin it forever. Roll back to
        # the anchor (before this whole call started) — semantically
        # equivalent to "this turn never happened" for history; the
        # user input is still in session because add_user happened at
        # app_mode level *before* we were called.
        logger.warning("tool iteration cap reached (%d), rolling back",
                       max_iterations)
        dropped = session.rollback_to(rollback_anchor)
        # Also truncate the caller's local messages list to mirror.
        del messages[rollback_anchor + 1:]  # +1 for the system message offset; see note below
        if ctx.event_bus:
            ctx.event_bus.emit("on_tool_iteration_limit",
                               {"iterations": iterations_done,
                                "dropped": dropped,
                                "sid": session.sid})
        return ""
    except asyncio.CancelledError:
        # Must-fix #4 (codex review): truncate both session.history
        # AND the caller's local messages list, otherwise the next
        # turn sees mismatched state.
        dropped = session.rollback_to(rollback_anchor)
        # messages = [system, *session.history], so anchor in session
        # corresponds to (anchor + 1) in messages.
        del messages[rollback_anchor + 1:]
        logger.info("tool round cancelled, rolled back %d messages", dropped)
        raise
```

### 4. `app_mode.py` integration

Replace `run_default_dialogue_turn` lines 131-216 with a call to
`stream_with_tools`. Wire callbacks:

- `on_assistant_token = self.slv.send_text` (today's behaviour)
- `on_tool_started/completed → self.broadcast(...)` (dashboard events)
- Keep `first_token_timeout` / `idle_timeout` via `asyncio.wait_for`
  wrapping each `__anext__()` of `llm.stream_events`.
- Keep error-path: if partial tokens already flushed, `slv.abort()` +
  `audio.stop_playback` + raise — unchanged.

**TTS half-sentence behaviour**: when LLM emits text *before* a
`tool_call` ("好的，我查一下…"), we DO send those tokens to TTS but DO
NOT `flush_tts` until the loop exits with a non-tool finish_reason.
Result: the preamble + the post-tool answer arrive as one continuous
TTS stream from the user's perspective.

**Cancel semantics**: existing `_llm_turn_task.cancel()` already wraps
the whole turn. `runner.stream_with_tools` catches `CancelledError`,
rolls back partial tool messages, re-raises. The outer handler in
`app_mode.py` then runs `slv.abort()` as today.

### 5. Config (`config.py`)

```python
tools_enabled: bool = False                  # master switch
tools_default_allowlist: list[str] = []      # global default
tools_max_iterations: int = 5
mode_overrides[<mode>].tools_allowlist       # per-mode (overrides global)
mode_overrides[<mode>].tools_enabled         # per-mode (overrides global)
```

`app_base.py` constructs the `ToolRegistry` and registers builtins iff
`tools_enabled`. `app_mode.py` resolves the effective allowlist per turn.

### 6. Dashboard events (event bus)

New emit points:
- `on_tool_call_started   {id, name, arguments_json}`
- `on_tool_call_completed {id, name, ok, duration_ms}`
- `on_tool_call_error     {id, name, error}` (when result["success"] is False)
- `on_tool_iteration_limit {iterations}`

Existing token events unchanged. Dashboard subscribers add a "Tools"
panel showing the latest 5 calls.

### 6a. UX trade-off (documented)

Preamble tokens stream straight to TTS as today. If the LLM emits
"好的，我帮你查一下" then makes a tool call that takes 1-3 s, the user
hears the preamble immediately, then a silence window equal to tool
latency, then the rest. We accept this — alternative (hold preamble
until loop ends) adds latency to the common text-only path. Tools
with long latency should emit a `display` string back via the result
dict so app_mode can optionally speak it as a "filler" later.

### 7. Prefix cache interaction

Server-side concern (edge-llm `tools.py` + `api_server.py` —
see warehouse `PREFIX_CACHE_SPEC.md`). Client-side rules:

1. **Tools list MUST be stable per mode**. If allowlist changes mid-
   session the upstream `formatted_prefix` changes and cache misses
   (safe degradation, no crash). Document this in mode-config docs.
2. **First iteration**: `prefix_cache=True` is sent as today (when
   `session.cache_warmed`). Server-side cache was warmed against the
   original message shape `[system, ...history, user_N]`.
3. **Tool-loop iterations beyond the first**: runner injects
   `extra_body.prefix_cache=False` (see must-fix #1). The message
   shape now includes `assistant(tool_calls)` + `tool(result)` —
   even if the server supports those in prefix matching, the cached
   formatted-request key was built against the shorter shape and
   would mismatch. Forcing no-prefix-cache for iter >0 avoids the
   stale-hit risk and the wasted upstream round-trip.
4. **End-of-call cache_warmed**: `EdgeLLMBackend.stream_events` still
   sets `cache_warmed=True` at the end of its successful stream. With
   tools, this fires after each iteration — but since iter >0 already
   has `prefix_cache=False`, re-setting `cache_warmed=True` is a no-op
   on those calls. The *next user turn* benefits from cache against
   `[system, ...history(including tool round), user_{N+1}]` which is a
   strict extension of the iter-0 prefix — clean hit.
5. **A4 fallback** still wraps the whole `stream_events` call. If
   `_disable_prefix_cache` latches during iter-0 of a tool turn, all
   subsequent iters already skip prefix_cache (no-op) and the latch
   persists for the rest of the session as today.
6. **Trimming**: when `_trim_to_budget` fires (rare), `cache_warmed=False`
   is set as today. With turn-aware trim, a turn that started with tools
   stays atomic — no partial tool round in the history.

### 8. Cancel + error contract

| Event | Action |
|---|---|
| User barge-in mid-text | `_llm_turn_task.cancel()` → `runner` catches `CancelledError` → if no tool round in flight, no rollback; if tool round in flight, rollback assistant_tool_calls + tool_results added in this call. `slv.abort()` from app_mode. |
| LLM stream error after partial tokens | `runner` re-raises; `app_mode` aborts TTS; no commit to history (existing behaviour). |
| Tool exception | Caught inside `registry.dispatch`, wrapped in `{success: False, error: ...}`, fed back to LLM for self-recovery. |
| Tool timeout | Same as exception, error message `tool X timed out`. |
| Invalid arguments JSON | Same, error message `invalid arguments JSON: ...`. |
| Iteration cap | Stop, emit `on_tool_iteration_limit`, return empty string (app_mode handles as empty assistant turn). |

## Test plan

| Module | Tests |
|---|---|
| `llm/openai_compat.py` | Mock OpenAI stream chunks: text-only, tool-only, mixed text→tool→text, finish_reason="tool_calls" ordering, multi-tool same response. |
| `llm/edge_llm.py` | A4 fallback path still works when stream_events raises mid-stream with prefix_cache marker. |
| `session.py` | tool_calls/tool message roundtrip, rollback_to, trim with tool turns (no half-turns dropped), echo recovery skips tool-call-only messages. |
| `tools/registry.py` | Decorator schema generation for str/int/float/bool/list/Literal/Optional. Dispatch happy path, unknown tool, kwargs sanitize, sync vs async, timeout, exception. |
| `tools/runner.py` | Single-iteration text-only (no tools). Tool round-trip happy path. Multi-tool (one assistant emits 2 tool_calls). Cancel mid-tool. Iteration cap. Invalid JSON args. |
| `app_mode.py` (smoke) | Stub LLM + stub TTS + builtin time_now; end-to-end `run_default_dialogue_turn("what time is it")` produces correct messages list and final text. |

Existing tests for `app_mode.py` text-only path must remain green
(default `tools_enabled=False`).

## Commit plan

1. `feat(llm): add stream_events channel with backward-compat stream()`
2. `feat(session): support tool_calls and role:tool messages; turn-aware trim`
3. `feat(tools): @tool registry + builtin set + multi-turn runner`
4. `feat(agent): wire tools into app_mode + dashboard events + config`
5. `docs(agent): tool usage guide + builtin tool reference`

## Open questions for review

- **Q1**: Should `stream_with_tools` be a method on `LLMBackend` (so
  `EdgeLLMBackend` can override for any edge-llm-specific tool flow),
  or stay as a free function in `tools/runner.py`? Spec proposes the
  latter for separation of concerns.
- **Q2**: For a tool that takes >1s (e.g. HTTP call), do we want to emit
  a TTS "filler" ("稍等…")? Spec says no — keep it simple, voice agents
  rarely need this and it muddies cancellation. Punt to a future plugin.
- **Q3**: Does the existing `LLMTimeoutError` first_token/idle taxonomy
  still make sense when the "first token" might be a tool_call delta?
  Spec proposes: any LLMEvent with payload counts as first token (same
  as warehouse_system `_has_payload_delta`).
