"""Tool registry + ``@tool`` decorator.

Builds OpenAI-style ``tools[]`` schemas from Python type hints and
dispatches function calls (sync or async) with timeout + error
isolation. Designed for local, in-process tools — every entry is
trusted code in the same Python process (no sandboxing, no MCP).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (OpenAI-style)
    fn: Callable[..., Any]
    timeout_s: float = 10.0
    # Optional per-tool "preamble" — short verbal acknowledgement (e.g.
    # "Okay." / "好的。") emitted via the agent's TTS channel the moment
    # the tool starts executing, BEFORE its result is appended to
    # history. Use for tools whose physical/real-world side-effect takes
    # noticeable time (robot motion, deploy, long-running query) and
    # whose hosting LLM structurally emits ``content=None + tool_calls``
    # (e.g. Qwen3-AWQ in no-think mode), making prompt-side "say OK
    # first" rules impossible to enforce.
    #
    # Default empty string = no preamble (backward compat — old @tool
    # call sites that don't pass this field behave exactly as before).
    preamble_text: str = ""
    # Fixed verbal acknowledgement spoken on successful tool completion
    # — used when ``response_mode == "template"`` (skip LLM round 2
    # entirely and synthesise this string instead) and as an optional
    # post-dispatch confirmation for "parallel". Empty string = no
    # completion text.
    completion_text: str = ""
    # How the runner should sequence the LLM round 2 / TTS reply after
    # dispatching this tool:
    #   * ``await``    — (default, backward compat) dispatch, wait for
    #                    the result, then run LLM round 2 normally.
    #   * ``parallel`` — dispatch returns fast (tool body is expected
    #                    to be non-blocking, e.g. starts an async job
    #                    and returns ``{"status":"started"}`` in ~200ms).
    #                    LLM round 2 runs immediately on that quick
    #                    result. Use when the tool's physical side-
    #                    effect overlaps the spoken acknowledgement.
    #   * ``template`` — dispatch normally, skip LLM round 2, emit the
    #                    fixed ``completion_text`` via the on_tool_
    #                    completion_text callback. Lowest latency post-
    #                    tool reply; no LLM creativity.
    #
    # Response modes (full contract):
    #   - "await" (default): runner waits for the tool body to complete,
    #     feeds the result to LLM round 2.
    #   - "parallel": tool body MUST return within ~200ms (hard ceiling
    #     ~500ms — runner warns above that). Runner kicks off LLM round 2
    #     in parallel with the real-world side-effect (motion / deploy)
    #     assuming the body returned ``{"started": True}``. For
    #     multi-stage operations split into ``dispatch_action`` (fast
    #     return) + ``wait_completion`` (background poller) — see
    #     ``arm_plugin.py`` for the canonical reference implementation.
    #   - "template": skip LLM round 2 entirely; runner uses
    #     ``completion_text`` directly via on_tool_completion_text.
    #     EMPTY ``completion_text`` falls back to "await" semantics
    #     (and logs a WARNING) so the user always hears a reply.
    response_mode: str = "await"


def _py_type_to_schema(t: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment.

    Supports: ``str`` / ``int`` / ``float`` / ``bool`` / ``list`` /
    ``dict`` / ``Literal[...]`` / ``Optional[T]`` / ``T | None``.
    Anything unknown falls back to ``{"type": "string"}`` — the LLM
    will still send something stringy that ``dispatch`` will reject if
    it's wrong.
    """
    origin = get_origin(t)
    args = get_args(t)

    # Literal[...] → enum
    if origin is Literal:
        sample = args[0]
        if isinstance(sample, bool):
            jtype = "boolean"
        elif isinstance(sample, int):
            jtype = "integer"
        elif isinstance(sample, float):
            jtype = "number"
        else:
            jtype = "string"
        return {"type": jtype, "enum": list(args)}

    # Optional[T] / Union[T, None] / ``T | None`` → unwrap and recurse.
    # Python 3.10+ ``X | Y`` uses ``types.UnionType``, not typing.Union;
    # check both.
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_schema(non_none[0])
        # Real union — fall back to string.
        return {"type": "string"}

    # Parameterized generics: list[X], dict[K, V]
    if origin in (list, tuple, set, frozenset):
        item_schema = (
            _py_type_to_schema(args[0]) if args else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}

    # Plain builtins
    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is list:
        return {"type": "array"}
    if t is dict:
        return {"type": "object"}

    return {"type": "string"}


class ToolRegistry:
    """Holds a set of registered tools, exports their OpenAI schemas, and
    dispatches calls. A module-level :data:`default_registry` is the
    one builtin tools attach to; tests may construct their own."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str = "",
        timeout_s: float = 10.0,
        preamble_text: str = "",
        completion_text: str = "",
        response_mode: str = "await",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register ``fn`` as a tool.

        Parameter schema is built from type hints, excluding ``ctx``
        (injected by the registry at dispatch time, not LLM-visible).
        Description defaults to the function's docstring."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(fn)
            try:
                hints = get_type_hints(fn)
            except Exception:  # pragma: no cover - defensive
                hints = {}
            props: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "ctx":
                    continue
                t = hints.get(pname, str)
                props[pname] = _py_type_to_schema(t)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            params: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                params["required"] = required
            tname = name or fn.__name__
            self._tools[tname] = Tool(
                name=tname,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn,
                timeout_s=timeout_s,
                preamble_text=preamble_text,
                completion_text=completion_text,
                response_mode=response_mode,
            )
            return fn

        return deco

    def list_openai_tools(
        self, allow: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return tools[] in OpenAI's chat-completions format. ``allow``
        filters by name; ``None`` exposes everything registered."""
        out: list[dict[str, Any]] = []
        for tname, t in self._tools.items():
            if allow is not None and tname not in allow:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    def list_advertise_tools(
        self, allow: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Like :meth:`list_openai_tools` but ALSO carries the per-tool
        ``preamble_text`` / ``completion_text`` / ``response_mode`` as
        top-level sibling keys on each entry.

        Used for the server-loop ``CLIENT_TOOL_ADVERTISE`` payload: voxedge's
        ConversationEngine reads these off the advertised entry
        (``entry.get("preamble_text")`` etc.) to fire the spoken preamble
        ("好的。") the moment the tool name streams in, and to skip LLM round 2
        for ``template`` tools (emit ``completion_text`` instead). Plain
        ``list_openai_tools`` strips them (the LLM schema must not carry them),
        which is exactly why server-loop lost the preamble + tight completion
        that the client-loop tool runner used locally.
        """
        out: list[dict[str, Any]] = []
        for tname, t in self._tools.items():
            if allow is not None and tname not in allow:
                continue
            entry: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            if t.preamble_text:
                entry["preamble_text"] = t.preamble_text
            if t.completion_text:
                entry["completion_text"] = t.completion_text
            if t.response_mode and t.response_mode != "await":
                entry["response_mode"] = t.response_mode
            out.append(entry)
        return out

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> list[str]:
        """Return registered tool names (registration order)."""
        return list(self._tools.keys())

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it was present.

        Idempotent: unregistering an unknown name returns False without
        raising. Use when a plugin needs to refresh its tool set at
        runtime (e.g. ``actions.yaml`` edited via an HTTP endpoint) —
        unregister the old names, then re-decorate with the new ones.
        """
        return self._tools.pop(name, None) is not None

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        ctx: Any,
    ) -> dict[str, Any]:
        """Invoke the named tool with ``arguments``.

        Always returns a JSON-serialisable dict. On error, returns
        ``{"success": False, "error": str}`` so the LLM can self-recover
        rather than crashing the voice loop."""
        t = self._tools.get(name)
        if t is None:
            return {"success": False, "error": f"unknown tool: {name}"}
        allowed = set(t.parameters.get("properties", {}).keys())
        clean: dict[str, Any] = {
            k: v for k, v in (arguments or {}).items() if k in allowed
        }
        try:
            if "ctx" in inspect.signature(t.fn).parameters:
                clean["ctx"] = ctx
            result = t.fn(**clean)
            if inspect.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=t.timeout_s)
            if isinstance(result, dict):
                return result
            return {"value": result}
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"tool {name} timed out after {t.timeout_s}s",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("tool %s raised %r", name, e)
            return {"success": False, "error": str(e)}


# Module-level default registry. Builtins register against this; the
# app wires it into ToolCallCtx. Tests can construct dedicated
# ToolRegistry() instances for isolation.
default_registry = ToolRegistry()
