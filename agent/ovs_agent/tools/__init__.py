"""Local in-process tool-calling for OpenVoiceStream Agent."""
from .registry import Tool, ToolRegistry, default_registry
from .runner import ToolCallCtx, stream_with_tools

# Importing builtin registers time_now / set_mode against
# default_registry as a side-effect — keep it last so the symbols
# above are stable.
from . import builtin  # noqa: F401  (side-effect import)

__all__ = [
    "Tool",
    "ToolRegistry",
    "default_registry",
    "ToolCallCtx",
    "stream_with_tools",
]
