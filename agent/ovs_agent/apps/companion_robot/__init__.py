"""CompanionRobotApp entry point package."""
from .app import CompanionRobotApp

# Side-effect import: registers the mock robot tools (move_head / play_emotion)
# onto default_registry so they are advertised when server-loop mode opens a
# session. Throwaway proof slice — real Reachy tools live in clawd-reachy-mini.
from . import demo_tools as _demo_tools  # noqa: F401,E402

App = CompanionRobotApp

__all__ = ["CompanionRobotApp", "App"]
