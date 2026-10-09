"""BaseApp._strip_wake_prefix: suppress / strip a leaked wake phrase.

The wake-word detector fires after the phrase audio is already in the server
ASR stream, so "Hey Jarvis" comes back transcribed as (or prefixing) the
utterance. A bare wake phrase must be dropped (no greeting reply); a prefixed
one must be stripped to the command.
"""
from __future__ import annotations

from ovs_agent.app_base import BaseApp


def _app():
    app = BaseApp.__new__(BaseApp)

    class _Cfg:
        wake_phrases = [
            "hey jarvis", "hi jarvis", "hello jarvis",
            "嘿 jarvis", "你好 jarvis",
        ]
    app.config = _Cfg()
    return app


def test_bare_wake_phrase_dropped():
    app = _app()
    for t in ["Hey Jarvis.", "hey jarvis", "Hey Jarvis!", "嘿 Jarvis。", "  Hi Jarvis  "]:
        assert app._strip_wake_prefix(t) is None, t


def test_wake_prefix_stripped_to_command():
    app = _app()
    assert app._strip_wake_prefix("Hey Jarvis 挥手") == "挥手"
    assert app._strip_wake_prefix("Hey Jarvis, 跳个舞") == "跳个舞"
    assert app._strip_wake_prefix("嘿 Jarvis，回到原位") == "回到原位"
    assert app._strip_wake_prefix("hi jarvis. turn on the light") == "turn on the light"


def test_non_wake_utterance_unchanged():
    app = _app()
    assert app._strip_wake_prefix("挥手") == "挥手"
    assert app._strip_wake_prefix("turn on the light") == "turn on the light"
    # "jarvis" alone is NOT a configured phrase → not stripped
    assert app._strip_wake_prefix("jarvis 挥手") == "jarvis 挥手"


def test_prefix_without_separator_not_stripped():
    # "jarvisturn" shouldn't be split — only a real separator after the phrase.
    app = _app()
    assert app._strip_wake_prefix("hey jarvistastic") == "hey jarvistastic"


def test_empty_and_no_phrases():
    app = _app()
    assert app._strip_wake_prefix("") == ""
    app.config.wake_phrases = []
    assert app._strip_wake_prefix("Hey Jarvis.") == "Hey Jarvis."
