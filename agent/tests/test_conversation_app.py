from ovs_agent import Config
from ovs_agent.apps.voice_so101.app import VoiceSO101App


def test_conversation_app_exposes_only_chat_mode():
    app = VoiceSO101App(Config())

    assert [mode["name"] for mode in app.modes.list_all()] == ["chat"]
