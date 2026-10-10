from ovs_agent import Config
from ovs_agent.apps.edgearm.app import EdgeArmApp


def test_conversation_app_exposes_only_chat_mode():
    app = EdgeArmApp(Config())

    assert [mode["name"] for mode in app.modes.list_all()] == ["chat"]
