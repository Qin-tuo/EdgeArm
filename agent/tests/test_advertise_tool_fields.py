"""list_advertise_tools must carry preamble_text/completion_text/response_mode
so voxedge's server-loop engine can fire the spoken preamble + skip LLM round 2.
Regression guard for the server-loop "no 好的 + delayed reply" bug."""
from ovs_agent.tools import ToolRegistry


def _reg():
    r = ToolRegistry()

    @r.tool(name="wave", description="wave the arm",
            preamble_text="好的。", completion_text="挥完了。",
            response_mode="template")
    def wave():
        return {"ok": True}

    @r.tool(name="time_now", description="current time")  # defaults
    def time_now():
        return {"t": 0}

    return r


def test_advertise_carries_preamble_completion_mode():
    entries = {e["function"]["name"]: e for e in _reg().list_advertise_tools()}
    wave = entries["wave"]
    assert wave["preamble_text"] == "好的。"
    assert wave["completion_text"] == "挥完了。"
    assert wave["response_mode"] == "template"
    assert wave["function"]["name"] == "wave"
    assert "preamble_text" not in wave["function"]  # siblings, not inside fn


def test_advertise_omits_defaults():
    tn = {e["function"]["name"]: e for e in _reg().list_advertise_tools()}["time_now"]
    assert "preamble_text" not in tn       # empty → omitted
    assert "completion_text" not in tn
    assert "response_mode" not in tn        # "await" default → omitted


def test_list_openai_tools_still_strips_them():
    """The LLM-facing schema must NOT carry the extra fields."""
    for e in _reg().list_openai_tools():
        assert set(e.keys()) == {"type", "function"}
        assert set(e["function"].keys()) == {"name", "description", "parameters"}
