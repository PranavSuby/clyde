import json
import os

from rich.console import Console

from clyde import config, session
from clyde.agent import Agent


class FakeProvider:
    model = "fake"
    num_ctx = 32768


def make_agent(cfg=None):
    return Agent(FakeProvider(), Console(file=open(os.devnull, "w")), cfg or {})


def test_trim_preserves_tool_pairing():
    a = make_agent()
    for _ in range(30):
        a.messages += [
            {"role": "user", "content": "x" * 15000},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "name": "bash", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "1", "name": "bash", "content": "y"},
            {"role": "assistant", "content": "done"},
        ]
    a._trim_history()
    assert a.messages[0]["role"] == "system"
    assert a.messages[1]["role"] != "tool"
    total = sum(len(str(m.get("content") or "")) for m in a.messages)
    assert total <= 32768 * 3


def test_ctx_percent_uses_estimate_when_underreported():
    a = make_agent()
    a.messages.append({"role": "user", "content": "x" * 70000})
    a.last_usage = {"prompt_tokens": 10}  # KV-cache-hit under-report
    assert a.ctx_percent() > 30


def test_rule_matches():
    assert config.rule_matches("bash(git *)", "bash", {"command": "git status"})
    assert not config.rule_matches("bash(git *)", "bash", {"command": "gitk"})
    assert not config.rule_matches("bash(git *)", "bash", {"command": "rm -rf /"})
    assert config.rule_matches("edit_file", "edit_file", {"path": "x"})
    assert not config.rule_matches("edit_file", "bash", {"command": "x"})


def test_allow_rule_skips_prompt():
    a = make_agent({"permissions": {"allow": ["bash(git *)"]}})
    assert a._approve("bash", {"command": "git log"}) is True


def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSIONS_DIR", str(tmp_path))
    a = make_agent()
    a.messages.append({"role": "user", "content": "remember me"})
    path = session.new_session_path()
    session.save(path, a, "local")
    found = session.list_sessions(cwd=a.cwd)
    assert found and found[0]["turns"] == 1
    data = session.load(found[0]["path"])
    assert data["messages"][-1]["content"] == "remember me"


def test_config_deep_merge_keeps_defaults():
    merged = config._deep_merge(
        json.loads(json.dumps(config.DEFAULT_CONFIG)),
        {"profiles": {"mine": {"type": "openai"}}},
    )
    assert "local" in merged["profiles"]
    assert "mine" in merged["profiles"]
