import json
import os

from rich.console import Console

from clyde import config, session
from clyde.agent import Agent
from clyde.providers import ContextOverflowError, ProviderError


class FakeProvider:
    model = "fake"
    num_ctx = 32768


def make_agent(cfg=None):
    return Agent(FakeProvider(), Console(file=open(os.devnull, "w")), cfg or {})


class ScriptedProvider:
    """Yields one pre-scripted event list (or raises) per chat() call."""

    model = "fake"
    num_ctx = 32768

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # deep copy of the messages each call saw

    def chat(self, messages, tools):
        self.calls.append(json.loads(json.dumps(messages)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        yield from step


def scripted_agent(script, cfg=None, yolo=True):
    return Agent(ScriptedProvider(script), Console(file=open(os.devnull, "w")),
                 cfg or {}, yolo=yolo)


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
    assert a.approval.approve("bash", {"command": "git log"}) is True


def test_run_turn_plain_answer():
    a = scripted_agent([
        [("text", "hello there"), ("usage", {"prompt_tokens": 1})],
    ])
    a.run_turn("hi")
    assert a.messages[-1] == {"role": "assistant", "content": "hello there"}
    assert not a.had_error


def test_run_turn_tool_roundtrip():
    tc = {"id": "t1", "name": "bash", "arguments": {"command": "echo roundtrip"}}
    a = scripted_agent([
        [("tool_calls", [tc]), ("usage", {})],
        [("text", "done"), ("usage", {})],
    ])
    a.run_turn("run it")
    tool_msg = next(m for m in a.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "t1"
    assert "roundtrip" in tool_msg["content"]
    assert a.messages[-1]["content"] == "done"
    # the second model call must have seen the tool result
    assert any(m["role"] == "tool" for m in a.provider.calls[1])


def test_run_turn_denied_tool_records_error():
    tc = {"id": "t1", "name": "bash", "arguments": {"command": "echo hi"}}
    a = scripted_agent([
        [("tool_calls", [tc]), ("usage", {})],
        [("text", "ok"), ("usage", {})],
    ], yolo=False)
    a.approval.approve = lambda name, args, preview=None: False
    a.run_turn("run")
    tool_msg = next(m for m in a.messages if m["role"] == "tool")
    assert tool_msg["content"].startswith("Error: user denied")
    assert a.messages[-1]["content"] == "ok"


def test_run_turn_retries_transient_error(monkeypatch):
    from clyde import agent as agent_mod
    monkeypatch.setattr(agent_mod.time, "sleep", lambda s: None)
    a = scripted_agent([
        ProviderError("HTTP 503: overloaded", retryable=True),
        [("text", "recovered"), ("usage", {})],
    ])
    a.run_turn("hi")
    assert not a.had_error
    assert a.messages[-1]["content"] == "recovered"


def test_run_turn_nonretryable_error_stops():
    a = scripted_agent([ProviderError("HTTP 401: bad key")])
    a.run_turn("hi")
    assert a.had_error


def test_run_turn_overflow_compacts_and_retries():
    a = scripted_agent([
        ContextOverflowError("maximum context length exceeded"),
        [("text", "a compact summary")],          # the compact() call
        [("text", "final answer"), ("usage", {})],
    ])
    a.messages += [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    a.run_turn("the question")
    assert len(a.provider.calls) == 3
    assert any("Summary of the conversation so far" in str(m.get("content"))
               for m in a.messages)
    # the original question survives the compaction verbatim
    assert any(m.get("content") == "the question" for m in a.messages)
    assert a.messages[-1]["content"] == "final answer"
    assert not a.had_error


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
    merged = config.deep_merge(
        json.loads(json.dumps(config.DEFAULT_CONFIG)),
        {"profiles": {"mine": {"type": "openai"}}},
    )
    assert "local" in merged["profiles"]
    assert "mine" in merged["profiles"]
