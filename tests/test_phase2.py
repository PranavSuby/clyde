import os
import sys
import time

import pytest

from clyde import tools
from clyde.agent import redact_secrets
from clyde.mcp import MCPServer, tool_schemas

STUB = os.path.join(os.path.dirname(__file__), "mcp_stub.py")


@pytest.fixture(autouse=True)
def fresh_state(tmp_path):
    tools._READ_FILES.clear()
    tools._SHELL["cwd"] = str(tmp_path)
    tools.set_workspace(str(tmp_path))
    yield
    tools._WORKSPACE["root"] = None


def test_background_bash_lifecycle():
    r = tools.execute("bash", {"command": "for i in 1 2 3; do echo tick$i; sleep 0.2; done",
                               "run_in_background": True})
    assert "background process #" in r
    bg_id = int(r.split("#")[1].split(" ")[0])
    time.sleep(1.2)
    out = tools.execute("bash_output", {"id": bg_id})
    assert "tick3" in out and "exited with code 0" in out


def test_background_bash_kill():
    r = tools.execute("bash", {"command": "sleep 60", "run_in_background": True})
    bg_id = int(r.split("#")[1].split(" ")[0])
    out = tools.execute("bash_output", {"id": bg_id})
    assert "still running" in out
    killed = tools.execute("bash_kill", {"id": bg_id})
    assert "Killed" in killed


def test_bash_streaming_callback(tmp_path):
    lines = []
    tools.execute("bash", {"command": "echo a; echo b"}, on_line=lines.append)
    assert lines == ["a", "b"]


def test_workspace_boundary(tmp_path):
    assert tools.outside_workspace("read_file", {"path": "inside.txt"}) is None
    outside = tools.outside_workspace("read_file", {"path": "/etc/passwd"})
    assert outside == "/etc/passwd"
    assert tools.outside_workspace("bash", {"command": "ls /"}) is None


def test_redaction():
    text = ("key=AKIAIOSFODNN7EXAMPLE and token ghp_" + "a" * 36
            + " and sk-" + "b" * 30)
    red = redact_secrets(text)
    assert "AKIA" not in red and "ghp_" not in red and "sk-" not in red
    assert redact_secrets("normal text 123") == "normal text 123"


def test_mcp_stub_roundtrip():
    server = MCPServer("stub", [sys.executable, STUB], timeout=10)
    try:
        assert [t["name"] for t in server.tools] == ["echo"]
        schemas = tool_schemas({"stub": server})
        assert schemas[0]["function"]["name"] == "mcp__stub__echo"
        assert server.call("echo", {"text": "hi"}) == "echo: hi"
    finally:
        server.close()
