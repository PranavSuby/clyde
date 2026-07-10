"""@file mentions must respect the same trust boundaries as tool results:
secrets are redacted before entering the conversation, and a mention of a
file outside the workspace must not pre-satisfy the edit read-gate."""

import io
import os

import pytest
from rich.console import Console

from clyde import tools
from clyde.cli import expand_mentions


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    tools._READ_FILES.clear()
    tools._PARTIAL_READS.clear()
    tools._SHELL["cwd"] = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    yield
    tools._WORKSPACE["root"] = None


def _console():
    return Console(file=io.StringIO())


def test_mention_redacts_secrets(tmp_path):
    p = tmp_path / "creds.env"
    p.write_text("API_KEY=supersecretvalue123\nAWS_KEY=AKIA" + "A" * 16 + "\n")
    out = expand_mentions(f"look at @{p}", _console())
    assert "supersecretvalue123" not in out
    assert "[redacted]" in out
    assert "creds.env" in out  # still attached, just scrubbed


def test_mention_outside_workspace_not_marked_read(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    tools.set_workspace(str(ws))
    outside = tmp_path / "outside.txt"
    outside.write_text("data\n")
    out = expand_mentions(f"see @{outside}", _console())
    assert "data" in out  # user asked for it explicitly, so it attaches
    assert os.path.realpath(str(outside)) not in tools._READ_FILES


def test_mention_inside_workspace_marked_read(tmp_path):
    tools.set_workspace(str(tmp_path))
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    expand_mentions("fix @f.txt", _console())
    assert os.path.realpath(str(p)) in tools._READ_FILES
