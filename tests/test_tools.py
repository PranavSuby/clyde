import os

import pytest

from clyde import tools


@pytest.fixture(autouse=True)
def fresh_state(tmp_path):
    tools._READ_FILES.clear()
    tools._SHELL["cwd"] = str(tmp_path)
    yield


def test_read_gate_blocks_unread(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    r = tools.execute("edit_file", {"path": str(p), "old_string": "hello",
                                    "new_string": "bye"})
    assert r.startswith("Error: you must read")


def test_read_gate_staleness(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    tools.execute("read_file", {"path": str(p)})
    p.write_text("changed\n")
    os.utime(p, (1, 1))
    r = tools.execute("edit_file", {"path": str(p), "old_string": "changed",
                                    "new_string": "x"})
    assert "modified since" in r


def test_edit_crlf_preserved(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"one\r\ntwo\r\n")
    tools.execute("read_file", {"path": str(p)})
    r = tools.execute("edit_file", {"path": str(p), "old_string": "two",
                                    "new_string": "2"})
    assert "Replaced" in r
    assert p.read_bytes() == b"one\r\n2\r\n"


def test_edit_requires_unique_match(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("dup\ndup\n")
    tools.execute("read_file", {"path": str(p)})
    r = tools.execute("edit_file", {"path": str(p), "old_string": "dup",
                                    "new_string": "x"})
    assert "2 times" in r
    r = tools.execute("edit_file", {"path": str(p), "old_string": "dup",
                                    "new_string": "x", "replace_all": True})
    assert "Replaced 2" in r


def test_bash_cwd_persists(tmp_path):
    (tmp_path / "sub").mkdir()
    tools.execute("bash", {"command": "cd sub"})
    out = tools.execute("bash", {"command": "pwd"})
    assert out.strip().endswith("/sub")


def test_bash_marker_stripped(tmp_path):
    out = tools.execute("bash", {"command": "echo visible"})
    assert out == "visible"
    assert tools._CWD_MARKER not in out


def test_bash_timeout_kills_group(tmp_path):
    import time
    t0 = time.time()
    r = tools.execute("bash", {"command": "sleep 30 & sleep 30", "timeout": 1})
    assert "timed out" in r
    assert time.time() - t0 < 10


def test_relative_paths_follow_shell_cwd(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_text("data\n")
    tools.execute("bash", {"command": "cd sub"})
    out = tools.execute("read_file", {"path": "x.txt"})
    assert "data" in out
