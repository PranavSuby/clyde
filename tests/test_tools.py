import os

import pytest

from clyde import tools


@pytest.fixture(autouse=True)
def fresh_state(tmp_path):
    tools._READ_FILES.clear()
    tools._PARTIAL_READS.clear()
    tools._SHELL["cwd"] = str(tmp_path)
    yield


def test_glob_with_path_returns_resolvable_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("x\n")
    out = tools.execute("glob", {"pattern": "*.py", "path": "sub"})
    # results must be readable as given, not bare basenames relative to nothing
    line = out.splitlines()[0]
    assert tools.execute("read_file", {"path": line}).strip().endswith("x")


def test_glob_absolute_pattern_flagged_outside_workspace(tmp_path):
    tools.set_workspace(str(tmp_path))
    try:
        assert tools.outside_workspace("glob", {"pattern": "/etc/*"}) == "/etc"
        assert tools.outside_workspace("glob", {"pattern": "*.py"}) is None
    finally:
        tools._WORKSPACE["root"] = None


def test_glob_rejects_dotdot_components(tmp_path):
    tools.set_workspace(str(tmp_path))
    try:
        # leading wildcard → empty static prefix → outside_workspace can't
        # flag it, so the tool itself must refuse to walk out of the root
        pattern = "**/" + "../" * 8 + "etc/*"
        assert tools.outside_workspace("glob", {"pattern": pattern}) is None
        out = tools.execute("glob", {"pattern": pattern})
        assert out.startswith("Error:")
        assert tools.execute("glob", {"pattern": "../*"}).startswith("Error:")
        # plain patterns still work
        (tmp_path / "a.py").write_text("x\n")
        assert "a.py" in tools.execute("glob", {"pattern": "**/*.py"})
    finally:
        tools._WORKSPACE["root"] = None


def test_config_dir_prefix_not_overmatched(tmp_path):
    tools.set_workspace(str(tmp_path))
    try:
        evil = os.path.expanduser("~/.config/clyde-evil/secrets")
        assert tools.outside_workspace("read_file", {"path": evil}) is not None
    finally:
        tools._WORKSPACE["root"] = None


def test_bash_timeout_keeps_partial_output(tmp_path):
    r = tools.execute("bash", {"command": "echo early; sleep 30", "timeout": 1})
    assert "timed out" in r
    assert "early" in r


def test_read_gate_blocks_unread(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello\n")
    r = tools.execute("edit_file", {"path": str(p), "old_string": "hello",
                                    "new_string": "bye"})
    assert r.startswith("Error: you must read")


def test_read_gate_partial_read_does_not_unlock(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("".join(f"line{i}\n" for i in range(10)))
    tools.execute("read_file", {"path": str(p), "limit": 1})
    r = tools.execute("edit_file", {"path": str(p), "old_string": "line9",
                                    "new_string": "x"})
    assert r.startswith("Error: you have only read lines 1-1")
    # chunked reads that eventually cover the whole file do unlock it
    tools.execute("read_file", {"path": str(p), "offset": 2, "limit": 4})
    tools.execute("read_file", {"path": str(p), "offset": 6, "limit": 100})
    r = tools.execute("edit_file", {"path": str(p), "old_string": "line9",
                                    "new_string": "x"})
    assert "Replaced" in r


def test_grep_fallback_timeout_on_catastrophic_regex(tmp_path, monkeypatch):
    (tmp_path / "bomb.txt").write_text("a" * 40 + "!\n")
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)  # force fallback
    monkeypatch.setattr(tools, "_GREP_TIMEOUT", 1)
    r = tools.execute("grep", {"pattern": "(a+)+$", "path": str(tmp_path)})
    assert "timed out" in r
    # normal fallback searches still work
    r = tools.execute("grep", {"pattern": "a+!", "path": str(tmp_path)})
    assert "bomb.txt" in r


def test_cleanup_background_kills_and_removes_logs(tmp_path):
    out = tools.execute("bash", {"command": "sleep 60",
                                 "run_in_background": True})
    assert "Started background process" in out
    bg = next(iter(tools._BG_PROCS.values()))
    log = bg["log"]
    killed = tools.cleanup_background()
    assert killed and "sleep 60" in killed[0]
    assert bg["proc"].poll() is not None
    assert not os.path.exists(log)
    assert not tools._BG_PROCS


def test_glob_prunes_skip_dirs(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "x.py").write_text("x\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x\n")
    (tmp_path / "top.py").write_text("x\n")
    out = tools.execute("glob", {"pattern": "**/*.py"})
    assert "src/a.py" in out and "top.py" in out
    assert "node_modules" not in out


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
