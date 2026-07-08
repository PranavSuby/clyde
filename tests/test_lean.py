import pytest

from clyde import lean, tools


def _lean_ready() -> bool:
    conf = lean.lean_conf()
    return (conf.get("enabled", True)
            and lean.find_project(conf) is not None
            and lean.find_lake(conf["elan_bin"]) is not None)


lean_required = pytest.mark.skipif(
    not _lean_ready(), reason="Lean/Mathlib project not set up")


def test_lean_cheat_detection():
    assert lean.uses_cheat("n = n := by sorry") == "sorry"
    assert lean.uses_cheat("n = n := by admit") == "admit"
    # substrings of identifiers must not trip the check
    assert lean.uses_cheat("theorem sorrymaker : True := trivial") is None
    assert lean.uses_cheat("exact rfl") is None


def test_lean_check_guards():
    assert tools.execute("lean_check", {"code": ""}).startswith("Error")
    r = tools.execute(
        "lean_check",
        {"code": "def f : IO Unit := IO.Process.exit 0"})
    assert "refusing" in r
    assert tools.execute("lean_check", {"code": "x" * 20001}).startswith("Error")


def test_lean_check_registered():
    names = [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    assert "lean_check" in names
    assert "lean_check" not in tools.APPROVAL_REQUIRED
    assert "lean_check" not in tools.SUBAGENT_TOOL_NAMES


def test_lean_check_disabled(monkeypatch):
    monkeypatch.setattr(lean, "lean_conf", lambda: {"enabled": False})
    assert "disabled" in lean.run({"code": "theorem t : True := trivial"})


@lean_required
def test_lean_verifies_valid_proof():
    code = "theorem t (n : Nat) : n + 0 = n := rfl"
    assert lean.run({"code": code}).startswith("✅")


@lean_required
def test_lean_rejects_false_claim():
    code = "theorem t : 1 + 1 = 3 := rfl"
    assert "NOT VERIFIED" in lean.run({"code": code})


@lean_required
def test_lean_rejects_sorry():
    code = "theorem t (n : Nat) : n + 0 = n := by sorry"
    assert "NOT A REAL PROOF" in lean.run({"code": code})
