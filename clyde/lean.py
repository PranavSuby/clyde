"""Verify a Lean 4 proof by compiling it against Mathlib (lean_check tool).

This is the deterministic ground truth for math: the model writes a formal
statement + proof in Lean, this tool runs the Lean elaborator on it, and
Lean — not the model — decides whether the proof is actually correct. A
proof "passes" only if Lean reports no errors AND the proof contains no
`sorry`/`admit` placeholders (which the elaborator accepts but which prove
nothing).

The proof is compiled inside a Lake project that has Mathlib as a built
dependency, so ``import Mathlib...`` resolves. Paths and the compile
timeout live under the ``lean`` key in the config; the default project dir
is ``~/.local/share/clyde/lean``, with a fallback to an existing
``~/.local/share/clydesk/lean`` build so both tools can share one Mathlib.
"""

import os

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "lean_check",
        "description": (
            "Verify a mathematical proof for real, using the Lean 4 theorem "
            "prover with Mathlib. Use this WHENEVER the user asks you to "
            "prove, disprove, or verify a mathematical claim, identity, "
            "inequality, or theorem. First reason out the proof, then "
            "formalize the FULL statement and proof as self-contained Lean 4 "
            "code and pass it here; Lean checks it rigorously and this "
            "returns whether it is a valid proof. If it fails, read the "
            "error, fix the Lean, and call again. Prefer narrow imports "
            "(e.g. 'import Mathlib.Tactic') over 'import Mathlib' — they "
            "load much faster. Do not use 'sorry' or 'admit': they make the "
            "proof trivially accepted but prove nothing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained Lean 4 source: any needed "
                        "import lines followed by the theorem/lemma and its "
                        "full proof. Must compile on its own."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}

# Tokens that turn a "proof checker" into an arbitrary code runner or that
# silently sidestep real verification. Rejected before Lean ever runs.
# `#eval`/`run_cmd`/`elab`/`macro` execute code at elaboration time, and
# `open IO` un-qualifies the blocked IO.* names — a proof needs none of them.
_FORBIDDEN = (
    "IO.Process", "IO.FS", "System.FilePath", "System.Platform",
    "@[extern", "@[implemented_by", "unsafe ", "initialize ",
    "#eval", "run_cmd", "run_elab", "run_meta",
    "open IO", "open System",
    "elab ", "elab_rules", "macro ", "macro_rules",
)
# Cheats that make Lean accept a non-proof.
_CHEATS = ("sorry", "admit")

# Existing Lake+Mathlib builds to fall back on when the configured
# project_dir isn't set up (a Mathlib build is multi-GB; share it).
_FALLBACK_PROJECT_DIRS = ("~/.local/share/clydesk/lean",)


def lean_conf() -> dict:
    """Read the `lean` config section, tolerating a broken config file."""
    defaults = {
        "enabled": True,
        "project_dir": "~/.local/share/clyde/lean",
        "elan_bin": "~/.elan/bin",
        "timeout": 90,
    }
    try:
        from . import config
        defaults.update(config.load_config().get("lean") or {})
    except Exception:
        pass
    return defaults


def _has_lakefile(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "lakefile.toml"))
            or os.path.isfile(os.path.join(path, "lakefile.lean")))


def find_project(conf: dict) -> str | None:
    """The configured Lake project dir, or a shared fallback build."""
    configured = os.path.expanduser(conf["project_dir"])
    if _has_lakefile(configured):
        return configured
    for cand in _FALLBACK_PROJECT_DIRS:
        cand = os.path.expanduser(cand)
        if _has_lakefile(cand):
            return cand
    return None


def find_lake(elan_bin: str) -> str | None:
    import shutil
    cand = os.path.join(os.path.expanduser(elan_bin), "lake")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return shutil.which("lake")


def uses_cheat(code: str) -> str | None:
    """Return the first `sorry`/`admit` used as a whole word, else None."""
    import re
    for word in _CHEATS:
        if re.search(rf"(?<![A-Za-z0-9_]){word}(?![A-Za-z0-9_])", code):
            return word
    return None


def run(args: dict, conf: dict | None = None) -> str:
    """Check the proof in ``args["code"]``.

    ``conf`` overrides the ``lean`` config section (keys: enabled,
    project_dir, elan_bin, timeout) — used by other frontends (clydesk)
    that keep their own config; defaults to this app's config.
    """
    import re
    import signal
    import subprocess
    import tempfile

    code = str(args.get("code", "")).strip()
    if not code:
        return "Error: no Lean code provided."
    if len(code) > 20000:
        return "Error: Lean code too long (limit 20000 chars)."

    for tok in _FORBIDDEN:
        if tok in code:
            return (f"Error: refusing to run Lean containing '{tok.strip()}' — "
                    "lean_check verifies proofs, it does not run I/O or "
                    "external code. Remove it and prove the statement instead.")

    if conf is None:
        conf = lean_conf()
    if not conf.get("enabled", True):
        return "Error: Lean proof checking is disabled in the config (lean.enabled)."

    project = find_project(conf)
    if project is None:
        return (f"Error: no Lean project at "
                f"{os.path.expanduser(conf['project_dir'])}. Lean+Mathlib is "
                "not set up yet; the proof cannot be checked. See the README "
                "('Lean proof checking') for setup.")

    lake = find_lake(conf["elan_bin"])
    if not lake:
        return ("Error: 'lake' (Lean build tool) not found. Install Lean via "
                "elan, or set lean.elan_bin in the config.")

    env = dict(os.environ)
    env["PATH"] = os.path.expanduser(conf["elan_bin"]) + os.pathsep + env.get("PATH", "")

    # Write the snippet into the project so `lake env lean` sees Mathlib.
    fd, path = tempfile.mkstemp(prefix="clyde_proof_", suffix=".lean", dir=project)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code + "\n")
        timeout = float(conf.get("timeout") or 90)
        try:
            proc = subprocess.Popen(
                [lake, "env", "lean", path],
                cwd=project, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group -> clean kill
            )
            try:
                out, _ = proc.communicate(timeout=timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
                return (f"Error: Lean timed out after {int(timeout)}s. The proof "
                        "may be too heavy, or the import too broad — try "
                        "narrower imports (e.g. 'import Mathlib.Tactic').")
        except FileNotFoundError:
            return "Error: could not launch Lean (lake/lean missing)."
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    out = (out or "").strip()
    # Lean prints the temp path in diagnostics; strip it to reduce noise.
    out = out.replace(path, "proof.lean")

    if rc != 0 or re.search(r"^[^\n]*error:", out, re.MULTILINE):
        detail = out[:1500] or "(no diagnostics)"
        return ("❌ NOT VERIFIED — Lean rejected the proof. Fix the Lean and "
                f"call lean_check again.\n\nLean errors:\n{detail}")

    # Compiles, but does it actually prove anything?
    cheat = uses_cheat(code)
    if cheat or "declaration uses 'sorry'" in out:
        return (f"❌ NOT A REAL PROOF — the proof uses '{cheat or 'sorry'}', a "
                "placeholder that proves nothing. Replace it with a genuine "
                "proof and call lean_check again.")

    warnings = "\n".join(line for line in out.splitlines()
                         if "warning:" in line)
    tail = f"\n\n(Lean warnings, non-fatal:\n{warnings[:400]})" if warnings else ""
    return ("✅ VERIFIED — Lean accepted the proof with no errors and no "
            "'sorry'. The statement is proven correct." + tail)
