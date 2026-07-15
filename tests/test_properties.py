"""Property-based tests (hypothesis) for the guard functions an attacker
would probe: permission-rule matching, secret redaction, spotlight escaping,
env scrubbing, glob escapes, and output truncation.

Example-based tests pin known bypasses; these search for unknown ones."""

import os
import string

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from clyde import config as config_mod  # noqa: E402
from clyde import tools  # noqa: E402
from clyde.agent import (  # noqa: E402
    UNTRUSTED_CLOSE,
    redact_secrets,
    spotlight,
)

# --- rule_matches: no shell construct may slip past a bash prefix rule ------

_METACHARS = [";", "&", "|", "`", "\n", "<", ">", "$("]


@given(
    suffix=st.text(min_size=0, max_size=40),
    meta=st.sampled_from(_METACHARS),
    tail=st.text(alphabet=string.ascii_letters + " -/.", min_size=1,
                 max_size=40),
)
def test_bash_prefix_rule_never_matches_chained_commands(suffix, meta, tail):
    # a metachar with a real payload after it (a pure trailing newline is
    # stripped before matching and is harmless to execute)
    command = f"git {suffix}{meta}{tail}"
    assert not config_mod.rule_matches("bash(git *)", "bash",
                                       {"command": command})


@given(st.text(alphabet=string.printable, min_size=0, max_size=60))
def test_bash_prefix_rule_matches_only_real_prefixes(command):
    matched = config_mod.rule_matches("bash(git *)", "bash",
                                      {"command": command})
    if matched:
        stripped = command.strip()
        assert stripped.startswith("git ") or stripped == "git "
        assert not any(m in stripped for m in _METACHARS)


# --- path rules: traversal never escapes the allowed directory --------------

_SEGMENTS = st.lists(
    st.sampled_from(["..", ".", "sub", "a", "b.txt", "..."]),
    min_size=1, max_size=6,
)


@given(segments=_SEGMENTS)
def test_path_rule_traversal_cannot_escape(segments):
    # build the sandbox inline: hypothesis re-runs the body many times
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        allowed = os.path.join(root, "allowed")
        os.makedirs(allowed, exist_ok=True)
        tools._SHELL["cwd"] = root
        rel = os.path.join("allowed", *segments)
        rule = f"edit_file({allowed}/*)"
        if config_mod.rule_matches(rule, "edit_file", {"path": rel}):
            real = os.path.realpath(os.path.join(root, rel))
            assert real == allowed or real.startswith(allowed + os.sep)


# --- redact_secrets: planted credentials never survive ----------------------

@st.composite
def planted_secret(draw):
    kind = draw(st.sampled_from(["aws", "github", "openai", "slack", "envline"]))
    up = string.ascii_uppercase + string.digits
    alnum = string.ascii_letters + string.digits
    if kind == "aws":
        return "AKIA" + "".join(draw(st.sampled_from(up)) for _ in range(16))
    if kind == "github":
        return "ghp_" + "".join(draw(st.sampled_from(alnum)) for _ in range(36))
    if kind == "openai":
        return "sk-" + "".join(draw(st.sampled_from(alnum)) for _ in range(30))
    if kind == "slack":
        return "xoxb-" + "".join(draw(st.sampled_from(string.digits)) for _ in range(12))
    val = "".join(draw(st.sampled_from(alnum)) for _ in range(12))
    return f"API_KEY={val}"


@given(
    prefix=st.text(alphabet=string.ascii_letters + " \n", max_size=30),
    secret=planted_secret(),
    suffix=st.text(alphabet=string.ascii_letters + " \n", max_size=30),
)
def test_redaction_removes_planted_secrets(prefix, secret, suffix):
    # place the secret on its own line so prose can't merge into the token
    text = f"{prefix}\n{secret}\n{suffix}"
    redacted = redact_secrets(text)
    token = secret.split("=", 1)[-1]
    assert token not in redacted


@given(st.text(max_size=300))
def test_redaction_is_idempotent(text):
    once = redact_secrets(text)
    assert redact_secrets(once) == once


# --- spotlight: content can never fake an early end of the wrapper ----------

@given(st.text(max_size=300))
def test_spotlight_single_live_close_tag(text):
    wrapped = spotlight(text)
    assert wrapped.endswith(UNTRUSTED_CLOSE)
    # the only live closing tag is the final one we appended
    assert wrapped.count(UNTRUSTED_CLOSE) == 1


# --- env scrubbing: no credential-shaped name survives ----------------------

_NAME_CHARS = string.ascii_uppercase + string.digits + "_"


@given(
    stem=st.text(alphabet=_NAME_CHARS, min_size=1, max_size=12),
    marker=st.sampled_from(["API_KEY", "TOKEN", "SECRET", "PASSWORD",
                            "CREDENTIALS", "ACCESS_KEY", "PRIVATE_KEY"]),
)
def test_env_scrub_drops_all_credential_shaped_names(stem, marker):
    name = f"{stem}_{marker}"
    old = os.environ.get(name)
    os.environ[name] = "value"
    try:
        tools.set_env_policy({})
        assert name not in tools.subprocess_env()
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


# --- glob: no pattern reaches a .. escape ------------------------------------

@given(
    parts=st.lists(st.sampled_from(["..", "*", "src", "?", "[ab]", "."]),
                   min_size=1, max_size=5),
)
def test_glob_rejects_all_dotdot_patterns(parts):
    pattern = "/".join(parts)
    if ".." not in parts:
        return
    out = tools.execute("glob", {"pattern": pattern, "path": "."})
    assert out.startswith("Error")


# --- truncation: output is always bounded ------------------------------------

@given(st.text(max_size=5000), st.integers(min_value=50, max_value=500))
@settings(max_examples=50)
def test_truncate_is_bounded(text, cap):
    out = tools._truncate(text, cap)
    assert len(out) <= cap + 60  # cap plus the "[N chars truncated]" notice
