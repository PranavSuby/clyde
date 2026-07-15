"""Configuration for clyde: profiles for local and online model backends."""

import json
import os
import re

CONFIG_DIR = os.path.expanduser("~/.config/clyde")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
HISTORY_PATH = os.path.join(CONFIG_DIR, "history")

DEFAULT_CONFIG = {
    "default_profile": "local",
    "profiles": {
        # Local model running on this PC via Ollama's native API.
        # num_ctx: an integer, or "auto" = largest context that fits this PC
        # (probed from GPU VRAM, free RAM, and the model's architecture),
        # always capped at the model's own maximum context length.
        "local": {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3-coder:30b",
            "num_ctx": "auto",
        },
        # Ollama Cloud model, proxied through the local Ollama daemon.
        # Run `ollama signin` once, then any *-cloud model works.
        "cloud": {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3-coder:480b-cloud",
        },
        # Any OpenAI-compatible online API (OpenRouter, Groq, Together, ...).
        # Set the API key in the named environment variable.
        "openrouter": {
            "type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "qwen/qwen3-coder",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    },
    # Tool calls matching these rules run without an approval prompt.
    # Forms: "edit_file" (whole tool) or "bash(git *)" (command prefix).
    "permissions": {"allow": []},
    # MCP servers (stdio): {"name": {"command": ["npx", "-y", "some-mcp"]}}
    "mcp_servers": {},
    # Lean 4 proof checking (lean_check tool). The project dir holds a Lake
    # project with Mathlib as a built dependency; see README for setup.
    # Falls back to an existing ~/.local/share/clydesk/lean build.
    "lean": {
        "enabled": True,
        "project_dir": "~/.local/share/clyde/lean",
        "elan_bin": "~/.elan/bin",
        "timeout": 90,
    },
    # Injection / egress defenses (see INJECTION_EVAL.md):
    # spotlight_tool_results — wrap tool output in <untrusted_tool_output>
    #   delimiters plus a system-prompt rule that it is data, not instructions.
    # taint_reapproval — once untrusted tool output has entered the context,
    #   mutating tools (bash/edit/write/MCP) need explicit confirmation even
    #   under --yolo or an allow rule ('a' at the prompt disables per session).
    # exfil_guard — confirm outbound commands carrying values read from files
    #   or credential-shaped strings.
    # scrub_bash_env — drop credential-named env vars (API keys, tokens) from
    #   bash subprocess environments; bash_env_keep lists exceptions to keep.
    "spotlight_tool_results": True,
    "taint_reapproval": True,
    "exfil_guard": True,
    "scrub_bash_env": True,
    "bash_env_keep": [],
    "auto_start_ollama": True,
    "max_tool_output_chars": 12000,
    "max_iterations": 40,
    # Compact automatically when the prompt exceeds this fraction of num_ctx
    # (0 disables). /context shows current usage.
    "auto_compact_threshold": 0.85,
    # Upper bound for "auto" num_ctx, even when more would fit.
    "auto_ctx_cap": 65536,
}


class ConfigError(Exception):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in place). Public: also used
    by clydesk, which shares this config machinery."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def atomic_write_json(path: str, data: dict):
    """Write JSON via a temp file + rename, so a crash can't truncate it.
    The tmp name is pid-unique: two processes persisting at once (e.g. both
    answering `p` to an approval) must not interleave into one tmp file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_config() -> dict:
    """Load config, creating the default file on first run.

    User config is deep-merged over defaults, so adding one profile
    doesn't wipe out the stock ones.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH) as f:
            user_cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{CONFIG_PATH} is not valid JSON (line {e.lineno}, col {e.colno}): "
            f"{e.msg}. Fix it or delete the file to regenerate defaults."
        ) from e
    return deep_merge(json.loads(json.dumps(DEFAULT_CONFIG)), user_cfg)


def save_config(cfg: dict):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    atomic_write_json(CONFIG_PATH, cfg)


def add_allow_rule(rule: str):
    """Persist an allow rule into the on-disk user config only, so the
    merged defaults never get frozen into the user's file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_PATH) as f:
            user_cfg = json.load(f)
    except (OSError, ValueError):
        user_cfg = {}
    allow = user_cfg.setdefault("permissions", {}).setdefault("allow", [])
    if rule not in allow:
        allow.append(rule)
        atomic_write_json(CONFIG_PATH, user_cfg)


# Shell constructs that let a command escape a "bash(git *)" prefix rule:
# chaining, pipes, substitution, redirection (files, fds, /dev/tcp), newlines.
_BASH_UNSAFE = re.compile(r"[;&|`\n<>]|\$\(")


def rule_matches(rule: str, name: str, args: dict) -> bool:
    """Permission rule check: "edit_file" or "bash(git *)" (prefix)."""
    if "(" not in rule:
        return rule == name
    rule_tool, _, pattern = rule.partition("(")
    pattern = pattern.rstrip(")")
    if rule_tool != name:
        return False
    if name == "bash":
        target = args.get("command", "").strip()
        if pattern.endswith("*"):
            # a prefix rule must not approve `git status; rm -rf ~` etc.
            return not _BASH_UNSAFE.search(target) \
                and target.startswith(pattern[:-1])
        return target == pattern
    # path rules match the resolved real path, so `src/../../etc/passwd`
    # cannot satisfy `edit_file(src/*)`
    from . import tools
    target = os.path.realpath(tools._resolve(str(args.get("path", ""))))
    if pattern.endswith("*"):
        prefix = os.path.realpath(tools._resolve(pattern[:-1] or "."))
        base = pattern[:-1]
        # `src/*` means "inside src/": realpath of "src/" keeps no trailing
        # sep, so match either the dir itself-prefixed children or the
        # literal filename prefix (`src/foo*`)
        if base.endswith(os.sep) or base.endswith("/"):
            return target == prefix or target.startswith(prefix + os.sep)
        # filename prefix (`src/foo*`): glob-style, so it must not cross a
        # separator — `config*` approves config-local.py, not the contents
        # of a sibling config-backup/ directory
        return target.startswith(prefix) \
            and os.sep not in target[len(prefix):]
    return target == os.path.realpath(tools._resolve(pattern))


def get_profile(cfg: dict, name: str) -> dict:
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        available = ", ".join(profiles) or "(none)"
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")
    return profiles[name]
