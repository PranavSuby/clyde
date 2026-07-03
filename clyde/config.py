"""Configuration for clyde: profiles for local and online model backends."""

import json
import os

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
    "auto_start_ollama": True,
    "max_tool_output_chars": 12000,
    "max_iterations": 40,
    # Compact automatically when the prompt exceeds this fraction of num_ctx
    # (0 disables). /context shows current usage.
    "auto_compact_threshold": 0.85,
    # Upper bound for "auto" num_ctx, even when more would fit.
    "auto_ctx_cap": 65536,
}


def load_config() -> dict:
    """Load config, creating the default file on first run."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with open(CONFIG_PATH) as f:
        user_cfg = json.load(f)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update(user_cfg)
    return cfg


def get_profile(cfg: dict, name: str) -> dict:
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        available = ", ".join(profiles) or "(none)"
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")
    return profiles[name]
