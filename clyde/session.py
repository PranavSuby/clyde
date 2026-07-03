"""Session persistence: conversations survive crashes and restarts."""

import json
import os
import time
import uuid

SESSIONS_DIR = os.path.expanduser("~/.local/share/clyde/sessions")


def new_session_path() -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ".json"
    return os.path.join(SESSIONS_DIR, name)


def save(path: str, agent, profile: str):
    """Write the agent's state; called after every turn (cheap, atomic)."""
    data = {
        "version": 1,
        "cwd": agent.cwd,
        "profile": profile,
        "model": agent.provider.model,
        "updated": time.time(),
        "first_prompt": next(
            (m["content"][:100] for m in agent.messages if m["role"] == "user"),
            "",
        ),
        "messages": agent.messages,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def list_sessions(cwd: str | None = None, limit: int = 10) -> list[dict]:
    """Most-recent-first session metadata (optionally for one directory)."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if cwd and data.get("cwd") != cwd:
            continue
        n_turns = sum(1 for m in data.get("messages", []) if m["role"] == "user")
        out.append({
            "path": fpath,
            "updated": data.get("updated", 0),
            "cwd": data.get("cwd", ""),
            "profile": data.get("profile", ""),
            "model": data.get("model", ""),
            "first_prompt": data.get("first_prompt", ""),
            "turns": n_turns,
        })
    out.sort(key=lambda s: s["updated"], reverse=True)
    return out[:limit]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
