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
    # pid-unique tmp name: two clyde processes saving the same session must
    # not clobber each other's half-written file
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _valid(data) -> bool:
    """True if this parsed JSON looks like a usable session file."""
    return (isinstance(data, dict)
            and isinstance(data.get("messages"), list)
            and all(isinstance(m, dict) and "role" in m
                    for m in data["messages"]))


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
        if not _valid(data):
            continue  # foreign or corrupt file in the sessions dir
        if cwd and data.get("cwd") != cwd:
            continue
        n_turns = sum(1 for m in data["messages"] if m.get("role") == "user")
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
    """Load a session file. Raises OSError or ValueError if unusable."""
    with open(path) as f:
        data = json.load(f)
    if not _valid(data):
        raise ValueError(f"{path} is not a clyde session file")
    return data


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def claim(path: str) -> str:
    """Claim a session file for this process so two concurrent `clyde -c`
    runs in the same directory don't overwrite each other's history.

    Returns `path` if we got it, or a fresh session path (forking the
    conversation) if another *live* process already holds it. A stale lock
    left by a crashed process is reclaimed. Best-effort: if locking isn't
    possible at all, just use `path`."""
    lock = path + ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        except FileExistsError:
            try:
                with open(lock) as f:
                    holder = int(f.read().strip() or "0")
            except (OSError, ValueError):
                holder = 0
            if holder != os.getpid() and _pid_alive(holder):
                return new_session_path()  # another clyde owns it; fork
            try:
                os.unlink(lock)  # stale lock from a dead process; retry
            except OSError:
                return new_session_path()
        except OSError:
            return path  # e.g. read-only fs — proceed without a lock
    return new_session_path()


def release(path: str):
    """Drop a lock previously taken by claim(), if this process holds it."""
    lock = path + ".lock"
    try:
        with open(lock) as f:
            holder = int(f.read().strip() or "0")
    except (OSError, ValueError):
        return
    if holder == os.getpid():
        try:
            os.unlink(lock)
        except OSError:
            pass
