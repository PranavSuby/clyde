"""Agent tools: schemas (OpenAI function format) and implementations."""

import fnmatch
import glob as globmod
import os
import re
import shutil
import subprocess

from . import lean

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Executes a bash command and returns its output. Use for git, running "
                "tests, installing packages, and anything not covered by other tools. "
                "Avoid using it for reading or searching files — use read_file, grep "
                "and glob instead of cat/find/grep commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Run detached and return immediately with an "
                                       "id; check on it later with bash_output. Use "
                                       "for dev servers and long builds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_output",
            "description": "Get the output (so far) of a background bash process "
                           "started with run_in_background, and whether it is "
                           "still running.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The background process id"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_kill",
            "description": "Kill a background bash process started with "
                           "run_in_background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The background process id"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Reads a file from the local filesystem and returns its contents "
                "with line numbers. You MUST read a file before editing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start from (default 1)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to read (default 1000)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Writes content to a file, creating it (and parent dirs) or "
                "overwriting it. ALWAYS prefer edit_file for existing files; never "
                "create files unless necessary for the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Performs an exact string replacement in a file. The edit FAILS if "
                "old_string does not match exactly (including whitespace/indentation) "
                "or is not unique in the file — include more surrounding context to "
                "make it unique, or set replace_all to change every occurrence. "
                "You must read the file before editing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "old_string": {"type": "string", "description": "Exact text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false)",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Launch a read-only research subagent with a FRESH context to "
                "explore the codebase and report back. Use it for broad searches "
                "('find where X is handled', 'summarize how Y works') so the "
                "file dumps don't fill up your own context — you only get its "
                "final report. It can read_file/list_dir/glob/grep but cannot "
                "edit or run commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The research task, self-contained (the "
                                       "subagent can't see this conversation)",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Create and manage a structured task list for the current session. "
                "Use it for multi-step tasks (3+ steps) to plan the work and show the "
                "user progress. Each call replaces the whole list. Mark exactly one "
                "item in_progress at a time, and mark items completed immediately "
                "after finishing them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The full updated todo list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The task description"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current dir)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Directory to search in (default: current dir)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex pattern. Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search (default: current dir)"},
                    "include": {
                        "type": "string",
                        "description": "Only search files matching this glob, e.g. '*.py'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    lean.TOOL_SCHEMA,
]

# Tools that mutate state and need user approval (unless --yolo).
APPROVAL_REQUIRED = {"bash", "write_file", "edit_file"}

# Tools whose results carry content clyde did not write itself (file
# contents, command output, subagent reports). Once any of these has entered
# the context, injected instructions may be steering the model — the taint
# re-approval gate keys off this set. MCP tools count too (checked by prefix).
UNTRUSTED_SOURCE_TOOLS = {"read_file", "bash", "bash_output", "grep", "glob",
                          "list_dir", "task"}

# Tools that change state outside the conversation; these are what an
# injection must reach to do damage, so they are the ones re-gated after
# untrusted content has been ingested.
MUTATING_TOOLS = {"bash", "edit_file", "write_file"}

# Read-only tools a subagent may use freely.
SUBAGENT_TOOL_NAMES = {"read_file", "list_dir", "glob", "grep"}

# The workspace boundary: reads outside it need approval (set by the Agent).
_WORKSPACE = {"root": None}


def set_workspace(root: str):
    _WORKSPACE["root"] = os.path.realpath(root)


def outside_workspace(name: str, args: dict) -> str | None:
    """If this read touches a path outside the workspace, return that path."""
    root = _WORKSPACE["root"]
    if root is None or name not in ("read_file", "list_dir", "glob", "grep"):
        return None
    raw = args.get("path") or "."
    if name == "glob":
        pattern = os.path.expanduser(str(args.get("pattern") or ""))
        static = pattern.split("*")[0].split("?")[0].split("[")[0]
        if os.path.isabs(pattern):
            # an absolute pattern ignores the path arg entirely
            raw = static or "/"
        elif static:
            # a relative pattern can still climb out via ".." components;
            # ones hidden behind a leading wildcard (empty static prefix)
            # are rejected outright in _glob
            raw = os.path.join(str(raw), static)
    target = os.path.realpath(_resolve(str(raw)))
    home_cfg = os.path.realpath(os.path.expanduser("~/.config/clyde"))
    if target == root or target.startswith(root + os.sep) \
            or target == home_cfg or target.startswith(home_cfg + os.sep):
        return None
    return target

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", "dist", "build"}

# Files the model has read *in full* this session (realpath -> mtime at read
# time). Editing an existing file it hasn't fully read — or one modified
# since the read — is rejected so the model can't apply blind edits.
_READ_FILES: dict[str, float] = {}

# Line ranges seen so far of files only partially read
# (realpath -> (mtime, merged sorted [(start, end)] 1-based inclusive)).
# Once the ranges cover the whole file it is promoted to _READ_FILES;
# a partial read alone must not unlock whole-file edits.
_PARTIAL_READS: dict[str, tuple[float, list[tuple[int, int]]]] = {}

# Provenance / taint: high-entropy tokens the model has READ from files this
# session. Redaction protects secrets *inbound* (in tool results) but only for
# values that match a known pattern; an opaque token matches nothing and would
# otherwise flow verbatim into an outbound tool argument. Tracking provenance
# catches exactly that: if a substantial token that came from a read shows up
# in an outbound bash/MCP argument, it's a likely exfiltration. See the exfil
# guard in agent._run_tool.
_TAINT_TOKENS: set[str] = set()
_TAINT_MAX = 4000  # cap the set so a huge read can't grow it without bound

# token candidates: >=12 chars from the "identifier/secret" alphabet.
# '=' is excluded on purpose so KEY=VALUE splits into KEY and VALUE (we want to
# track the value, not the whole assignment); base64 bodies still match without
# their trailing '=' padding.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-./+]{12,}")


def _looks_secretish(tok: str) -> bool:
    """A token worth tracking as provenance: high-entropy enough to be a
    credential, not an ordinary word or file path. Mixed case + a digit, or
    very long, and not a pure path/dotted-name."""
    if tok.count("/") >= 2 or tok.count(".") >= 2:
        return False  # path or dotted identifier — too FP-prone to taint
    has_digit = any(c.isdigit() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    has_lower = any(c.islower() for c in tok)
    return (has_digit and has_upper and has_lower) or len(tok) >= 24


def record_read_taint(text: str):
    """Remember the high-entropy tokens in content the model just read."""
    if len(_TAINT_TOKENS) >= _TAINT_MAX:
        return
    for tok in _TOKEN_RE.findall(text or ""):
        if _looks_secretish(tok):
            _TAINT_TOKENS.add(tok)
            if len(_TAINT_TOKENS) >= _TAINT_MAX:
                break


def tainted_hits(text: str) -> list[str]:
    """Tokens from read-content that appear verbatim in `text` (e.g. an
    outbound command). A non-empty result means read data is leaving via an
    argument."""
    if not text:
        return []
    return sorted(t for t in _TAINT_TOKENS if t in text)


def _mark_read(path: str):
    rp = os.path.realpath(path)
    try:
        _READ_FILES[rp] = os.stat(rp).st_mtime
    except OSError:
        pass
    _PARTIAL_READS.pop(rp, None)


def _note_read_range(path: str, start: int, end: int, total: int):
    """Record a partial read; promote to fully-read once the accumulated
    ranges cover every line of the (unchanged) file."""
    rp = os.path.realpath(path)
    try:
        mtime = os.stat(rp).st_mtime
    except OSError:
        return
    prev_mtime, ranges = _PARTIAL_READS.get(rp, (mtime, []))
    if prev_mtime != mtime:
        ranges = []  # file changed under us: earlier chunks are stale
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges + [(start, end)]):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    if merged and merged[0][0] <= 1 and merged[0][1] >= total:
        _mark_read(path)
    else:
        _PARTIAL_READS[rp] = (mtime, merged)


def _check_read_gate(path: str) -> str | None:
    """Return an error string if the file may not be edited yet."""
    rp = os.path.realpath(path)
    if not os.path.exists(rp):
        return None  # new file: no gate
    if rp not in _READ_FILES:
        if rp in _PARTIAL_READS:
            seen = ", ".join(f"{lo}-{hi}" for lo, hi in _PARTIAL_READS[rp][1])
            return (f"Error: you have only read lines {seen} of this file. "
                    "Read the rest of it before editing.")
        return "Error: you must read this file with read_file before editing it."
    try:
        if os.stat(rp).st_mtime != _READ_FILES[rp]:
            return ("Error: the file was modified since you last read it. "
                    "Read it again before editing.")
    except OSError:
        pass
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return text[:half] + f"\n... [{omitted} chars truncated] ...\n" + text[-half:]


def malformed_args_error(args: dict) -> str | None:
    """The providers mark tool calls whose argument JSON didn't parse;
    surface that to the model instead of running with empty args."""
    if not isinstance(args, dict) or "_malformed_json" not in args:
        return None
    return ("Error: the arguments for this tool call were not valid JSON "
            f"(received: {str(args['_malformed_json'])[:200]!r}). "
            "Re-issue the call with correctly encoded JSON arguments.")


def execute(name: str, args: dict, max_chars: int = 12000, on_line=None) -> str:
    """Run a tool and return its result as a string (errors included, never raises)."""
    err = malformed_args_error(args)
    if err:
        return err
    try:
        fn = _IMPLS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        result = fn(args, on_line=on_line) if name == "bash" else fn(args)
        return _truncate(result, max_chars)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# cwd persists between bash calls so `cd` behaves the way models expect
_SHELL = {"cwd": None}
_CWD_MARKER = "__CLYDE_CWD__"

# Tool subprocesses must not inherit credentials. The parent environment is
# where provider API keys live (see api_key_env in profiles), so any bash
# call — or an injected `printenv | curl ...` — could read them without
# touching a file, bypassing both redaction and the read-taint exfil guard.
# Variables whose NAME looks credential-bearing are dropped; everything else
# (PATH, HOME, VIRTUAL_ENV, ...) passes through so dev tooling keeps working.
_ENV_SENSITIVE = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|token|passw|credential"
    r"|private[_-]?key|access[_-]?key)")
_ENV_POLICY = {"scrub": True, "keep": frozenset()}


def set_env_policy(cfg: dict):
    _ENV_POLICY["scrub"] = bool(cfg.get("scrub_bash_env", True))
    _ENV_POLICY["keep"] = frozenset(cfg.get("bash_env_keep") or [])


def subprocess_env() -> dict | None:
    """Environment for tool subprocesses (None = inherit unchanged)."""
    if not _ENV_POLICY["scrub"]:
        return None
    return {k: v for k, v in os.environ.items()
            if k in _ENV_POLICY["keep"] or not _ENV_SENSITIVE.search(k)}


def _resolve(path: str) -> str:
    """Expand ~ and resolve relative paths against the persistent shell cwd."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_SHELL["cwd"] or os.getcwd(), path)
    return path


# Background processes: id -> {"proc", "log", "command"}
_BG_PROCS: dict[int, dict] = {}
_BG_NEXT_ID = [1]


def _kill_group(proc):
    import os as _os
    import signal
    try:
        _os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _bash(args: dict, on_line=None) -> str:
    import os as _os
    import shlex
    import tempfile
    import threading
    timeout = int(args.get("timeout") or 120)
    cwd = _SHELL["cwd"] or _os.getcwd()
    if not _os.path.isdir(cwd):
        cwd = _os.getcwd()

    if args.get("run_in_background"):
        log = tempfile.NamedTemporaryFile(
            mode="w", prefix="clyde-bg-", suffix=".log", delete=False)
        proc = subprocess.Popen(
            ["bash", "-c", f"cd {shlex.quote(cwd)} 2>/dev/null\n{args['command']}"],
            stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True, env=subprocess_env(),
        )
        log.close()  # the child holds its own handle
        bg_id = _BG_NEXT_ID[0]
        _BG_NEXT_ID[0] += 1
        _BG_PROCS[bg_id] = {"proc": proc, "log": log.name,
                            "command": args["command"][:120]}
        return (f"Started background process #{bg_id} (pid {proc.pid}). "
                f"Use bash_output with id={bg_id} to check on it.")

    wrapped = (
        f"cd {shlex.quote(cwd)} 2>/dev/null\n"
        f"{args['command']}\n"
        f"__rc=$?; printf '\\n{_CWD_MARKER}%s' \"$PWD\"; exit $__rc"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapped],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,  # own process group: timeouts kill children too
        env=subprocess_env(),
    )
    timed_out = threading.Event()

    def _on_timeout():
        timed_out.set()
        _kill_group(proc)

    timer = threading.Timer(timeout, _on_timeout)
    timer.start()
    lines: list[str] = []

    def _reader():
        # Draining the pipe in its own thread lets us stop waiting the moment
        # the bash process exits — even if a backgrounded grandchild
        # (`server &`) is still holding the stdout pipe open.
        pending_blanks = 0  # hold blanks: the cwd marker's printf adds one
        for line in proc.stdout:
            lines.append(line)
            if not on_line or _CWD_MARKER in line:
                continue
            if not line.strip():
                pending_blanks += 1
                continue
            for _ in range(pending_blanks):
                on_line("")
            pending_blanks = 0
            on_line(line.rstrip("\n"))

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    leaked_bg = False
    try:
        proc.wait()  # the bash process itself is done
        reader.join(timeout=0.3)  # let buffered output drain
        if reader.is_alive():
            # bash exited but the pipe is still open: a process it spawned in
            # the background outlived it. Don't block for the full timeout —
            # reap the group and move on.
            leaked_bg = True
            _kill_group(proc)
            reader.join(timeout=1.0)
    except BaseException:
        # Ctrl-C etc.: the child is in its own process group, so the
        # terminal's SIGINT never reaches it — kill it ourselves.
        _kill_group(proc)
        proc.wait()
        reader.join(timeout=1.0)
        raise
    finally:
        timer.cancel()
    if timed_out.is_set():
        partial = "".join(lines).strip()
        msg = f"Error: command timed out after {timeout}s (process group killed)"
        if partial:
            msg += "\nOutput before the timeout:\n" + partial[-2000:]
        return msg

    stdout = "".join(lines)
    marker_at = stdout.rfind(_CWD_MARKER)
    if marker_at >= 0:
        new_cwd = stdout[marker_at + len(_CWD_MARKER):].strip()
        if new_cwd:
            _SHELL["cwd"] = new_cwd
        stdout = stdout[:marker_at].rstrip("\n")
    out = stdout
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    if leaked_bg:
        out += ("\n[note: the command returned but left a process holding the "
                "output stream open (e.g. a trailing '&'); it was terminated. "
                "For servers or long tasks use run_in_background instead.]")
    return out.strip() or "(no output)"


def _bash_output(args: dict) -> str:
    bg = _BG_PROCS.get(int(args.get("id", 0)))
    if not bg:
        return f"Error: no background process #{args.get('id')}"
    running = bg["proc"].poll() is None
    try:
        with open(bg["log"], "r", errors="replace") as f:
            content = f.read()
    except OSError:
        content = "(no output captured)"
    tail = content[-6000:]
    status = "still running" if running \
        else f"exited with code {bg['proc'].returncode}"
    return f"[{bg['command']}] {status}\n{tail or '(no output yet)'}"


def cleanup_background() -> list[str]:
    """Kill still-running background processes and remove their log files.

    Called when clyde exits: after that there is no bash_output/bash_kill
    left to manage them with, so anything still running is just an orphan
    and every log is a leaked temp file."""
    killed = []
    for bg_id, bg in list(_BG_PROCS.items()):
        proc = bg["proc"]
        if proc.poll() is None:
            _kill_group(proc)
            killed.append(f"#{bg_id} ({bg['command']})")
        try:
            proc.wait(timeout=5)  # reap; the group is already SIGKILLed
        except (subprocess.TimeoutExpired, OSError):
            pass
        try:
            os.unlink(bg["log"])
        except OSError:
            pass
        del _BG_PROCS[bg_id]
    return killed


def _bash_kill(args: dict) -> str:
    bg = _BG_PROCS.get(int(args.get("id", 0)))
    if not bg:
        return f"Error: no background process #{args.get('id')}"
    if bg["proc"].poll() is None:
        _kill_group(bg["proc"])
        return f"Killed background process #{args['id']}"
    return f"Background process #{args['id']} had already exited"


def _read_file(args: dict) -> str:
    path = _resolve(args["path"])
    offset = max(1, int(args.get("offset") or 1))
    limit = int(args.get("limit") or 1000)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        _mark_read(path)
        return "(empty file)"
    chunk = lines[offset - 1 : offset - 1 + limit]
    if chunk:
        end = offset - 1 + len(chunk)
        if offset <= 1 and end >= len(lines):
            _mark_read(path)  # whole file seen in one call
        else:
            _note_read_range(path, offset, end, len(lines))
    record_read_taint("".join(chunk))  # provenance for the exfil guard
    numbered = [f"{i}: {line.rstrip(chr(10))}" for i, line in enumerate(chunk, start=offset)]
    result = "\n".join(numbered)
    remaining = len(lines) - (offset - 1 + len(chunk))
    if remaining > 0:
        result += f"\n... ({remaining} more lines, use offset={offset + len(chunk)} to continue)"
    return result


def _write_file(args: dict) -> str:
    path = _resolve(args["path"])
    gate = _check_read_gate(path)
    if gate:
        return gate + " (Use read_file first, then edit_file for changes.)"
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    content = args["content"]
    with open(path, "w", newline="") as f:
        f.write(content)
    _mark_read(path)
    return f"Wrote {len(content)} chars to {path}"


def _edit_file(args: dict) -> str:
    path = _resolve(args["path"])
    gate = _check_read_gate(path)
    if gate:
        return gate
    old, new = args["old_string"], args["new_string"]
    if old == new:
        return "Error: old_string and new_string are identical"
    try:
        # newline="" preserves CRLF/LF exactly as on disk
        with open(path, "r", newline="") as f:
            raw = f.read()
    except UnicodeDecodeError:
        return "Error: file is not valid UTF-8 text; edit it with a bash command instead"
    # match against LF-normalized text so old_string from read_file output
    # works on CRLF files; write back with the file's original line endings
    crlf = "\r\n" in raw
    content = raw.replace("\r\n", "\n") if crlf else raw
    old_n = old.replace("\r\n", "\n")
    new_n = new.replace("\r\n", "\n")
    count = content.count(old_n)
    if count == 0:
        return "Error: old_string not found in file (must match exactly, including whitespace)"
    if count > 1 and not args.get("replace_all"):
        return f"Error: old_string appears {count} times; provide more context or set replace_all=true"
    content = content.replace(old_n, new_n) if args.get("replace_all") \
        else content.replace(old_n, new_n, 1)
    if crlf:
        content = content.replace("\n", "\r\n")
    with open(path, "w", newline="") as f:
        f.write(content)
    _mark_read(path)
    n = count if args.get("replace_all") else 1
    return f"Replaced {n} occurrence(s) in {path}"


def _list_dir(args: dict) -> str:
    path = _resolve(args.get("path") or ".")
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    out = []
    for e in entries:
        full = os.path.join(path, e)
        out.append(e + "/" if os.path.isdir(full) else e)
    return "\n".join(out)


def _glob(args: dict) -> str:
    pattern = str(args["pattern"])
    # A ".." component walks out of root_dir even when a wildcard hides it
    # from the static-prefix check in outside_workspace ("**/../../.ssh/*"),
    # so it would list files outside the workspace with no prompt.
    if ".." in pattern.replace("\\", "/").split("/"):
        return ("Error: glob patterns may not contain '..' components. "
                "Pass the directory to search in the 'path' argument instead.")
    root = _resolve(args.get("path") or ".")
    rest = pattern[3:] if pattern.startswith("**/") else None
    if rest and "/" not in rest and "**" not in rest:
        # the common recursive case (`**/*.py`): walk with SKIP_DIRS pruned
        # instead of letting glob() traverse .git/node_modules/.venv fully
        # and discarding those matches afterwards
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            # glob's `*` never matches hidden entries, so skip them too
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
            relbase = os.path.relpath(dirpath, root)
            for name in filenames + dirnames:
                if name.startswith(".") and not rest.startswith("."):
                    continue
                if fnmatch.fnmatch(name, rest):
                    matches.append(name if relbase == "."
                                   else os.path.join(relbase, name))
    else:
        matches = globmod.glob(pattern, root_dir=root, recursive=True)
    matches = [
        # join with root so results resolve correctly from the shell cwd
        os.path.join(root, m) if root != (_SHELL["cwd"] or os.getcwd()) else m
        for m in matches
        if not any(part in SKIP_DIRS for part in m.split(os.sep))
    ]
    matches.sort()
    if not matches:
        return "(no matches)"
    shown = matches[:200]
    result = "\n".join(shown)
    if len(matches) > len(shown):
        result += f"\n... ({len(matches) - len(shown)} more matches)"
    return result


_GREP_TIMEOUT = 60


def _grep(args: dict) -> str:
    pattern = args["pattern"]
    path = _resolve(args.get("path") or ".")
    include = args.get("include")

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--max-count", "50",
               "--max-columns", "300", "-e", pattern]
        if include:
            cmd += ["--glob", include]
        cmd.append(path)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_GREP_TIMEOUT)
        if proc.returncode == 1:
            return "(no matches)"
        if proc.returncode not in (0, 1):
            return f"Error: {proc.stderr.strip()}"
        return proc.stdout.strip()

    # Fallback: pure-python search, in a child process we can kill — a
    # catastrophic-backtracking pattern stuck inside re.search() cannot be
    # interrupted from Python (not even by Ctrl-C) and would wedge the REPL.
    import multiprocessing
    import queue as queue_mod
    import re
    re.compile(pattern)  # surface a bad pattern as an immediate error
    # spawn, not fork: the REPL runs helper threads (bash readers/timers),
    # and forking a multi-threaded process can deadlock the child
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    child = ctx.Process(target=_grep_child, args=(q, pattern, path, include),
                        daemon=True)
    child.start()
    try:
        return q.get(timeout=_GREP_TIMEOUT)
    except queue_mod.Empty:
        return (f"Error: search timed out after {_GREP_TIMEOUT}s. "
                "Try a simpler pattern or a narrower path.")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        if child.is_alive():
            child.kill()
        child.join(5)


def _grep_child(q, pattern: str, path: str, include: str | None):
    try:
        q.put(_grep_python(pattern, path, include))
    except Exception as e:  # noqa: BLE001 — must reach the parent as a string
        q.put(f"Error: {type(e).__name__}: {e}")


def _grep_python(pattern: str, path: str, include: str | None) -> str:
    import re
    rx = re.compile(pattern)
    results = []
    if os.path.isfile(path):
        files = [path]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if include and not fnmatch.fnmatch(fname, include):
                    continue
                files.append(os.path.join(dirpath, fname))
    for fpath in files:
        try:
            with open(fpath, "r", errors="strict") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line):
                        results.append(f"{fpath}:{lineno}:{line.rstrip()[:300]}")
                        if len(results) >= 200:
                            return "\n".join(results) + "\n... (truncated)"
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(results) if results else "(no matches)"


def _todo_write(args: dict) -> str:
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    todos = args.get("todos") or []
    if not todos:
        return "Todo list cleared."
    lines = [f"{marks.get(t.get('status', 'pending'), '[ ]')} {t.get('content', '')}"
             for t in todos]
    return "\n".join(lines)


_IMPLS = {
    "bash": _bash,
    "bash_output": _bash_output,
    "bash_kill": _bash_kill,
    "todo_write": _todo_write,
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "list_dir": _list_dir,
    "glob": _glob,
    "grep": _grep,
    "lean_check": lean.run,
}
