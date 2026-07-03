"""Agent tools: schemas (OpenAI function format) and implementations."""

import fnmatch
import glob as globmod
import os
import shutil
import subprocess

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
                },
                "required": ["command"],
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
]

# Tools that mutate state and need user approval (unless --yolo).
APPROVAL_REQUIRED = {"bash", "write_file", "edit_file"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", "dist", "build"}

# Files the model has read this session (realpath -> mtime at read time).
# Editing an existing file it hasn't read — or one modified since the read —
# is rejected so the model can't apply blind edits.
_READ_FILES: dict[str, float] = {}


def _mark_read(path: str):
    rp = os.path.realpath(path)
    try:
        _READ_FILES[rp] = os.stat(rp).st_mtime
    except OSError:
        pass


def _check_read_gate(path: str) -> str | None:
    """Return an error string if the file may not be edited yet."""
    rp = os.path.realpath(path)
    if not os.path.exists(rp):
        return None  # new file: no gate
    if rp not in _READ_FILES:
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


def execute(name: str, args: dict, max_chars: int = 12000) -> str:
    """Run a tool and return its result as a string (errors included, never raises)."""
    try:
        fn = _IMPLS.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        result = fn(args)
        return _truncate(result, max_chars)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# cwd persists between bash calls so `cd` behaves the way models expect
_SHELL = {"cwd": None}
_CWD_MARKER = "__CLYDE_CWD__"


def _resolve(path: str) -> str:
    """Expand ~ and resolve relative paths against the persistent shell cwd."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_SHELL["cwd"] or os.getcwd(), path)
    return path


def _bash(args: dict) -> str:
    import os as _os
    import shlex
    import signal
    timeout = int(args.get("timeout") or 120)
    cwd = _SHELL["cwd"] or _os.getcwd()
    if not _os.path.isdir(cwd):
        cwd = _os.getcwd()
    wrapped = (
        f"cd {shlex.quote(cwd)} 2>/dev/null\n"
        f"{args['command']}\n"
        f"__rc=$?; printf '\\n{_CWD_MARKER}%s' \"$PWD\"; exit $__rc"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapped],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,  # own process group: timeouts kill children too
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            _os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return f"Error: command timed out after {timeout}s (process group killed)"
    marker_at = stdout.rfind(_CWD_MARKER)
    if marker_at >= 0:
        new_cwd = stdout[marker_at + len(_CWD_MARKER):].strip()
        if new_cwd:
            _SHELL["cwd"] = new_cwd
        stdout = stdout[:marker_at].rstrip("\n")
    out = stdout
    if stderr:
        out += ("\n" if out else "") + stderr
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out.strip() or "(no output)"


def _read_file(args: dict) -> str:
    path = _resolve(args["path"])
    offset = max(1, int(args.get("offset") or 1))
    limit = int(args.get("limit") or 1000)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    _mark_read(path)
    if not lines:
        return "(empty file)"
    chunk = lines[offset - 1 : offset - 1 + limit]
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
    root = _resolve(args.get("path") or ".")
    matches = globmod.glob(args["pattern"], root_dir=root, recursive=True)
    matches = [
        m for m in matches
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 1:
            return "(no matches)"
        if proc.returncode not in (0, 1):
            return f"Error: {proc.stderr.strip()}"
        return proc.stdout.strip()

    # Fallback: pure-python search
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
    "todo_write": _todo_write,
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "list_dir": _list_dir,
    "glob": _glob,
    "grep": _grep,
}
