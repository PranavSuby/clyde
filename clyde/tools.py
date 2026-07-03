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

# Files the model has read this session; editing an existing file it hasn't
# read is rejected so the model can't guess at contents.
_READ_FILES: set[str] = set()


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


def _bash(args: dict) -> str:
    timeout = int(args.get("timeout") or 120)
    try:
        proc = subprocess.run(
            ["bash", "-c", args["command"]],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    out = proc.stdout
    if proc.stderr:
        out += ("\n" if out else "") + proc.stderr
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out.strip() or "(no output)"


def _read_file(args: dict) -> str:
    path = os.path.expanduser(args["path"])
    offset = max(1, int(args.get("offset") or 1))
    limit = int(args.get("limit") or 1000)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    _READ_FILES.add(os.path.abspath(path))
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
    path = os.path.expanduser(args["path"])
    abspath = os.path.abspath(path)
    if os.path.exists(abspath) and abspath not in _READ_FILES:
        return ("Error: this file already exists and you have not read it. "
                "Use read_file first, then edit_file for changes.")
    parent = os.path.dirname(abspath)
    os.makedirs(parent, exist_ok=True)
    content = args["content"]
    with open(path, "w") as f:
        f.write(content)
    _READ_FILES.add(abspath)
    return f"Wrote {len(content)} chars to {path}"


def _edit_file(args: dict) -> str:
    path = os.path.expanduser(args["path"])
    if os.path.abspath(path) not in _READ_FILES:
        return "Error: you must read this file with read_file before editing it."
    old, new = args["old_string"], args["new_string"]
    if old == new:
        return "Error: old_string and new_string are identical"
    with open(path, "r") as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        return "Error: old_string not found in file (must match exactly, including whitespace)"
    if count > 1 and not args.get("replace_all"):
        return f"Error: old_string appears {count} times; provide more context or set replace_all=true"
    if args.get("replace_all"):
        content = content.replace(old, new)
    else:
        content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    n = count if args.get("replace_all") else 1
    return f"Replaced {n} occurrence(s) in {path}"


def _list_dir(args: dict) -> str:
    path = os.path.expanduser(args.get("path") or ".")
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    out = []
    for e in entries:
        full = os.path.join(path, e)
        out.append(e + "/" if os.path.isdir(full) else e)
    return "\n".join(out)


def _glob(args: dict) -> str:
    root = os.path.expanduser(args.get("path") or ".")
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
    path = os.path.expanduser(args.get("path") or ".")
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
