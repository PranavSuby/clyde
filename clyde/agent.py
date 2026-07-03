"""The agent loop: stream model output, execute tool calls, repeat."""

import datetime
import json
import os
import platform
import subprocess
import time

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from . import config as config_mod
from . import tools
from .providers import BaseProvider, ProviderError

CONTEXT_FILES = ("CLYDE.md", "AGENTS.md", "CLAUDE.md")


class ThinkFilter:
    """Split a token stream into ('text', s) / ('thinking', s) on <think> tags.

    Handles tags split across chunk boundaries by holding back any suffix
    that could be the start of a tag.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, s: str) -> list[tuple[str, str]]:
        self.buf += s
        out = []
        while True:
            tag = self.CLOSE if self.in_think else self.OPEN
            kind = "thinking" if self.in_think else "text"
            idx = self.buf.find(tag)
            if idx >= 0:
                if idx > 0:
                    out.append((kind, self.buf[:idx]))
                self.buf = self.buf[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            # hold back the longest suffix that could start the tag
            keep = 0
            for i in range(1, len(tag)):
                if self.buf.endswith(tag[:i]):
                    keep = i
            emit = self.buf[: len(self.buf) - keep] if keep else self.buf
            self.buf = self.buf[len(self.buf) - keep:] if keep else ""
            if emit:
                out.append((kind, emit))
            return out

    def flush(self) -> list[tuple[str, str]]:
        kind = "thinking" if self.in_think else "text"
        out = [(kind, self.buf)] if self.buf else []
        self.buf = ""
        return out


SYSTEM_PROMPT = """\
You are Clyde, an interactive CLI tool that helps users with software \
engineering tasks. Use the instructions below and the tools available to you \
to assist the user.

# Tone and style
You should be concise, direct, and to the point. Your output is displayed in \
a terminal.
- You MUST answer concisely with fewer than 4 lines of text (not including \
tool use or code generation) unless the user asks for detail. One-word \
answers are best when appropriate.
- Avoid preamble ("Here is what I will do...") and postamble ("To \
summarize..."). Answer the question directly.
- When you run a non-trivial bash command, briefly explain what it does and why.
- Only use emojis if the user explicitly asks.

# Proactiveness
You are allowed to be proactive, but only when the user asks you to do \
something. Do the right thing when asked, including follow-up actions, but \
do not surprise the user with actions they did not ask for. If the user asks \
how to approach something, answer their question first instead of \
immediately editing files. After finishing a change, stop — do not add a \
summary of what you did unless asked.

# Following conventions
When making changes to files, first understand the file's code conventions. \
Mimic code style, use existing libraries and utilities, and follow existing \
patterns.
- NEVER assume a library is available, even a well-known one. Before using \
one, check that this codebase already uses it (neighboring files, \
requirements.txt, pyproject.toml, package.json, ...).
- When you create a new component or function, look at existing ones first \
and follow their conventions.
- Always follow security best practices. Never introduce code that exposes \
or logs secrets or keys. Never commit secrets.

# Code style
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS to code unless asked.

# Doing tasks
The user will primarily request software engineering tasks: fixing bugs, \
adding functionality, refactoring, explaining code. Recommended steps:
1. If the task requires 3 or more steps, use the todo_write tool to plan it, \
and keep the todo list updated as you work (mark items in_progress and \
completed as you go).
2. Use grep, glob, list_dir and read_file to understand the codebase and the \
user's query. ALWAYS read a file before editing it.
3. Implement the change. Prefer edit_file for existing files; write_file \
only for new files or full rewrites.
4. Verify the solution with tests if possible. NEVER assume a specific test \
framework or test script — check the README or the codebase to find out.
5. VERY IMPORTANT: when a task is complete, run lint/typecheck commands \
(e.g. ruff, npm run lint) if you know them for this project; if you cannot \
find them, ask the user and suggest writing them to CLYDE.md.
Never invent file paths, APIs, or command output. If unsure, check with a tool.

# Tool usage policy
- Call tools only when needed; answer directly when you already have the \
information.
- File paths passed to tools should be absolute, or relative to the working \
directory shown below.
"""


class StreamStyler:
    """Style streamed text line-by-line without buffering whole responses:
    code-fence contents cyan, fence markers dim, headers bold. Holds back at
    most the first few chars of each line to classify it, so streaming stays
    live."""

    def __init__(self):
        self.line_start = True
        self.hold = ""
        self.in_fence = False
        self.line_style = None

    def _classify(self):
        stripped = self.hold.lstrip()
        if stripped.startswith("```"):
            self.in_fence = not self.in_fence
            self.line_style = "bright_black"
        elif self.in_fence:
            self.line_style = "cyan"
        elif stripped.startswith("#"):
            self.line_style = "bold"
        else:
            self.line_style = None

    def feed(self, s: str) -> list[tuple[str, str | None]]:
        out = []
        while s:
            if self.line_start:
                nl = s.find("\n")
                take = s if nl == -1 else s[:nl + 1]
                self.hold += take
                s = "" if nl == -1 else s[nl + 1:]
                complete = self.hold.endswith("\n")
                if complete or len(self.hold.lstrip()) >= 3 or len(self.hold) >= 8:
                    self._classify()
                    out.append((self.hold, self.line_style))
                    self.hold = ""
                    self.line_start = complete
            else:
                nl = s.find("\n")
                if nl == -1:
                    out.append((s, self.line_style))
                    s = ""
                else:
                    out.append((s[:nl + 1], self.line_style))
                    s = s[nl + 1:]
                    self.line_start = True
        return out

    def flush(self) -> list[tuple[str, str | None]]:
        if not self.hold:
            return []
        self._classify()
        held, self.hold = self.hold, ""
        return [(held, self.line_style)]


def build_system_prompt(cwd: str) -> str:
    git_info = "no"
    try:
        branch = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            git_info = f"yes, branch {branch.stdout.strip()}"
    except Exception:
        pass
    parts = [
        SYSTEM_PROMPT,
        "Here is useful information about the environment you are running in:",
        "<env>",
        f"Working directory: {cwd}",
        f"Is directory a git repo: {git_info}",
        f"Platform: {platform.system().lower()} {platform.release()}",
        f"Python version: {platform.python_version()}",
        f"Today's date: {datetime.date.today().isoformat()}",
        "</env>",
    ]
    for fname in CONTEXT_FILES:
        fpath = os.path.join(cwd, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    content = f.read()[:8000]
                parts += ["", f"Project notes from {fname}:", content]
            except OSError:
                pass
            break
    return "\n".join(parts)


def _args_preview(name: str, args: dict) -> str:
    if name == "bash":
        return args.get("command", "")
    if name in ("write_file", "read_file", "edit_file"):
        return args.get("path", "")
    return json.dumps(args, ensure_ascii=False)[:120]


class Agent:
    def __init__(self, provider: BaseProvider, console: Console, cfg: dict,
                 yolo: bool = False):
        self.provider = provider
        self.console = console
        self.cfg = cfg
        self.yolo = yolo
        self.cwd = os.getcwd()
        self.last_usage: dict = {}
        self.session_allow: set[str] = set()
        self.had_error = False
        self.messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self.cwd)}
        ]

    def clear(self):
        self.messages = [self.messages[0]]
        self.last_usage = {}
        tools._READ_FILES.clear()

    def _context_size(self) -> int | None:
        """The window we're budgeting against, for any provider type."""
        return getattr(self.provider, "num_ctx", None) \
            or getattr(self.provider, "context_window", None)

    def _estimated_prompt_tokens(self) -> int:
        """Client-side estimate (~3.5 chars/token). Backstop for providers
        that under-report (Ollama omits KV-cache-hit tokens) or not at all."""
        chars = sum(
            len(str(m.get("content") or ""))
            + (len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0)
            for m in self.messages
        )
        return int(chars / 3.5)

    def ctx_percent(self) -> int | None:
        window = self._context_size()
        if not window:
            return None
        pt = max(self.last_usage.get("prompt_tokens") or 0,
                 self._estimated_prompt_tokens())
        return round(100 * pt / window) if pt else None

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _print_stream(self, kind: str, text: str, state: dict):
        """Print streamed tokens, tracking whether we're mid-line."""
        if state.get("last_kind") not in (None, kind):
            self.console.print()  # separate thinking from answer
        state["last_kind"] = kind
        if kind == "thinking":
            self.console.print(text, style="dim italic", end="",
                               markup=False, highlight=False)
        else:
            styler = state.setdefault("styler", StreamStyler())
            for segment, style in styler.feed(text):
                self.console.print(segment, style=style, end="",
                                   markup=False, highlight=False)
        state["midline"] = not text.endswith("\n")

    def _end_stream(self, state: dict):
        styler = state.get("styler")
        if styler:
            for segment, style in styler.flush():
                self.console.print(segment, style=style, end="",
                                   markup=False, highlight=False)
        if state.get("midline"):
            self.console.print()
            state["midline"] = False

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n… (+{len(text) - limit} more chars)"

    def _diff_preview(self, args: dict, is_edit: bool) -> str | None:
        """Unified diff of what the change would do, as a Rich markup string."""
        import difflib
        path = tools._resolve(str(args.get("path", "")))
        try:
            with open(path, "r", newline="") as f:
                old_content = f.read().replace("\r\n", "\n")
        except (OSError, UnicodeDecodeError):
            return None
        if is_edit:
            old_s = args.get("old_string", "").replace("\r\n", "\n")
            new_s = args.get("new_string", "").replace("\r\n", "\n")
            if args.get("replace_all"):
                new_content = old_content.replace(old_s, new_s)
            else:
                new_content = old_content.replace(old_s, new_s, 1)
            if new_content == old_content:
                return None  # old_string not found; the tool will explain
        else:
            new_content = args.get("content", "")
        lines = list(difflib.unified_diff(
            old_content.splitlines(), new_content.splitlines(),
            lineterm="", n=3,
        ))[2:]  # drop ---/+++ header
        if not lines:
            return None
        if len(lines) > 80:
            lines = lines[:80] + [f"… ({len(lines) - 80} more diff lines)"]
        styled = []
        for line in lines:
            e = escape(line)
            if line.startswith("+"):
                styled.append(f"[green]{e}[/green]")
            elif line.startswith("-"):
                styled.append(f"[red]{e}[/red]")
            elif line.startswith("@@"):
                styled.append(f"[cyan]{e}[/cyan]")
            else:
                styled.append(e)
        return "\n".join(styled)

    def _rule_for(self, name: str, args: dict) -> str:
        """The allow-rule suggestion for the 'p' (persist) approval answer."""
        if name == "bash":
            first = (args.get("command", "").strip().split() or ["?"])[0]
            return f"bash({first} *)"
        return name

    def _approve(self, name: str, args: dict) -> bool:
        if self.yolo or name in self.session_allow \
                or name not in tools.APPROVAL_REQUIRED:
            return True
        for rule in self.cfg.get("permissions", {}).get("allow", []):
            if config_mod.rule_matches(rule, name, args):
                return True
        if name in ("edit_file", "write_file"):
            diff = self._diff_preview(args, is_edit=(name == "edit_file"))
            if diff is not None:
                self.console.print(Panel(
                    diff, title=f"{name.split('_')[0]} {args.get('path', '')}",
                    border_style="yellow"))
            elif name == "edit_file":
                body = (
                    f"[red]- {escape(self._clip(args.get('old_string', ''), 1500))}[/red]\n"
                    f"[green]+ {escape(self._clip(args.get('new_string', ''), 1500))}[/green]"
                )
                self.console.print(Panel(body, title=f"edit {args.get('path', '')}",
                                         border_style="yellow"))
            else:
                preview = escape(self._clip(args.get("content", ""), 2000))
                self.console.print(Panel(
                    preview, title=f"write new file {args.get('path', '')}",
                    border_style="yellow"))
        persist_rule = self._rule_for(name, args)
        try:
            answer = self.console.input(
                f"[yellow]Allow {name}? \\[y/n/a=always allow {name} this session"
                f"/p=permanently allow {escape(persist_rule)}][/yellow] "
            ).strip().lower()
        except EOFError:
            return False
        if answer == "a":
            self.session_allow.add(name)
            self.console.print(f"[dim]{name} auto-approved for this session "
                               f"(/yolo for everything)[/dim]")
            return True
        if answer == "p":
            allow = self.cfg.setdefault("permissions", {}).setdefault("allow", [])
            if persist_rule not in allow:
                allow.append(persist_rule)
                config_mod.save_config(self.cfg)
            self.console.print(f"[dim]saved allow rule: {escape(persist_rule)} "
                               f"({config_mod.CONFIG_PATH})[/dim]")
            return True
        return answer in ("y", "yes")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_turn(self, user_input: str):
        self.had_error = False
        self._run_turn(user_input)
        threshold = self.cfg.get("auto_compact_threshold", 0.85)
        window = self._context_size()
        pt = max(self.last_usage.get("prompt_tokens") or 0,
                 self._estimated_prompt_tokens())
        if threshold and window and pt > threshold * window:
            self.console.print(
                f"[yellow]Context {100 * pt / window:.0f}% full — "
                f"auto-compacting...[/yellow]"
            )
            self.compact()

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        s = str(err)
        return any(marker in s for marker in
                   ("429", "500", "502", "503", "504", "Cannot reach",
                    "overloaded", "timed out"))

    @staticmethod
    def _is_context_overflow(err: Exception) -> bool:
        s = str(err).lower()
        return any(marker in s for marker in
                   ("context length", "context_length", "too long",
                    "maximum context", "context window"))

    def _run_turn(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()
        retries_left = 2
        compacted_already = False

        for _ in range(self.cfg.get("max_iterations", 40)):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            usage = {}
            think = ThinkFilter()
            state: dict = {"last_kind": None, "midline": False}

            try:
                for kind, payload in self.provider.chat(self.messages, tools.TOOL_SCHEMAS):
                    if kind == "text":
                        for fkind, ftext in think.feed(payload):
                            if fkind == "text":
                                text_parts.append(ftext)
                            self._print_stream(fkind, ftext, state)
                    elif kind == "thinking":
                        self._print_stream("thinking", payload, state)
                    elif kind == "tool_calls":
                        tool_calls = payload
                    elif kind == "usage":
                        usage = payload
                        self.last_usage = payload
                for fkind, ftext in think.flush():
                    if fkind == "text":
                        text_parts.append(ftext)
                    self._print_stream(fkind, ftext, state)
            except ProviderError as e:
                self._end_stream(state)
                nothing_streamed = not text_parts and not tool_calls
                if nothing_streamed and self._is_context_overflow(e) \
                        and not compacted_already and len(self.messages) > 3 \
                        and self.messages[-1]["role"] == "user":
                    compacted_already = True
                    self.console.print("[yellow]Context overflow — "
                                       "compacting and retrying...[/yellow]")
                    pending = self.messages.pop()  # keep the question verbatim
                    self.compact()
                    self.messages.append(pending)
                    continue
                if nothing_streamed and retries_left > 0 and self._is_retryable(e):
                    retries_left -= 1
                    delay = 2 ** (2 - retries_left)
                    self.console.print(f"[yellow]Transient provider error, "
                                       f"retrying in {delay}s...[/yellow] "
                                       f"[dim]{escape(str(e)[:120])}[/dim]")
                    time.sleep(delay)
                    continue
                self.had_error = True
                self.console.print(f"[red]Provider error:[/red] {escape(str(e))}")
                # keep history consistent: drop nothing, just stop the turn
                return
            except KeyboardInterrupt:
                self._end_stream(state)
                self.console.print("[yellow]Interrupted.[/yellow]")
                for fkind, ftext in think.flush():
                    if fkind == "text":
                        text_parts.append(ftext)
                partial = "".join(text_parts).strip()
                if partial:
                    self.messages.append({"role": "assistant", "content": partial})
                return
            finally:
                self._end_stream(state)

            content = "".join(text_parts).strip()
            self._print_usage(usage)

            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if not tool_calls:
                return  # model is done

            try:
                for tc in tool_calls:
                    self._run_tool(tc)
            except KeyboardInterrupt:
                # Keep history well-formed: every tool_call needs a result.
                self.console.print("\n[yellow]Interrupted.[/yellow]")
                answered = {
                    m.get("tool_call_id") for m in self.messages
                    if m["role"] == "tool"
                }
                for tc in tool_calls:
                    if tc["id"] not in answered:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": "Error: interrupted by the user before this ran.",
                        })
                return

        self.console.print("[red]Stopped: hit max iterations for one turn.[/red]")

    def _run_tool(self, tc: dict):
        name, args = tc["name"], tc["arguments"]
        self.console.print(
            f"[bold cyan]●[/bold cyan] [bold]{escape(name)}[/bold]"
            f"([white]{escape(_args_preview(name, args))}[/white])",
            markup=True, highlight=False,
        )
        if self._approve(name, args):
            result = tools.execute(
                name, args, self.cfg.get("max_tool_output_chars", 12000)
            )
        else:
            result = "Error: user denied this tool call. Ask before retrying."
            self.console.print("[red]  denied[/red]")
        all_lines = result.splitlines()
        shown = all_lines if name == "todo_write" else all_lines[:4]
        for line in shown:
            self.console.print("  " + line[:200], style="dim",
                               markup=False, highlight=False)
        if len(all_lines) > len(shown):
            self.console.print(
                f"  ... ({len(all_lines) - len(shown)} more lines)",
                style="dim",
            )
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": name,
            "content": result,
        })

    def compact(self):
        """Replace the conversation history with a model-written summary."""
        if len(self.messages) < 3:
            self.console.print("[dim]Nothing to compact yet.[/dim]")
            return
        ask = {
            "role": "user",
            "content": (
                "Summarize this session concisely for a fresh context: the task, "
                "key decisions, exact file paths touched and how, current state, "
                "and next steps. Output only the summary."
            ),
        }
        parts: list[str] = []
        think = ThinkFilter()
        state: dict = {"last_kind": None, "midline": False}
        try:
            for kind, payload in self.provider.chat(self.messages + [ask], []):
                if kind == "text":
                    for fkind, ftext in think.feed(payload):
                        if fkind == "text":
                            parts.append(ftext)
                        self._print_stream("thinking", ftext, state)
            for fkind, ftext in think.flush():
                if fkind == "text":
                    parts.append(ftext)
        except KeyboardInterrupt:
            self._end_stream(state)
            self.console.print("[yellow]Compact interrupted; history unchanged.[/yellow]")
            return
        except ProviderError as e:
            self._end_stream(state)
            self.console.print(f"[red]Compact failed: {escape(str(e))}[/red]")
            return
        finally:
            self._end_stream(state)
        summary = "".join(parts).strip()
        if not summary:
            self.console.print("[red]Compact failed: empty summary.[/red]")
            return
        self.messages = [
            self.messages[0],
            {"role": "user",
             "content": "Summary of the conversation so far:\n" + summary},
            {"role": "assistant",
             "content": "Understood. Continuing from that summary."},
        ]
        tools._READ_FILES.clear()  # file contents are no longer in context
        self.last_usage = {}
        self.console.print("[dim]History compacted.[/dim]")

    def print_context(self):
        """Show a context-usage report (/context)."""
        num_ctx = self._context_size()
        pt = self.last_usage.get("prompt_tokens")
        lines = [f"Model: {self.provider.model}"]
        if num_ctx:
            lines.append(f"Context window (num_ctx): {num_ctx:,} tokens")
        else:
            lines.append("Context window: model default (unknown)")
        if num_ctx and pt:
            pct = min(1.0, pt / num_ctx)
            bar = "█" * round(pct * 30) + "░" * (30 - round(pct * 30))
            lines.append(f"Used (last request): {bar} {pt:,} tokens ({pct:.0%})")
            threshold = self.cfg.get("auto_compact_threshold", 0.85)
            if threshold:
                lines.append(f"Auto-compact at: {threshold:.0%}")
        else:
            lines.append("Used: unknown (send a message first)")
        n_user = sum(1 for m in self.messages if m["role"] == "user")
        n_tool = sum(1 for m in self.messages if m["role"] == "tool")
        chars = sum(len(str(m.get("content") or "")) for m in self.messages)
        lines.append(
            f"History: {len(self.messages)} messages "
            f"({n_user} user, {n_tool} tool results), ~{chars:,} chars"
        )
        lines.append(f"Files read this session: {len(tools._READ_FILES)}")
        lines.append("Free it up: /compact (summarize) or /clear (start over)")
        self.console.print(Panel("\n".join(lines), title="context",
                                 border_style="cyan"))

    def _print_usage(self, usage: dict):
        if not usage:
            return
        secs = usage.get("seconds") or 0
        ct = usage.get("completion_tokens") or 0
        pt = usage.get("prompt_tokens") or 0
        rate = f" · {ct / secs:.0f} tok/s" if secs > 0 and ct else ""
        ctx = ""
        window = self._context_size()
        if window and pt:
            ctx = f" · ctx {100 * max(pt, self._estimated_prompt_tokens()) / window:.0f}%"
        if pt or ct:
            self.console.print(
                f"[dim]  {secs:.1f}s · {pt} prompt · {ct} gen{rate}{ctx}[/dim]"
            )

    def _trim_history(self):
        """Drop oldest turns when history gets too big (rough char budget)."""
        budget = (self._context_size() or 32768) * 3  # ~3 chars/token, conservative
        def size(m):
            return len(str(m.get("content") or "")) + (
                len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0
            )

        while len(self.messages) > 3 and sum(map(size, self.messages)) > budget:
            # remove the oldest non-system message, plus orphaned tool results
            del self.messages[1]
            while len(self.messages) > 1 and self.messages[1]["role"] == "tool":
                del self.messages[1]
