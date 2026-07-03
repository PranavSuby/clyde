"""The agent loop: stream model output, execute tool calls, repeat."""

import datetime
import json
import os
import platform
import subprocess

from rich.console import Console
from rich.panel import Panel

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
        self.messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self.cwd)}
        ]

    def clear(self):
        self.messages = [self.messages[0]]
        self.last_usage = {}
        tools._READ_FILES.clear()

    def ctx_percent(self) -> int | None:
        num_ctx = getattr(self.provider, "num_ctx", None)
        pt = self.last_usage.get("prompt_tokens")
        if num_ctx and pt:
            return round(100 * pt / num_ctx)
        return None

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _print_stream(self, kind: str, text: str, state: dict):
        """Print streamed tokens, tracking whether we're mid-line."""
        style = "dim italic" if kind == "thinking" else None
        if state.get("last_kind") not in (None, kind):
            self.console.print()  # separate thinking from answer
        state["last_kind"] = kind
        self.console.print(text, style=style, end="", markup=False, highlight=False)
        state["midline"] = not text.endswith("\n")

    def _end_stream(self, state: dict):
        if state.get("midline"):
            self.console.print()
            state["midline"] = False

    def _approve(self, name: str, args: dict) -> bool:
        if self.yolo or name not in tools.APPROVAL_REQUIRED:
            return True
        if name == "edit_file":
            body = (
                f"[red]- {args.get('old_string', '')[:1000]}[/red]\n"
                f"[green]+ {args.get('new_string', '')[:1000]}[/green]"
            )
            self.console.print(Panel(body, title=f"edit {args.get('path', '')}",
                                     border_style="yellow"))
        elif name == "write_file":
            preview = args.get("content", "")[:1500]
            self.console.print(Panel(preview, title=f"write {args.get('path', '')}",
                                     border_style="yellow"))
        try:
            answer = self.console.input(
                f"[yellow]Allow {name}? \\[y/n/a=always][/yellow] "
            ).strip().lower()
        except EOFError:
            return False
        if answer == "a":
            self.yolo = True
            return True
        return answer in ("y", "yes")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_turn(self, user_input: str):
        self._run_turn(user_input)
        threshold = self.cfg.get("auto_compact_threshold", 0.85)
        num_ctx = getattr(self.provider, "num_ctx", None)
        pt = self.last_usage.get("prompt_tokens", 0)
        if threshold and num_ctx and pt > threshold * num_ctx:
            self.console.print(
                f"[yellow]Context {100 * pt / num_ctx:.0f}% full — "
                f"auto-compacting...[/yellow]"
            )
            self.compact()

    def _run_turn(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

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
                self.console.print(f"[red]Provider error:[/red] {e}")
                # keep history consistent: drop nothing, just stop the turn
                return
            except KeyboardInterrupt:
                self._end_stream(state)
                self.console.print("[yellow]Interrupted.[/yellow]")
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
            f"[bold cyan]●[/bold cyan] [bold]{name}[/bold]"
            f"([white]{_args_preview(name, args)}[/white])",
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
        except (ProviderError, KeyboardInterrupt) as e:
            self._end_stream(state)
            self.console.print(f"[red]Compact failed: {e}[/red]")
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
        num_ctx = getattr(self.provider, "num_ctx", None)
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
        num_ctx = getattr(self.provider, "num_ctx", None)
        if num_ctx and pt:
            ctx = f" · ctx {100 * pt / num_ctx:.0f}%"
        if pt or ct:
            self.console.print(
                f"[dim]  {secs:.1f}s · {pt} prompt · {ct} gen{rate}{ctx}[/dim]"
            )

    def _trim_history(self):
        """Drop oldest turns when history gets too big (rough char budget)."""
        num_ctx = getattr(self.provider, "num_ctx", None) or 32768
        budget = num_ctx * 3  # ~3 chars per token, conservative
        def size(m):
            return len(str(m.get("content") or "")) + (
                len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0
            )

        while len(self.messages) > 3 and sum(map(size, self.messages)) > budget:
            # remove the oldest non-system message, plus orphaned tool results
            del self.messages[1]
            while len(self.messages) > 1 and self.messages[1]["role"] == "tool":
                del self.messages[1]
