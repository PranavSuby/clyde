"""clyde: a minimal Claude Code-style terminal coding agent.

Usage:
  clyde                      # interactive REPL (default profile)
  clyde -P cloud             # use the 'cloud' profile
  clyde -m qwen3:8b          # override the model
  clyde "fix the tests"      # one-shot mode
  clyde --yolo               # skip approval prompts
"""

import argparse
import os
import re
import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markup import escape

from . import __version__, config, session as session_mod, tools
from .agent import Agent
from .providers import ProviderError, ensure_ollama_running, make_provider

_MENTION_RE = re.compile(r"@([~\w./\\-]+)")


def expand_mentions(text: str, console: Console) -> str:
    """Inline @path mentions: append file contents so the model doesn't
    need a read_file round-trip (expensive on slow local models)."""
    blocks = []
    for raw in dict.fromkeys(_MENTION_RE.findall(text)):
        path = os.path.expanduser(raw)
        if not os.path.isfile(path):
            # "look at @foo.py." — the sentence's punctuation is not the path
            trimmed = raw.rstrip(".,:;!?")
            if trimmed != raw and os.path.isfile(os.path.expanduser(trimmed)):
                raw, path = trimmed, os.path.expanduser(trimmed)
            else:
                continue
        try:
            with open(path, "r", errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        if len(content) > 20000:
            content = content[:20000] + "\n... (file truncated at 20k chars)"
        tools._mark_read(path)  # mentioned file may be edited without re-read
        blocks.append(f"--- {raw} ---\n{content}")
        console.print(f"[dim]attached {raw} ({len(content)} chars)[/dim]")
    if blocks:
        return text + "\n\nAttached files:\n" + "\n\n".join(blocks)
    return text

HELP = """\
Commands:
  /help              show this help
  /profile [name]    show or switch profile (local, cloud, ...)
  /model [name]      show or switch model within the current profile
  /models            list models available on the current backend
  /context           show context window usage and history size
  /resume            pick an earlier session to continue
  /undo              revert the last file edit/write
  /mcp               list connected MCP servers and their tools
  /clear             clear conversation history (and file-read tracking)
  /compact           summarize the conversation to free up context
  /yolo              toggle auto-approval of tools
  /exit              quit (also Ctrl-D)
@path/to/file in a message attaches that file's contents.
Anything else is sent to the model. Ctrl-C interrupts a running response."""


def _setup_provider(cfg, profile_name, model_override, console):
    profile = config.get_profile(cfg, profile_name)
    if profile.get("type") == "ollama":
        if not ensure_ollama_running(profile["base_url"], cfg.get("auto_start_ollama", True)):
            console.print(
                f"[red]Ollama is not reachable at {profile['base_url']} "
                f"and could not be started.[/red]"
            )
            sys.exit(1)
    provider = make_provider(profile, model_override)
    _resolve_ctx(provider, cfg, console)
    return provider, profile


def _resolve_ctx(provider, cfg, console):
    if hasattr(provider, "resolve_num_ctx"):
        _, note = provider.resolve_num_ctx(cfg.get("auto_ctx_cap", 65536))
        console.print(f"[dim]num_ctx: {note}[/dim]")


def _handle_slash(cmd: str, agent: Agent, cfg: dict, state: dict, console: Console) -> bool:
    """Handle a slash command. Returns False if the REPL should exit."""
    parts = cmd.split(maxsplit=1)
    name, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")

    if name in ("/exit", "/quit", "/q"):
        return False
    if name == "/help":
        console.print(HELP)
    elif name == "/clear":
        agent.clear()
        console.print("[dim]History cleared.[/dim]")
    elif name == "/compact":
        agent.compact()
    elif name == "/yolo":
        agent.yolo = not agent.yolo
        console.print(f"[dim]Auto-approve: {'on' if agent.yolo else 'off'}[/dim]")
    elif name == "/models":
        try:
            for m in agent.provider.list_models():
                console.print(f"  {m}")
        except ProviderError as e:
            console.print(f"[red]{e}[/red]")
    elif name == "/context":
        agent.print_context()
    elif name == "/undo":
        agent.undo()
    elif name == "/mcp":
        if not agent.mcp_servers:
            console.print("[dim]No MCP servers configured "
                          "(add mcp_servers to the config).[/dim]")
        for sname, server in agent.mcp_servers.items():
            tool_names = ", ".join(t["name"] for t in server.tools) or "(none)"
            console.print(f"  [bold]{sname}[/bold]: {tool_names}")
    elif name == "/resume":
        sessions = session_mod.list_sessions(cwd=agent.cwd) \
            or session_mod.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions yet.[/dim]")
            return True
        for i, s in enumerate(sessions, 1):
            age = time.strftime("%b %d %H:%M", time.localtime(s["updated"]))
            console.print(
                f"  [bold]{i}[/bold]. {age} · {s['turns']} turns · "
                f"[cyan]{s['model']}[/cyan] · {escape(s['first_prompt'][:60])}"
            )
        try:
            pick = console.input("[yellow]Resume which? (number/blank)[/yellow] ").strip()
        except EOFError:
            return True
        if pick.isdigit() and 1 <= int(pick) <= len(sessions):
            chosen = sessions[int(pick) - 1]
            data = session_mod.load(chosen["path"])
            agent.messages = data["messages"]
            state["session_path"] = chosen["path"]
            console.print(f"[dim]Resumed {chosen['path']} "
                          f"({chosen['turns']} turns).[/dim]")
    elif name == "/model":
        if not arg:
            console.print(f"Model: [bold]{agent.provider.model}[/bold]")
        else:
            agent.provider.model = arg
            agent.last_usage = {}
            console.print(f"[dim]Model set to {arg}[/dim]")
            _resolve_ctx(agent.provider, cfg, console)
    elif name == "/profile":
        if not arg:
            console.print(
                f"Profile: [bold]{state['profile']}[/bold] "
                f"(available: {', '.join(cfg['profiles'])})"
            )
        else:
            try:
                provider, _ = _setup_provider(cfg, arg, None, console)
            except (KeyError, ProviderError) as e:
                console.print(f"[red]{e}[/red]")
                return True
            agent.provider = provider
            agent.last_usage = {}
            state["profile"] = arg
            console.print(f"[dim]Switched to profile '{arg}' (model {provider.model})[/dim]")
    else:
        console.print(f"[red]Unknown command {name}. Try /help[/red]")
    return True


def main():
    parser = argparse.ArgumentParser(prog="clyde", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="one-shot prompt (omit for REPL)")
    parser.add_argument("-P", "--profile", default=None, help="config profile to use")
    parser.add_argument("-m", "--model", default=None, help="override the profile's model")
    parser.add_argument("-c", "--continue", dest="cont", action="store_true",
                        help="continue the most recent session in this directory")
    parser.add_argument("--yolo", action="store_true", help="auto-approve all tool calls")
    parser.add_argument("--version", action="version", version=f"clyde {__version__}")
    args = parser.parse_args()

    console = Console()
    try:
        cfg = config.load_config()
    except config.ConfigError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    profile_name = args.profile or cfg.get("default_profile", "local")

    try:
        provider, _profile = _setup_provider(cfg, profile_name, args.model, console)
    except (KeyError, ProviderError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    agent = Agent(provider, console, cfg, yolo=args.yolo)
    if cfg.get("mcp_servers"):
        from . import mcp
        agent.mcp_servers = mcp.load_servers(cfg, console)

    session_path = session_mod.new_session_path()
    if args.cont:
        recent = session_mod.list_sessions(cwd=os.getcwd(), limit=1)
        if recent:
            data = session_mod.load(recent[0]["path"])
            agent.messages = data["messages"]
            session_path = recent[0]["path"]
            console.print(f"[dim]Continuing session with "
                          f"{recent[0]['turns']} prior turns.[/dim]")
        else:
            console.print("[dim]No previous session here; starting fresh.[/dim]")

    def save_session():
        try:
            session_mod.save(state["session_path"], agent, state["profile"])
        except OSError as e:
            console.print(f"[dim]session save failed: {e}[/dim]")

    state = {"profile": profile_name, "session_path": session_path}

    def close_mcp():
        for server in agent.mcp_servers.values():
            server.close()

    # One-shot mode
    if args.prompt:
        if not args.yolo and not sys.stdin.isatty():
            console.print(
                "[red]stdin is not a terminal, so approval prompts can't be "
                "answered. Re-run with --yolo for non-interactive use.[/red]"
            )
            sys.exit(2)
        try:
            agent.run_turn(expand_mentions(" ".join(args.prompt), console))
        finally:
            save_session()
            close_mcp()
        sys.exit(1 if agent.had_error else 0)

    console.print(
        f"[bold]clyde[/bold] v{__version__} · profile [cyan]{profile_name}[/cyan] "
        f"· model [cyan]{provider.model}[/cyan]\n"
        f"[dim]/help for commands · config: {config.CONFIG_PATH}[/dim]"
    )

    session = PromptSession(history=FileHistory(config.HISTORY_PATH))

    def rprompt():
        pct = agent.ctx_percent()
        label = agent.provider.model
        if pct is not None:
            label += f" · ctx {pct}%"
        return label

    try:
        _repl_loop(session, agent, cfg, state, console, rprompt, save_session)
    finally:
        close_mcp()
    console.print("[dim]bye[/dim]")


def _repl_loop(session, agent, cfg, state, console, rprompt, save_session):
    while True:
        try:
            line = session.prompt("\n❯ ", rprompt=rprompt).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_slash(line, agent, cfg, state, console):
                break
            save_session()
            continue
        try:
            agent.run_turn(expand_mentions(line, console))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:  # never lose the session to a bug
            import traceback
            traceback.print_exc()
            console.print(
                f"[red]Internal error ({type(e).__name__}) — session preserved. "
                f"Please report the traceback above.[/red]"
            )
        save_session()


if __name__ == "__main__":
    main()
