"""Tool-call approval policy: decides whether a tool call may run.

Combines the --yolo flag, per-session "always allow" answers, persisted
allow rules from the config, and the interactive y/n/a/p prompt. Rendering
of change previews stays with the caller (passed in as a callback) so the
policy holds no drawing logic.
"""

from rich.markup import escape

from . import config as config_mod
from . import tools


class ApprovalPolicy:
    def __init__(self, cfg: dict, console, yolo: bool = False):
        self.cfg = cfg
        self.console = console
        self.yolo = yolo
        self.session_allow: set[str] = set()

    @staticmethod
    def rule_for(name: str, args: dict) -> str:
        """The allow-rule suggestion for the 'p' (persist) approval answer."""
        if name == "bash":
            first = (args.get("command", "").strip().split() or ["?"])[0]
            return f"bash({first} *)"
        return name

    def approve(self, name: str, args: dict, preview=None) -> bool:
        """True if this tool call may run, prompting the user if needed.

        `preview` (optional callable) is invoked just before the prompt to
        show the user what the call would change.
        """
        needs_approval = name in tools.APPROVAL_REQUIRED \
            or name.startswith("mcp__")  # MCP tools may have side effects
        if self.yolo or name in self.session_allow or not needs_approval:
            return True
        for rule in self.cfg.get("permissions", {}).get("allow", []):
            if config_mod.rule_matches(rule, name, args):
                return True
        if preview is not None:
            preview()
        persist_rule = self.rule_for(name, args)
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

    def confirm_outside_read(self, path: str) -> bool:
        """Extra confirmation for reads that leave the workspace."""
        if self.yolo:
            return True
        try:
            answer = self.console.input(
                f"[yellow]Reads outside the workspace "
                f"({escape(path)}) — allow? \\[y/n][/yellow] "
            ).strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")
