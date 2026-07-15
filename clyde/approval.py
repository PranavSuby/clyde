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
        # session-scoped allow rules ("bash(git *)", "edit_file"), matched
        # with the same semantics as persisted permission rules
        self.session_allow: set[str] = set()
        # 'a' answer to the taint re-approval prompt: the user has decided to
        # trust content ingested this session, stop re-gating mutations
        self.taint_trusted = False

    @staticmethod
    def rule_for(name: str, args: dict) -> str:
        """The allow-rule suggestion for the 'p' (persist) approval answer."""
        if name == "bash":
            first = (args.get("command", "").strip().split() or ["?"])[0]
            return f"bash({first} *)"
        return name

    def approve(self, name: str, args: dict, preview=None,
                tainted: bool = False) -> bool:
        """True if this tool call may run, prompting the user if needed.

        `preview` (optional callable) is invoked just before the prompt to
        show the user what the call would change.

        `tainted` means untrusted tool output has entered the context this
        session. Auto-approval paths (--yolo, allow rules) then no longer
        cover mutating tools: an injected instruction that survives the
        model must still get past an explicit human confirmation. Measured
        in the injection eval: this is what takes *executed* attack success
        to 0% even under --yolo.
        """
        needs_approval = name in tools.APPROVAL_REQUIRED \
            or name.startswith("mcp__")  # MCP tools may have side effects
        regate = (tainted and not self.taint_trusted
                  and self.cfg.get("taint_reapproval", True)
                  and (name in tools.MUTATING_TOOLS
                       or name.startswith("mcp__")))
        if self.yolo or not needs_approval:
            return self._confirm_tainted(name, preview) if regate else True
        for rule in self.session_allow:
            if config_mod.rule_matches(rule, name, args):
                return self._confirm_tainted(name, preview) if regate else True
        for rule in self.cfg.get("permissions", {}).get("allow", []):
            if config_mod.rule_matches(rule, name, args):
                return self._confirm_tainted(name, preview) if regate else True
        if preview is not None:
            preview()
        rule = self.rule_for(name, args)
        try:
            answer = self.console.input(
                f"[yellow]Allow {escape(name)}? \\[y/n/a=allow "
                f"{escape(rule)} this session"
                f"/p=permanently allow {escape(rule)}][/yellow] "
            ).strip().lower()
        except EOFError:
            return False
        if answer == "a":
            self.session_allow.add(rule)
            self.console.print(f"[dim]{escape(rule)} auto-approved for this "
                               f"session (/yolo for everything)[/dim]")
            return True
        if answer == "p":
            allow = self.cfg.setdefault("permissions", {}).setdefault("allow", [])
            if rule not in allow:
                allow.append(rule)
            config_mod.add_allow_rule(rule)
            self.console.print(f"[dim]saved allow rule: {escape(rule)} "
                               f"({config_mod.CONFIG_PATH})[/dim]")
            return True
        return answer in ("y", "yes")

    def _confirm_tainted(self, name: str, preview=None) -> bool:
        """Re-approval for a mutating tool after untrusted ingestion.

        Fires only where the normal gate would have auto-approved (--yolo or
        an allow rule): an interactive prompt is already an explicit human
        check, so it is not doubled. Silence / no TTY fails closed — a
        headless --yolo run cannot mutate anything after reading untrusted
        content unless taint_reapproval is turned off in the config."""
        if preview is not None:
            preview()
        try:
            answer = self.console.input(
                f"[yellow]⚠ untrusted tool output has entered this session — "
                f"re-approve {escape(name)}? \\[y/n/a=trust session content, "
                f"stop asking][/yellow] "
            ).strip().lower()
        except (EOFError, OSError):
            return False  # fail closed
        if answer == "a":
            self.taint_trusted = True
            self.console.print(
                "[dim]taint re-approval off for the rest of this session "
                "(config: taint_reapproval)[/dim]")
            return True
        return answer in ("y", "yes")

    def confirm_exfil(self, name: str, detail: str) -> bool:
        """Exfiltration guard confirmation. Unlike normal approval this fires
        even under --yolo and even for allow-ruled tools: sending read data or a
        credential to an outbound tool is exactly the case a blanket 'yes'
        should not cover. Silence / no TTY (EOFError) means block."""
        try:
            answer = self.console.input(
                f"[red]⚠ possible exfiltration:[/red] {escape(name)} would send "
                f"{escape(detail)} to an outbound tool. Allow? \\[y/N] "
            ).strip().lower()
        except (EOFError, OSError):
            return False  # no interactive TTY → fail closed (block)
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
