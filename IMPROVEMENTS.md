# Clyde CLI — Improvement Plan

From a deep code review (independent reviewer pass, 2026-07-02) plus a gap
analysis against Claude Code. Ordered so each phase is shippable on its own.

## Phase 0 — Bug fixes (do before any new features, ~half a day)

1. **Rich markup crash (critical).** Tool headers and approval panels
   interpolate raw strings with `markup=True` (`agent.py` `_run_tool` header,
   edit/write approval panels). A bash command like `ls [ab]*.py` or file
   content containing `[dim]` raises `MarkupError` and **kills the whole
   REPL**. Fix: `rich.markup.escape()` everywhere user/model content enters
   markup strings.
2. **Catch-all per turn.** Only `ProviderError`/`KeyboardInterrupt` are
   caught; a `json.JSONDecodeError` from a malformed stream line (proxy
   error page, truncated line) crashes the session. Wrap `json.loads` in the
   stream parsers → `ProviderError`, and wrap `run_turn` in the REPL with a
   broad handler that preserves the session.
3. **One-shot mode non-interactive safety.** `clyde "..."` without `--yolo`
   calls `console.input()` for approvals: at EOF every tool is denied; on a
   never-closing pipe it hangs forever. If `not sys.stdin.isatty()`: print a
   clear error suggesting `--yolo`, and exit non-zero on provider errors.
4. **`a` (always) is a footgun.** It silently flips global `--yolo`,
   auto-approving arbitrary bash for the session. Change to per-tool
   session allow (`always allow edit_file`), and require explicit `/yolo`
   for the global switch.
5. **Line-ending / encoding corruption.** `edit_file` rewrites CRLF files
   to LF (open with `newline=""` on read and write) and chokes on files
   `read_file` could read (`errors="replace"` mismatch — align them).
6. **Bash zombie hang.** Timeout kills only the direct bash process; a
   spawned server keeps the stdout pipe open and blocks forever. Use
   `start_new_session=True` + `os.killpg` on timeout.
7. **Context accounting for OpenAI providers.** They have no `num_ctx`, so
   trimming silently uses a 32k default (destroys history on 200k cloud
   models) and ctx%/auto-compact never work. Add `stream_options:
   {include_usage: true}` and a per-profile `context_window` config field.
8. **Ollama `prompt_eval_count` under-reports with KV cache hits**, so
   auto-compact may never fire. Track a client-side token estimate as
   backstop (max of reported and estimated).
9. **Config robustness.** Deep-merge user config (a user `profiles` key
   currently deletes the stock profiles); friendly error on invalid JSON.
10. **Read-gate holes.** Use `os.path.realpath` for `_READ_FILES`; a
    1-line `read_file(limit=1)` currently unlocks whole-file writes —
    only mark fully-read files (or track read ranges); consider mtime
    staleness checks.
11. **`resolve_num_ctx` edges.** Cap configured ints at model max even when
    the probe fails; `_is_local` substring check ("localhost.evil.com");
    guide the user to `ollama pull` when the model isn't installed.

## Phase 1 — Daily-driver essentials (~2 days)

1. **Session persistence + `--resume`/`--continue`.** Biggest gap: any
   crash or Ctrl-D loses everything. Save messages JSONL per session under
   `~/.local/share/clyde/sessions/`; `/resume` picker in the REPL.
2. **Persisted permission allowlists.** `allow: ["bash(git *)",
   "bash(pytest*)", "edit_file"]` in config; approval prompt gains an
   option to save the current command prefix.
3. **Real diff previews.** Unified diff (difflib) with color for
   edit/write approvals and after apply — approving edits fast and safely
   is the core UX of a coding agent.
4. **Persistent shell state.** `cd`/env/venv don't survive between bash
   calls, which constantly confuses models. Keep one long-lived shell, or
   minimally track cwd between calls.
5. **@file mentions.** `@src/app.py` injects contents directly — on slow
   local models every avoided tool round-trip is ~30s.
6. **Markdown rendering at end of stream** (rich.markdown re-render after
   the raw stream) — code blocks with syntax highlighting.
7. **Retry with backoff** on 429/transient 5xx; on context-overflow 400
   from cloud providers, auto-compact and retry once.
8. **Tests.** pytest for ThinkFilter chunk boundaries, trim pairing, SSE
   parsing, num_ctx math, tools (read gate, edit semantics). This code
   regresses silently otherwise.

## Phase 2 — Agentic power (~week, pick à la carte)

1. **Subagent tool** — spawn a fresh-context Agent for searches, return
   only conclusions. Disproportionately valuable with small local num_ctx.
2. **Checkpoint / `/undo`** — snapshot (git stash or copy) before mutating
   tools; weak local models misfire on edit_file often enough to need it.
3. **Streaming bash output + background processes** (`run_in_background`
   style) so dev servers and long builds don't block.
4. **MCP client support** — stdio MCP servers from config unlock a whole
   tool ecosystem.
5. **Workspace boundary** — approval for reads outside cwd (a cloud
   profile can currently read `~/.ssh/id_rsa` silently); simple secret
   redaction pass on tool results before they leave the machine.

## Cross-cutting

- `git init` + baseline commit (no history exists today for either repo).
- Extract shared code with clyde-desktop (ThinkFilter and the Ollama
  client are duplicated) into a small shared module, or have clydesk
  depend on the clyde package.
- Add ruff config; run it in tests.
