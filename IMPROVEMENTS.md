# Clyde CLI — Improvement Plan

From a deep code review (independent reviewer pass, 2026-07-02) plus a gap
analysis against Claude Code. Ordered so each phase is shippable on its own.

**Status (2026-07-08):** Phases 0–2 are implemented and shipped. A second
review pass found a set of trust-boundary and robustness regressions, all
fixed in Phase 3 below.

## Phase 3 — Security & robustness hardening (2026-07-08) — DONE

1. **Allow-rule shell-chaining bypass (critical).** A persisted
   `bash(git *)` rule matched `git status; rm -rf ~`, `git log | sh`,
   `$(...)`, backticks, redirects, and newlines. `config.rule_matches` now
   rejects any command containing shell metacharacters before prefix
   matching, and path rules match the resolved real path (no `../` escape).
2. **Subagent read-boundary bypass (critical).** The `task` subagent called
   tools directly, skipping the workspace read gate — it could read
   `~/.aws/credentials` with no prompt. Subagent reads now go through
   `confirm_outside_read` like top-level calls.
3. **`lean_check` code execution (major).** `open IO in #eval …` walked past
   the substring blocklist and ran shell commands at elaboration time. Added
   `#eval`/`run_cmd`/`elab`/`macro`/`open IO`/`open System` to the blocklist.
4. **Ctrl-C orphaned bash process groups (major).** Interrupting a
   foreground command left it (and its children) running. The handler now
   kills the process group and reaps on any `BaseException`.
5. **Compact-on-overflow self-defeat (major).** `compact()` re-sent the same
   oversized history to summarize and wedged the session. It now halves the
   history and retries on overflow.
6. **Bogus `/undo` checkpoints (major).** Failed edits pushed a no-op
   checkpoint. Checkpoints are now discarded when the mutation returns an
   error.
7. **Corrupt/foreign session files crashed `-c` and `/resume` (major).**
   `list_sessions`/`load` now validate shape and the callers handle load
   errors; save uses pid-unique tmp names.
8. **Minors:** MCP `_notify` BrokenPipe + `close()` killpg/reap + bounded
   handshake timeout; `glob` `../` escape closed; malformed tool-call JSON
   surfaced to the model instead of running with `{}`; redaction now covers
   AWS-secret/`AIza`/`Bearer`/`.env KEY=value`; the `a` approval answer is
   scoped to a rule (not the whole tool); `p` persists to the user config
   only; Ctrl-C cancels prompts cleanly; slash commands are inside the REPL
   exception guard. Added `[tool.ruff]` config and regression tests.

## Phase 4 — Resume & robustness follow-ups (2026-07-08) — DONE

1. **`/resume` and `-c` now restore the saved profile/model** and rebuild the
   system prompt for the *current* cwd (tools resolve against it), warning
   when the resumed session was started elsewhere. CLI `-P`/`-m` still win.
2. **Trailing-`&` foreground bash no longer hangs.** The pipe is drained in a
   reader thread; once bash exits we stop waiting instead of blocking on a
   backgrounded grandchild holding stdout, reap the group, and tell the model
   to use `run_in_background`.
3. **Concurrent `clyde -c` can't clobber a session.** A pid-stamped lock file
   (with stale-lock reclaim) makes a second `-c` in the same directory fork
   to a new session instead of overwriting the first.
4. **Token accounting is script-aware.** `estimate_tokens` counts ASCII at
   ~4 chars/token and non-ASCII at ~1, so CJK/code-dense history no longer
   under-counts; `_trim_history` uses it and targets 90% of the window.
5. **MCP failure paths tested.** New tests cover a server dying mid-handshake
   (non-fatal), a silent server (bounded handshake), and `close()` reaping
   the whole process group.

Still open (deliberately deferred): bare AWS *secret* keys with no key label
remain unredacted — hard to distinguish a 40-char secret from any base64 blob.

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
