# clyde

A minimal Claude Code-style coding agent for your terminal. Runs against a
**local model** (Ollama on your PC) or an **online model** (Ollama Cloud,
OpenRouter, or any OpenAI-compatible API) — switch with one flag.

## Install

```bash
cd ~/Documents/clyde
python3 -m venv .venv
.venv/bin/pip install -e .
ln -sf ~/Documents/clyde/.venv/bin/clyde ~/.local/bin/clyde
```

## Usage

```bash
cd ~/some/project
clyde                      # REPL with the default (local) profile
clyde -P cloud             # use the cloud profile
clyde -m qwen3:8b          # override the model for this session
clyde "why do the tests fail?"   # one-shot mode
clyde --yolo               # skip tool approval prompts
```

In the REPL: `/profile cloud`, `/model qwen3:8b`, `/models`, `/context`
(usage report), `/resume` (pick an earlier session), `/undo` (revert the
last file change), `/mcp`, `/clear`, `/compact`, `/yolo`, `/help`, `/exit`.
Ctrl-C interrupts a response, Ctrl-D quits.

## Sessions

Every turn is saved to `~/.local/share/clyde/sessions/`. `clyde -c`
continues the most recent session in the current directory; `/resume`
lists recent ones to pick from. A crash never loses a conversation.

## Approvals & permissions

Mutating tools prompt with a real unified diff. Answers: `y` once, `n`
deny, `a` always allow *that tool* this session, `p` permanently save an
allow rule (e.g. `bash(git *)`) to the config. Rules live under
`permissions.allow`. Reads outside the working directory prompt separately,
and obvious secrets (AWS/GitHub/Slack/API keys, private key blocks) are
redacted from tool results before they enter the conversation.

## Agentic extras

- `@path/to/file` in a message attaches the file directly (no tool round-trip).
- The model can spawn a **read-only subagent** (`task` tool) for broad
  searches, keeping its own context small — valuable at local context sizes.
- Long commands stream output live; `run_in_background: true` starts
  servers/builds detached (`bash_output` / `bash_kill` to manage them).
- `cd` persists between bash calls, and relative paths follow it.
- **MCP**: add stdio servers under `mcp_servers` in the config; their tools
  appear as `mcp__server__tool` (approval required).
- Transient provider errors retry with backoff; context-overflow errors
  auto-compact and retry once.

## Context management

- The `local` profile defaults to `"num_ctx": "auto"`: clyde reads the
  model's architecture from Ollama (max context length, KV-cache cost per
  token) and your PC's VRAM/free RAM, then picks the largest context that
  fits — never more than the model's own maximum, and never more than
  `auto_ctx_cap` (default 65536; raise it in the config if you want more).
  A fixed integer `num_ctx` is also capped at the model's maximum.
- Every response shows `ctx N%` (how full the window is), and the same
  indicator lives at the right of the input prompt. `/context` shows a full
  report with a usage bar.
- When usage passes `auto_compact_threshold` (default 85%), the history is
  automatically summarized (`/compact` does it manually; set the threshold
  to 0 to disable). `/clear` wipes history entirely.
- When clyde auto-starts `ollama serve` it enables flash attention. For even
  bigger contexts you can put `OLLAMA_KV_CACHE_TYPE=q8_0` in your
  environment to halve KV-cache memory.

## Profiles

Config lives at `~/.config/clyde/config.json`. Three profiles ship by default:

| Profile      | What it is                                                       |
|--------------|------------------------------------------------------------------|
| `local`      | Ollama on this PC (`qwen3-coder:30b`, auto-sized context)        |
| `cloud`      | Ollama Cloud model proxied through local Ollama (`ollama signin` once, then any `*-cloud` model works, e.g. `qwen3-coder:480b-cloud`) |
| `openrouter` | OpenRouter (or edit to any OpenAI-compatible API); needs `OPENROUTER_API_KEY` in your environment |

Profile fields:

```jsonc
{
  "type": "ollama" | "openai",   // ollama = native API (supports num_ctx), openai = /v1 chat completions
  "base_url": "http://localhost:11434",
  "model": "qwen3-coder:30b",
  "num_ctx": 32768,              // ollama only: context window
  "api_key_env": "SOME_ENV_VAR"  // optional: env var holding the API key
}
```

To use Ollama Cloud directly (without the local daemon as proxy):
`{"type": "ollama", "base_url": "https://ollama.com", "model": "qwen3-coder:480b", "api_key_env": "OLLAMA_API_KEY"}`.

## Tools

The agent can call: `bash`, `read_file`, `write_file`, `edit_file`,
`todo_write`, `list_dir`, `glob`, `grep`. Mutating tools (`bash`,
`write_file`, `edit_file`) prompt for approval — answer `y`, `n`, or `a`
(always, this session). The agent must `read_file` an existing file before
it is allowed to edit or overwrite it.

## Project context

If the working directory contains a `CLYDE.md`, `AGENTS.md`, or `CLAUDE.md`,
its contents are added to the system prompt.

## Notes

- Local tool calling works best with `qwen3-coder:30b`; the smaller `qwen3`
  models also support tools (their `<think>` reasoning is shown dimmed).
- Ollama's default 4k context is too small for agent work — that's why
  clyde always sets `num_ctx` explicitly (see Context management above).
- History is trimmed oldest-first when it exceeds the context budget.
