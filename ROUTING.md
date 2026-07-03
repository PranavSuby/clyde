# How clyde routes your requests

This explains where your prompts go, how the backend is chosen, and how the
context window is sized. Nothing here is magic — it's all driven by
`~/.config/clyde/config.json`.

## The big picture

```
you type a prompt
      │
      ▼
┌─ active profile ──────────────────────────────────────────┐
│  "local"      → Ollama daemon on this PC (native API)     │
│  "cloud"      → Ollama Cloud model via the local daemon   │
│  "openrouter" → any OpenAI-compatible API over HTTPS      │
└───────────────────────────────────────────────────────────┘
      │
      ▼
agent loop: model responds → tool calls (bash, read_file, edit_file, ...)
→ results go back to the model → repeat until it answers in text
```

The profile is chosen in this order:
1. `clyde -P <name>` flag at launch
2. `/profile <name>` inside the REPL (switches mid-session)
3. `default_profile` in the config (ships as `"local"`)

`clyde -m <model>` (or `/model <name>`) overrides only the model, staying on
the same backend.

## Provider types

Each profile has a `type` that picks the wire protocol:

| type | endpoint | used for | why |
|---|---|---|---|
| `ollama` | `POST {base_url}/api/chat` (native) | local + Ollama Cloud | supports `num_ctx`, streamed tool calls, `thinking` field |
| `openai` | `POST {base_url}/chat/completions` | OpenRouter, Groq, any OpenAI-compatible host | universal standard for online providers |

Both are translated from one internal message format, so switching backends
mid-conversation keeps your history.

### Where each stock profile actually sends data

- **local** — never leaves your PC. `http://localhost:11434`.
- **cloud** — goes to Ollama's datacenter GPUs (`*-cloud` models are proxied
  through your local daemon after `ollama signin`). Ollama requires
  no-logging / no-training / zero-retention from its compute partners.
- **openrouter** — goes to openrouter.ai, which forwards to a downstream GPU
  host. Opt out of training in your OpenRouter account settings, or the
  request may route to hosts that train on prompts. Never use `:free`
  models for private code — those explicitly require logging/training.

## Context window sizing (num_ctx)

Ollama's out-of-the-box default (4k) is useless for agents, so clyde always
sets `num_ctx` explicitly for `ollama`-type profiles:

- `"num_ctx": "auto"` (default for `local`) — at startup clyde probes
  `/api/show` for the model's architecture and computes:
  - the model's **own max context** (hard cap — e.g. qwen3:8b caps at 40,960)
  - the KV-cache cost per token (layers × KV heads × head dim)
  - what fits in **your** VRAM after the weights, plus 25% of free RAM
  - the `auto_ctx_cap` config value (default 65,536, a speed guardrail)
  The smallest of those wins, rounded to a multiple of 4,096.
- `"num_ctx": <integer>` — used as-is, but still capped at the model's max.
- omitted (the `cloud` profile) — the server picks; cloud models manage
  their own context.

The resolved value is printed at startup (`num_ctx: auto → 65,536 ...`) and
re-resolved whenever you `/model` switch.

## Context tracking and compaction

- Every response ends with a stats line: `4.2s · 2481 prompt · 95 gen ·
  15 tok/s · ctx 8%`. `ctx` = how full the window was for that request.
  The same percentage sits at the right of the input prompt.
- `/context` prints a full report (usage bar, message counts, files read).
- At `auto_compact_threshold` (default 85%) the history is auto-summarized
  by the model and replaced — like Claude Code's compaction. `/compact`
  does it on demand; `/clear` wipes instead. Both reset read-file tracking,
  so the agent must re-read files before editing them again.
- Oldest turns are silently trimmed if history somehow outgrows the budget.

## Adding a new backend

Add a profile to `~/.config/clyde/config.json`:

```json
"groq": {
  "type": "openai",
  "base_url": "https://api.groq.com/openai/v1",
  "model": "qwen/qwen3-32b",
  "api_key_env": "GROQ_API_KEY"
}
```

Then `clyde -P groq` or `/profile groq`. Any host that speaks the OpenAI
chat-completions protocol with tool calling works.
