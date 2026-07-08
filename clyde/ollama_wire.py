"""Shared Ollama /api/chat wire-protocol helpers.

Both the CLI provider (sync generator) and the desktop client (async
generator) speak the same native Ollama protocol. The streaming loops can't
be shared across the sync/async boundary, but the per-chunk parsing is
identical and lives here so the two clients can't drift apart.
"""

import json
import uuid
from urllib.parse import urlparse


def new_call_id() -> str:
    """A synthetic tool-call id (Ollama's native API doesn't supply one)."""
    return "call_" + uuid.uuid4().hex[:12]


def parse_tool_calls(msg: dict) -> list[dict]:
    """Normalize a message's tool_calls into the internal format
    [{"id", "name", "arguments": dict}]. Tolerates string-encoded arguments
    (some models emit the arguments object as a JSON string)."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            raw = args
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                # let the tool layer report the parse failure to the model
                args = {"_malformed_json": raw}
        if not isinstance(args, dict):
            args = {"_malformed_json": json.dumps(args)}
        calls.append({
            "id": new_call_id(),
            "name": fn.get("name", ""),
            "arguments": args,
        })
    return calls


def parse_usage(chunk: dict) -> dict:
    """Token usage from a done-chunk (Ollama's prompt_eval_count/eval_count)."""
    return {
        "prompt_tokens": chunk.get("prompt_eval_count", 0),
        "completion_tokens": chunk.get("eval_count", 0),
    }


def is_local_url(base_url: str) -> bool:
    """True if base_url points at an Ollama daemon on this machine."""
    host = urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")
