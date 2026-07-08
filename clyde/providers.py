"""Model backends.

Internal message format (provider-neutral, OpenAI-flavored):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id": str, "name": str, "arguments": dict}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Providers yield a stream of events:
  ("thinking", str)   - reasoning tokens
  ("text", str)       - answer tokens
  ("tool_calls", [..])- normalized tool calls, once, at the end
  ("usage", dict)     - {"prompt_tokens", "completion_tokens", "seconds"}
"""

import json
import os
import shutil
import subprocess
import time

import httpx

from .ollama_wire import is_local_url, new_call_id, parse_tool_calls, parse_usage


class ProviderError(Exception):
    """A backend request failed. `retryable` marks transient faults
    (rate limits, 5xx, unreachable host) worth an automatic retry."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ContextOverflowError(ProviderError):
    """The prompt exceeded the model's context window."""


# The wire gives no typed signal for these; classify the error text once,
# here at the boundary, so callers can dispatch on the exception type.
_OVERFLOW_MARKERS = ("context length", "context_length", "too long",
                     "maximum context", "context window")
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _classify_error(message: str, status: int | None = None) -> ProviderError:
    low = message.lower()
    if any(m in low for m in _OVERFLOW_MARKERS):
        return ContextOverflowError(message)
    retryable = status in _RETRYABLE_STATUS \
        or "overloaded" in low or "timed out" in low
    return ProviderError(message, retryable=retryable)


def _gpu_vram_bytes() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return sum(int(x) for x in out.stdout.split()) * 1024 ** 2
    except Exception:
        pass
    return 0


def _ram_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 8 * 1024 ** 3


class BaseProvider:
    type = "base"

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: list[dict], tools: list[dict]):
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError


class OllamaProvider(BaseProvider):
    """Ollama native /api/chat: tool calling, streaming, and num_ctx control."""

    type = "ollama"

    def __init__(self, base_url, model, api_key=None, num_ctx=None):
        super().__init__(base_url, model, api_key)
        # num_ctx setting: int, "auto", or None (model default)
        self.num_ctx_setting = num_ctx
        self.num_ctx = num_ctx if isinstance(num_ctx, int) else None

    def _is_local(self) -> bool:
        return is_local_url(self.base_url)

    def model_details(self) -> dict:
        """Probe the model: max context, KV cache cost/token, weight size."""
        resp = self.client.post(
            f"{self.base_url}/api/show",
            json={"model": self.model, "name": self.model},
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        info = data.get("model_info") or {}
        arch = info.get("general.architecture", "")

        def g(key):
            return info.get(f"{arch}.{key}")

        heads, emb = g("attention.head_count"), g("embedding_length")
        head_dim = g("attention.key_length") or (emb // heads if heads and emb else 0)
        layers, kv_heads = g("block_count"), g("attention.head_count_kv")
        kv_per_tok = None
        if layers and kv_heads and head_dim:
            kv_per_tok = 2 * layers * kv_heads * head_dim * 2  # K+V, f16

        weights = None
        try:
            tags = self.client.get(f"{self.base_url}/api/tags",
                                   headers=self._headers()).json()
            names = {self.model, self.model + ":latest",
                     self.model.removesuffix(":latest")}
            for m in tags.get("models", []):
                if m.get("name") in names:
                    weights = m.get("size")
        except httpx.HTTPError:
            pass
        return {
            "max_ctx": g("context_length"),
            "kv_bytes_per_token": kv_per_tok,
            "weights_bytes": weights,
            "quant": (data.get("details") or {}).get("quantization_level"),
        }

    def resolve_num_ctx(self, auto_cap: int = 65536) -> tuple[int | None, str]:
        """Turn the num_ctx setting into a concrete value for this model/PC.

        Returns (num_ctx, human-readable note). Caps at the model's own max
        context; 'auto' also fits the KV cache into VRAM + a slice of RAM.
        """
        setting = self.num_ctx_setting
        if not setting:
            self.num_ctx = None
            return None, "model default (set num_ctx in config to control it)"
        try:
            det = self.model_details()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                fallback = setting if isinstance(setting, int) else 16384
                self.num_ctx = fallback
                return fallback, (f"{fallback} (model '{self.model}' is not "
                                  f"installed — run: ollama pull {self.model})")
            fallback = setting if isinstance(setting, int) else 16384
            self.num_ctx = fallback
            return fallback, f"{fallback} (model probe failed: {e})"
        except (httpx.HTTPError, ProviderError, ValueError) as e:
            fallback = setting if isinstance(setting, int) else 16384
            self.num_ctx = fallback
            return fallback, f"{fallback} (model probe failed: {e})"

        max_ctx = det["max_ctx"] or auto_cap
        if isinstance(setting, int):
            n = min(setting, max_ctx)
            self.num_ctx = n
            note = str(n) if n == setting else \
                f"{n} (configured {setting}, capped to model max {max_ctx})"
            return n, note

        # "auto": fit KV cache in VRAM after weights, plus 25% of free RAM
        kv = det["kv_bytes_per_token"] or 100 * 1024
        if self._is_local():
            vram, ram = _gpu_vram_bytes(), _ram_available_bytes()
            weights = det["weights_bytes"] or 0
            budget = max(0, vram - weights - int(1.5 * 1024 ** 3)) + int(0.25 * ram)
            n = min(max_ctx, auto_cap, budget // kv)
        else:
            n = min(max_ctx, auto_cap)
        n = max(4096, int(n // 4096) * 4096)
        self.num_ctx = n
        return n, (
            f"auto → {n:,} (model max {max_ctx:,}, "
            f"KV cache {kv // 1024} KiB/token)"
        )

    def _to_wire(self, messages: list[dict]) -> list[dict]:
        wire = []
        for m in messages:
            if m["role"] == "assistant":
                wm = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    wm["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in m["tool_calls"]
                    ]
                wire.append(wm)
            elif m["role"] == "tool":
                wire.append({
                    "role": "tool",
                    "tool_name": m.get("name", ""),
                    "content": m["content"],
                })
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    def chat(self, messages, tools):
        payload = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if self.num_ctx:
            payload["options"] = {"num_ctx": self.num_ctx}

        tool_calls = []
        usage = {}
        start = time.time()
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/api/chat",
                json=payload, headers=self._headers(),
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode(errors="replace")
                    raise _classify_error(
                        f"Ollama HTTP {resp.status_code}: {body[:500]}",
                        resp.status_code)
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue  # tolerate malformed/truncated stream lines
                    if chunk.get("error"):
                        raise _classify_error(f"Ollama: {chunk['error']}")
                    msg = chunk.get("message", {})
                    if msg.get("thinking"):
                        yield ("thinking", msg["thinking"])
                    if msg.get("content"):
                        yield ("text", msg["content"])
                    tool_calls.extend(parse_tool_calls(msg))
                    if chunk.get("done"):
                        usage = parse_usage(chunk)
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach Ollama at {self.base_url}: {e}",
                                retryable=True) from e

        if tool_calls:
            yield ("tool_calls", tool_calls)
        usage["seconds"] = time.time() - start
        yield ("usage", usage)

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", headers=self._headers())
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot list models: {e}") from e


class OpenAIProvider(BaseProvider):
    """Any OpenAI-compatible /chat/completions endpoint (OpenRouter, Groq, ...)."""

    type = "openai"

    def __init__(self, base_url, model, api_key=None, context_window=None):
        super().__init__(base_url, model, api_key)
        # advisory size for history trimming; cloud models manage their own ctx
        self.context_window = context_window

    def _to_wire(self, messages: list[dict]) -> list[dict]:
        wire = []
        for m in messages:
            if m["role"] == "assistant":
                wm = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    wm["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                wire.append(wm)
            elif m["role"] == "tool":
                wire.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                })
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        return wire

    def chat(self, messages, tools, _with_usage=True):
        payload = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if _with_usage:
            payload["stream_options"] = {"include_usage": True}

        # index -> partial tool call
        partial: dict[int, dict] = {}
        usage = {}
        start = time.time()
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers(),
            ) as resp:
                if resp.status_code == 400 and _with_usage:
                    # some providers reject stream_options; retry without it
                    resp.read()
                    yield from self.chat(messages, tools, _with_usage=False)
                    return
                if resp.status_code != 200:
                    body = resp.read().decode(errors="replace")
                    raise _classify_error(
                        f"HTTP {resp.status_code}: {body[:500]}",
                        resp.status_code)
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue  # tolerate malformed/truncated stream lines
                    if chunk.get("usage"):
                        usage = {
                            "prompt_tokens": chunk["usage"].get("prompt_tokens", 0),
                            "completion_tokens": chunk["usage"].get("completion_tokens", 0),
                        }
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        yield ("thinking", reasoning)
                    if delta.get("content"):
                        yield ("text", delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = partial.setdefault(
                            idx, {"id": None, "name": "", "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach {self.base_url}: {e}",
                                retryable=True) from e

        if partial:
            tool_calls = []
            for idx in sorted(partial):
                slot = partial[idx]
                try:
                    args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except json.JSONDecodeError:
                    # let the tool layer report the parse failure to the model
                    args = {"_malformed_json": slot["arguments"]}
                if not isinstance(args, dict):
                    args = {"_malformed_json": slot["arguments"]}
                tool_calls.append({
                    "id": slot["id"] or new_call_id(),
                    "name": slot["name"],
                    "arguments": args,
                })
            yield ("tool_calls", tool_calls)
        usage["seconds"] = time.time() - start
        yield ("usage", usage)

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get(f"{self.base_url}/models", headers=self._headers())
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot list models: {e}") from e


def make_provider(profile: dict, model_override: str | None = None) -> BaseProvider:
    ptype = profile.get("type", "ollama")
    model = model_override or profile["model"]
    api_key = None
    if profile.get("api_key_env"):
        api_key = os.environ.get(profile["api_key_env"])
        if not api_key and ptype == "openai":
            raise ProviderError(
                f"Environment variable {profile['api_key_env']} is not set "
                f"(needed for this profile)."
            )
    elif profile.get("api_key"):
        api_key = profile["api_key"]

    if ptype == "ollama":
        return OllamaProvider(
            profile["base_url"], model, api_key=api_key,
            num_ctx=profile.get("num_ctx"),
        )
    if ptype == "openai":
        return OpenAIProvider(profile["base_url"], model, api_key=api_key,
                              context_window=profile.get("context_window"))
    raise ProviderError(f"Unknown provider type '{ptype}'")


def ensure_ollama_running(base_url: str, auto_start: bool = True) -> bool:
    """If the profile points at a local Ollama, start the daemon if needed."""
    if not is_local_url(base_url):
        return True

    def _is_ollama() -> bool:
        try:
            resp = httpx.get(base_url, timeout=2.0)
            return resp.status_code == 200 and "ollama" in resp.text.lower()
        except httpx.HTTPError:
            return False

    if _is_ollama():
        return True
    if not auto_start:
        return False
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return False
    env = dict(os.environ, OLLAMA_FLASH_ATTENTION="1")
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, env=env,
    )
    for _ in range(30):
        time.sleep(0.5)
        if _is_ollama():
            return True
    return False
