import json

import httpx
import pytest

from clyde.providers import OpenAIProvider, ProviderError


def sse(lines):
    return "\n".join(f"data: {json.dumps(x) if not isinstance(x, str) else x}"
                     for x in lines) + "\n"


def make_provider(handler):
    p = OpenAIProvider("https://api.test/v1", "m", api_key="k",
                       context_window=100000)
    p.client = httpx.Client(transport=httpx.MockTransport(handler))
    return p


def test_sse_text_and_toolcall_accumulation():
    body = sse([
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "ba", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "sh", "arguments": "{\"comm"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "and\": \"ls\"}"}}]}}]},
        {"usage": {"prompt_tokens": 5, "completion_tokens": 7}, "choices": []},
        "[DONE]",
    ])
    p = make_provider(lambda req: httpx.Response(200, text=body))
    events = list(p.chat([{"role": "user", "content": "hi"}], []))
    text = "".join(t for k, t in events if k == "text")
    assert text == "Hello"
    calls = next(pl for k, pl in events if k == "tool_calls")
    assert calls[0]["name"] == "bash"
    assert calls[0]["arguments"] == {"command": "ls"}
    usage = next(pl for k, pl in events if k == "usage")
    assert usage["prompt_tokens"] == 5


def test_malformed_stream_lines_skipped():
    body = ("data: not-json-at-all\n"
            + sse([{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"]))
    p = make_provider(lambda req: httpx.Response(200, text=body))
    events = list(p.chat([{"role": "user", "content": "hi"}], []))
    assert "".join(t for k, t in events if k == "text") == "ok"


def test_stream_options_retry_on_400():
    seen = []

    def handler(req):
        payload = json.loads(req.content)
        seen.append("stream_options" in payload)
        if "stream_options" in payload:
            return httpx.Response(400, text="stream_options unsupported")
        return httpx.Response(
            200, text=sse([{"choices": [{"delta": {"content": "hi"}}]}, "[DONE]"]))

    p = make_provider(handler)
    events = list(p.chat([{"role": "user", "content": "x"}], []))
    assert seen == [True, False]
    assert "".join(t for k, t in events if k == "text") == "hi"


def test_http_error_raises_provider_error():
    p = make_provider(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(ProviderError):
        list(p.chat([{"role": "user", "content": "x"}], []))


def test_ollama_wire_helpers():
    from clyde import ollama_wire
    # string-encoded arguments are tolerated and decoded
    calls = ollama_wire.parse_tool_calls(
        {"tool_calls": [{"function": {"name": "bash",
                                      "arguments": '{"command": "ls"}'}}]})
    assert calls[0]["name"] == "bash"
    assert calls[0]["arguments"] == {"command": "ls"}
    assert calls[0]["id"].startswith("call_")
    # bad JSON arguments degrade to an empty dict, never raise
    bad = ollama_wire.parse_tool_calls(
        {"tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]})
    assert bad[0]["arguments"] == {}
    assert ollama_wire.parse_usage(
        {"prompt_eval_count": 12, "eval_count": 3}) == {
        "prompt_tokens": 12, "completion_tokens": 3}
    assert ollama_wire.is_local_url("http://localhost:11434")
    assert not ollama_wire.is_local_url("https://api.openrouter.ai")
