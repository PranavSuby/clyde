"""Minimal MCP client (stdio transport, newline-delimited JSON-RPC).

Config:
    "mcp_servers": {
        "weather": {"command": ["npx", "-y", "@example/weather-mcp"]}
    }

Each server's tools appear to the model as mcp__<server>__<tool>.
"""

import json
import queue
import subprocess
import threading

PROTOCOL_VERSION = "2025-03-26"


class MCPError(Exception):
    pass


class MCPServer:
    def __init__(self, name: str, command: list[str], timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        self._id = 0
        self._responses: "queue.Queue[dict]" = queue.Queue()
        try:
            self.proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                start_new_session=True,
            )
        except OSError as e:
            raise MCPError(f"cannot start MCP server '{name}': {e}") from e
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.tools = self._handshake()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg:  # response (not a notification)
                self._responses.put(msg)

    def _request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method,
               "params": params or {}}
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"{self.name}: server pipe closed: {e}") from e
        deadline_q = self._responses
        try:
            while True:
                msg = deadline_q.get(timeout=self.timeout)
                if msg.get("id") == self._id:
                    if "error" in msg:
                        raise MCPError(f"{self.name}.{method}: "
                                       f"{msg['error'].get('message', msg['error'])}")
                    return msg.get("result", {})
                # response to an older request; drop it
        except queue.Empty:
            raise MCPError(f"{self.name}.{method}: timed out after "
                           f"{self.timeout}s") from None

    def _notify(self, method: str):
        note = {"jsonrpc": "2.0", "method": method}
        try:
            self.proc.stdin.write(json.dumps(note) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"{self.name}: server pipe closed: {e}") from e

    def _handshake(self) -> list[dict]:
        # a dead server must not stall startup for the full call timeout
        call_timeout, self.timeout = self.timeout, min(self.timeout, 10.0)
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "clyde", "version": "0.2"},
            })
            self._notify("notifications/initialized")
            result = self._request("tools/list")
        finally:
            self.timeout = call_timeout
        return result.get("tools", [])

    def call(self, tool: str, args: dict) -> str:
        result = self._request("tools/call", {"name": tool, "arguments": args})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type')} content]")
        text = "\n".join(parts) or "(empty result)"
        if result.get("isError"):
            text = "Error: " + text
        return text

    def close(self):
        import os
        import signal
        if self.proc.poll() is not None:
            return
        # the server may have spawned children (npx -> node): signal the
        # whole process group, and reap so no zombie lingers
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def load_servers(cfg: dict, console=None) -> dict[str, MCPServer]:
    """Start configured MCP servers; failures are reported, not fatal."""
    servers = {}
    for name, spec in (cfg.get("mcp_servers") or {}).items():
        try:
            servers[name] = MCPServer(name, spec["command"],
                                      timeout=spec.get("timeout", 30.0))
            if console:
                console.print(f"[dim]mcp: {name} up "
                              f"({len(servers[name].tools)} tools)[/dim]")
        except (MCPError, KeyError, OSError) as e:
            if console:
                console.print(f"[red]mcp: {name} failed: {e}[/red]")
    return servers


def tool_schemas(servers: dict[str, MCPServer]) -> list[dict]:
    """MCP tools in OpenAI function format, namespaced mcp__server__tool."""
    out = []
    for sname, server in servers.items():
        for t in server.tools:
            out.append({
                "type": "function",
                "function": {
                    "name": f"mcp__{sname}__{t['name']}",
                    "description": (t.get("description") or "")[:1000],
                    "parameters": t.get("inputSchema")
                    or {"type": "object", "properties": {}},
                },
            })
    return out
