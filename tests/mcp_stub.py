"""A minimal MCP server for tests: one 'echo' tool over stdio."""

import json
import sys


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method", "")
        if "id" not in msg:
            continue  # notification
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {},
                      "serverInfo": {"name": "stub", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{
                "name": "echo",
                "description": "Echo the input back",
                "inputSchema": {"type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"]},
            }]}
        elif method == "tools/call":
            text = msg["params"]["arguments"].get("text", "")
            result = {"content": [{"type": "text", "text": f"echo: {text}"}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}),
              flush=True)


if __name__ == "__main__":
    main()
