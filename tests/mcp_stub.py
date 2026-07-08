"""A minimal MCP server for tests: one 'echo' tool over stdio.

With no argument it is a normal echo server. The other modes (argv[1])
exercise failure paths the client must survive:
  die-after-init   answer `initialize`, then exit (dies mid-handshake)
  silent           never respond at all (client must bound the handshake)
  spawn-child F     normal server that also spawns a child process, writing
                    the child's pid to file F (to check group-kill on close)
"""

import json
import sys
import time


def _serve(child_pid_file=None):
    if child_pid_file:
        import subprocess
        child = subprocess.Popen(["sleep", "300"])
        with open(child_pid_file, "w") as f:
            f.write(str(child.pid))
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


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "silent":
        time.sleep(60)
        return
    if mode == "die-after-init":
        line = sys.stdin.readline()
        msg = json.loads(line)
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                          "result": {"protocolVersion": "2025-03-26",
                                     "capabilities": {}}}), flush=True)
        return  # exit before tools/list — server dies mid-handshake
    if mode == "spawn-child":
        _serve(child_pid_file=sys.argv[2])
        return
    _serve()


if __name__ == "__main__":
    main()
