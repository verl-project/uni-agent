"""Check a stdio model wrapper using a local Chat Completions tool-call stub."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    workdir = Path("/testbed")
    workdir.mkdir(exist_ok=True)
    marker = workdir / f".sidecar-check-{uuid.uuid4().hex}"
    state = {"calls": 0, "tool_result": False}

    class ModelStub(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tool_result = any(message.get("role") == "tool" for message in request.get("messages", []))
            if tool_result:
                message = {"role": "assistant", "content": "sidecar-ok"}
                finish = "stop"
                state["tool_result"] = True
            else:
                tools = {tool.get("function", {}).get("name") for tool in request.get("tools", [])}
                command = f"printf sidecar-ok > {shlex.quote(str(marker))}"
                if "bash" in tools:
                    name, arguments = "bash", {"command": command, "description": "Write the check marker"}
                elif "exec_command" in tools:
                    name, arguments = "exec_command", {"cmd": command}
                elif "shell_command" in tools:
                    name, arguments = "shell_command", {"command": command}
                else:
                    self.send_error(400, "The check requires a bash, exec_command or shell_command tool")
                    return
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "check-call",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
                finish = "tool_calls"
            state["calls"] += 1
            base = {"id": f"chatcmpl-check-{state['calls']}", "created": 0, "model": "default"}
            if request.get("stream"):
                delta = dict(message)
                delta.pop("role", None)
                if "tool_calls" in delta:
                    delta["tool_calls"] = [{"index": 0, **delta["tool_calls"][0]}]
                events = [
                    {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", **delta},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish,
                            }
                        ],
                    },
                ]
                body = ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        **base,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish,
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    env = {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "PROXY"))
    }
    env.update(NO_PROXY="*", no_proxy="*")
    probe = subprocess.run(case["probe_argv"], capture_output=True, env=env, timeout=30)
    if probe.returncode:
        raise RuntimeError(f"Dependency probe failed with exit code {probe.returncode}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = {
        "messages": [{"role": "user", "content": "Execute the requested shell command, then finish."}],
        "model": {
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "api_key": "local-check",
            "model_name": "default",
        },
        "workdir": str(workdir),
    }
    try:
        result = subprocess.run(
            case["entrypoint"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
            cwd=workdir,
            env=env,
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(f"Model wrapper exited {result.returncode}: {result.stderr[-1500:]}")
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get("finished") is not True:
            raise RuntimeError("Wrapper did not report finished=true")
        if not isinstance(response.get("output"), dict):
            raise RuntimeError("Wrapper did not return an output object")
        if not marker.is_file() or marker.read_text() != "sidecar-ok" or not state["tool_result"]:
            raise RuntimeError("The model-requested shell command and tool result were not both observed")
        print(
            json.dumps(
                {
                    "recipe": case["recipe"],
                    "model_calls": state["calls"],
                    "tool_execution": "passed",
                    "wrapper_result": "passed",
                    "result": "passed",
                }
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
