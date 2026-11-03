"""Opt-in mounted Linux sidecar: isolate CLI loading and preserve tool libraries."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.sandbox.local import LocalSandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


@pytest.mark.parametrize("library_env", ["unset", "empty", "glibc", "custom"])
async def test_sidecar_preserves_task_library_environment(
    tmp_path, monkeypatch, library_env, *, agent_cls=OpenCodeAgent, config_cls=OpenCodeConfig, cli=None
):
    cli = cli or os.environ.get("OPENCODE_SIDECAR_BIN")
    if not cli:
        pytest.skip("set OPENCODE_SIDECAR_BIN to the mounted Linux supported sidecar")
    assert sys.platform == "linux"
    library = tmp_path / "custom libraries"
    library.mkdir()
    source = library / "marker.c"
    source.write_text("int ua_phase2_marker(void) { return 73; }\n")
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library / "libua_phase2.so")], check=True)
    values = {
        "unset": None,
        "empty": "",
        "glibc": "/usr/lib/x86_64-linux-gnu",
        "custom": f"{library}:/usr/lib/x86_64-linux-gnu",
    }
    expected = values[library_env]
    if expected is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", expected)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "check_environment.py"
    script.write_text(
        "import ctypes, json, os, pathlib, platform\n"
        "data = dict(library_path=os.environ.get('LD_LIBRARY_PATH'), libc=platform.libc_ver()[0])\n"
        + ("data['marker'] = ctypes.CDLL('libua_phase2.so').ua_phase2_marker()\n" if library_env == "custom" else "")
        + "pathlib.Path('result.json').write_text(json.dumps(data))\n"
    )
    # The task uses system Python; runner Python needs a separate libpython path.
    command = shlex.join(["/usr/bin/python3", "-I", str(script)])
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            tool = not any(m.get("role") == "tool" for m in body["messages"])
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_env",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": command, "description": "Check task libraries"}),
                            },
                        }
                    ]
                }
                if tool
                else {"content": "ENV_DONE"}
            )
            common = {"id": "chatcmpl-env", "object": "chat.completion.chunk", "created": 0, "model": "scripted"}
            chunks = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {
                    **common,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = config_cls(
            opencode_bin=cli,
            state_root=str(tmp_path / "state"),
            run_timeout=30,
            model=ModelConfig(
                base_url=f"http://127.0.0.1:{server.server_port}/v1", model_name="scripted", api_key="env-dummy"
            ),
        )
        result = await agent_cls(config).run(
            sandbox=LocalSandbox(),
            messages=[{"role": "user", "content": "Run the library check."}],
            workdir=str(workspace),
        )
        assert result.finished is True, result.info
        assert len(requests) == 2
        observed = json.loads((workspace / "result.json").read_text())
        assert observed["library_path"] == expected
        assert observed["libc"] == "glibc"
        if library_env == "custom":
            assert observed["marker"] == 73
        assert not list((tmp_path / "state").iterdir())
        print(
            "sidecar_environment=" + json.dumps({"case": library_env, "observed": observed, "requests": len(requests)})
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
