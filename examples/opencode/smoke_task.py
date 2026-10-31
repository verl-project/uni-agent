"""A local Task lifecycle demo with real OpenCode and scripted model responses.

Run from the repository root with ``python -m examples.opencode.smoke_task
--opencode-bin /path/to/opencode``. No provider account, model, GPU, Gateway, or
training dependencies are used. The scripted patch is a fixture, not evidence
of model task-solving ability. LocalSandbox runs on the host, not in a container.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeConfig
from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import Task, TaskConfig, TaskResult

Scenario = Literal["patch", "no-edit"]
_PROMPT = "Fix add in calculator.py so it adds both numbers. Finish after editing the file."
_VERIFY = "import calculator; assert calculator.add(2, 3) == 5; assert calculator.add(-4, 1) == -3"


@dataclasses.dataclass
class ScriptedEndpoint:
    base_url: str
    requests: int = 0


@contextmanager
def scripted_endpoint(scenario: Scenario) -> Iterator[ScriptedEndpoint]:
    """Serve a fixed relative-path edit, or a no-edit negative control."""
    endpoint = ScriptedEndpoint(base_url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            endpoint.requests += 1
            has_tool_result = any(message.get("role") == "tool" for message in request["messages"])
            edit = scenario == "patch" and not has_tool_result
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_smoke_patch",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {
                                        "command": "printf 'def add(a, b):\\n    return a + b\\n' > calculator.py",
                                        "description": "Apply the scripted fixture patch in the task workspace",
                                    }
                                ),
                            },
                        }
                    ]
                }
                if edit
                else {"content": "SMOKE_DONE"}
            )
            common = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "scripted-policy",
            }
            chunks = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {
                    **common,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if edit else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    endpoint.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class PatchSmokeTask(Task):
    """Own a temporary workspace, the Agent invocation, verification, and cleanup.

    Kept in examples, without adding a production Task registry entry. This
    fixture is deliberately local-only and emits only a short file-writing tool
    command. It does not address LocalSandbox's process-cancellation limitations.
    """

    async def run(self) -> TaskResult:
        if self.config.sandbox.provider != "local" or not isinstance(self.config.agent, OpenCodeConfig):
            raise ValueError("the smoke Task requires LocalSandbox and OpenCodeConfig")

        with tempfile.TemporaryDirectory(prefix="uni-agent-opencode-smoke-") as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            agent = self.build_agent()
            agent.config = self.config.agent.model_copy(update={"state_root": str(root / "state")})
            async with self.build_sandbox() as sandbox:
                await sandbox.write_file(str(workspace / "calculator.py"), "def add(a, b):\n    return a - b\n")
                agent_result = await agent.run(sandbox=sandbox, messages=self.config.prompt, workdir=str(workspace))
                # Verify the file independently; the assistant's text is not a score.
                # Do not start scoring an unfinished run that might still be writing.
                verification = None
                if agent_result.finished is True:
                    verification = await sandbox.exec(
                        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, '.'); " + _VERIFY],
                        workdir=str(workspace),
                        timeout=10,
                    )
                passed = verification is not None and verification.exit_code == 0
                result = TaskResult(
                    reward=float(passed),
                    accuracy=float(passed),
                    finished=agent_result.finished,
                    extra_info={
                        "backend": "scripted",
                        "real_model": False,
                        "gateway": False,
                        "verifier_exit_code": verification.exit_code if verification is not None else None,
                        "agent_error": agent_result.info.get("error_kind"),
                    },
                )
        result.extra_info["workspace_removed"] = not root.exists()
        return result


async def run_smoke(opencode_bin: str, scenario: Scenario = "patch") -> TaskResult:
    with scripted_endpoint(scenario) as endpoint:
        config = TaskConfig(
            sandbox=SandboxConfig(provider="local"),
            agent=OpenCodeConfig(
                opencode_bin=opencode_bin,
                run_timeout=30,
                model=ModelConfig(base_url=endpoint.base_url, model_name="scripted-policy", api_key="smoke-dummy"),
            ),
            prompt=[{"role": "user", "content": _PROMPT}],
        )
        result = await PatchSmokeTask(config).run()
        result.extra_info.update(scenario=scenario, requests=endpoint.requests)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencode-bin", default=os.environ.get("OPENCODE_BIN"))
    parser.add_argument("--scenario", choices=("patch", "no-edit"), default="patch")
    args = parser.parse_args()
    if not args.opencode_bin:
        parser.error("provide --opencode-bin or explicitly set OPENCODE_BIN to OpenCode 1.19.3")
    result = asyncio.run(run_smoke(args.opencode_bin, args.scenario))
    print(json.dumps(dataclasses.asdict(result), indent=2))
    return 0 if result.finished is True and result.reward == 1.0 and result.extra_info["workspace_removed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
