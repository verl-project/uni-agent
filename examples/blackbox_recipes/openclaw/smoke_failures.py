"""Real CLI negative paths: endpoint rejection and deadline, without a GPU."""
import argparse
import asyncio
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.openclaw.agent import OpenClawAgent, OpenClawConfig
from uni_agent.sandbox.docker import DockerSandbox


async def check(mode, output):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if mode == "timeout":
                time.sleep(15)
                return
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"intentional test rejection","type":"authentication_error"}}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sandbox = DockerSandbox(image="openclaw-recipe-runtime:2026.9.2", pull_policy="never",
                            run_args=["--network", "host"])
    started = time.monotonic()
    try:
        async with sandbox:
            config = OpenClawConfig(model=ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                   model_name="Qwen3.5-9B"), cli_timeout_seconds=5, run_timeout=12)
            result = await OpenClawAgent(config).run(sandbox=sandbox,
                         messages=[{"role": "user", "content": "Write /workspace/answer.txt using exec."}],
                         workdir="/workspace")
    finally:
        server.shutdown()
        server.server_close()
    cleaned = not await sandbox.is_alive()
    record = {"mode": mode, "result": dataclasses.asdict(result), "elapsed_seconds": time.monotonic()-started,
              "sandbox_stopped": cleaned, "passed": result.finished is False and cleaned
              and result.info.get("error_kind") == ("timeout" if mode == "timeout" else "agent_failure")}
    (output / (mode + ".json")).write_text(json.dumps(record, indent=2))
    print(json.dumps({"mode": mode, "passed": record["passed"], "error_kind": result.info.get("error_kind"),
                      "exit_code": result.info.get("exit_code"), "elapsed_seconds": record["elapsed_seconds"]}))
    assert record["passed"], record


async def main(output):
    output.mkdir(parents=True, exist_ok=False)
    for mode in ("rejection", "timeout"):
        await check(mode, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.output))
