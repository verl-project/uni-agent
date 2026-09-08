"""Real host Agent + common DockerSandbox smoke; mock LLM, not Qwen validation."""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
import subprocess

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.openclaw.agent import OpenClawAgent, OpenClawConfig
from uni_agent.sandbox.docker import DockerSandbox
from smoke_standalone import create_mock_server


async def run(output: Path, tool_mount: str | None = None):
    output.mkdir(parents=True, exist_ok=False)
    server, requests = create_mock_server()
    run_args = ["--network", "host", "--env", "PATH=/usr/local/bin:/usr/bin:/bin"]
    if tool_mount:
        run_args += ["--mount", f"type=bind,src={Path(tool_mount).resolve()},dst=/opt/openclaw,readonly"]
    sandbox = DockerSandbox(image="openclaw-recipe-runtime:2026.9.2", pull_policy="never",
                            run_args=run_args)
    result = None
    try:
        async with sandbox:
            config = OpenClawConfig(
                model=ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model_name="Qwen3.5-9B"),
                cli_timeout_seconds=90, run_timeout=120, artifact_dir=str(output / "trajectories"),
            )
            result = await OpenClawAgent(config).run(sandbox=sandbox,
                messages=[{"role": "user", "content": "Use exec to calculate 6*7 and write /workspace/answer.txt. Reply DONE."}],
                workdir="/workspace")
            answer = await sandbox.exec(["cat", "/workspace/answer.txt"])
            subprocess.run(["docker", "cp", f"{sandbox._require_container()}:{result.info['state_dir']}",
                            str(output / "state")], check=True, capture_output=True)
            report = {"test_kind": "host_agent_real_openclaw_mock_llm", "result": dataclasses.asdict(result),
                      "answer": answer.stdout.strip(), "request_count": len(requests)}
            report["passed"] = result.finished is True and answer.exit_code == 0 and answer.stdout.strip() == "42"
            (output / "report.json").write_text(json.dumps(report, indent=2))
            (output / "requests.json").write_text(json.dumps(requests, indent=2))
    finally:
        server.shutdown()
    cleaned = not await sandbox.is_alive()
    (output / "cleanup.json").write_text(json.dumps({"sandbox_stopped": cleaned}))
    print(json.dumps({"passed": report["passed"], "sandbox_stopped": cleaned,
                      "trajectory_audit": result.info.get("trajectory_audit"), "output": str(output)}))
    return report["passed"] and cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--tool-mount")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output, args.tool_mount)) else 1)
