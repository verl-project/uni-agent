"""Full common Task -> OpenClaw -> late verifier integration, scripted model only."""
import asyncio
import base64
import dataclasses
import json
from pathlib import Path
import sys

from uni_agent.tasks.registry import get_task
from dataset import task_payload
from smoke_standalone import create_mock_server
from test_verifier import CORRECT, WRONG


async def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for label, solution, expected in [("correct", CORRECT, 1.0), ("wrong", WRONG, 0.0)]:
        encoded = base64.b64encode(solution.encode()).decode()
        command = f"printf %s {encoded} | base64 -d > /workspace/solution.py"
        server, requests = create_mock_server(command)
        try:
            data = task_payload()
            data["agent"]["artifact_dir"] = str(output / label / "trajectories")
            data["agent"]["model"] = {"base_url": f"http://127.0.0.1:{server.server_port}/v1",
                                       "model_name": "Qwen3.5-9B"}
            result = await get_task(data).run()
            report = dataclasses.asdict(result)
            (output / (label + ".json")).write_text(json.dumps(report, indent=2))
            assert result.reward == expected, report
            assert result.finished is True, report
            assert result.extra_info["agent"]["trajectory_audit"]["verified"] is True
            path = Path(result.extra_info["agent"]["trajectory_path"])
            assert path.is_file()
            results.append({"case": label, "reward": result.reward, "finished": result.finished,
                            "single_trajectory": True, "request_count": len(requests)})
        finally:
            server.shutdown()
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
