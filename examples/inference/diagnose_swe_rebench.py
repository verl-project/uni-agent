"""Record single-instance Docker gold/no-patch controls without a model or Ray.

Uses the normal SWEREBenchTask.run and reward implementation. The no-patch
control substitutes an agent that makes no edits. Each control gets a new
container with the original instance installation, without dependency overrides.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import subprocess
from pathlib import Path

from datasets import load_dataset

from uni_agent.agents import Agent, AgentConfig, AgentResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.tasks.swe_rebench.task import SWEREBenchTask, SWEREBenchTaskConfig


class NoOpAgent(Agent):
    async def run(self, **kwargs):
        return AgentResult(finished=True)


class RecordingDockerSandbox(DockerSandbox):
    def __init__(self, *, output: Path, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.sequence = 0

    async def start(self):
        await super().start()
        # Read the original image state before Task resets the repository.
        await self.exec_shell(
            "ls -ld /opt/conda /opt/miniconda3; git rev-parse HEAD; "
            "for root in /opt/miniconda3 /opt/conda; do "
            'py="$root/envs/testbed/bin/python"; if [ -x "$py" ]; then '
            "\"$py\" -c 'import sys; print(sys.executable, sys.version)'; "
            '"$py" -m pip list --format=json; fi; done',
            workdir="/testbed",
            timeout=60,
        )

    async def _exec(self, argv, **kwargs):
        self.sequence += 1
        prefix = self.output / f"command-{self.sequence:03}"
        prefix.with_suffix(".json").write_text(json.dumps({"argv": argv, **kwargs}, indent=2) + "\n")
        result = await super()._exec(argv, **kwargs)
        prefix.with_suffix(".stdout").write_text(result.stdout)
        prefix.with_suffix(".stderr").write_text(result.stderr)
        prefix.with_suffix(".exit").write_text(f"{result.exit_code}\n")
        if any(arg.startswith("bash /tmp/eval_script_") for arg in argv):
            (self.output / "eval-output.log").write_text(result.stdout)
        return result

    async def write_file(self, remote_path, content):
        if str(remote_path).startswith("/tmp/eval_script_"):
            (self.output / "eval-script.sh").write_text(content)
        return await super().write_file(remote_path, content)


class DiagnosticTask(SWEREBenchTask):
    def __init__(self, config, sandbox):
        super().__init__(config)
        self.sandbox = sandbox

    def build_sandbox(self):
        return self.sandbox

    def build_agent(self):
        return NoOpAgent(AgentConfig())


async def run_controls(args, metadata):
    summary = {}
    for mode in ("gold", "no-patch"):
        output = args.output_dir / mode
        output.mkdir()
        run_args = ["--platform", args.platform, "--cpus", "2", "--memory", "4g"]
        config = SWEREBenchTaskConfig(
            metadata=metadata,
            run_oracle_solution=mode == "gold",
            sandbox={
                "provider": "docker",
                "image": args.image,
                "runtime_timeout": 1800,
                "sandbox_kwargs": {"pull_policy": "never", "run_args": run_args},
            },
        )
        (output / "task.json").write_text(config.model_dump_json(indent=2) + "\n")
        sandbox = RecordingDockerSandbox(
            output=output,
            image=config.sandbox.image,
            runtime_timeout=config.sandbox.runtime_timeout,
            **config.sandbox.sandbox_kwargs,
        )
        try:
            result = await DiagnosticTask(config, sandbox).run()
        except Exception as exc:
            (output / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            raise
        payload = dataclasses.asdict(result)
        (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        summary[mode] = payload
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({mode: result["reward"] for mode, result in summary.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="Uni-Agent preprocessed SWE-rebench parquet")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--image", required=True, help="Locally pulled image pinned with @sha256:...")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing evidence is not overwritten"
    )
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("--image must be pinned by digest")
    dataset = load_dataset("parquet", data_files=str(args.data_path), split="train")
    matches = []
    for row in dataset:
        task = row["extra_info"]["tools_kwargs"]["task"]
        if task["name"] == "swe_rebench" and task["metadata"]["instance_id"] == args.instance_id:
            matches.append(task["metadata"])
    if len(matches) != 1:
        parser.error(f"expected exactly one matching instance, found {len(matches)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, command in (
        ("docker-version.txt", ["docker", "version"]),
        ("image-inspect.json", ["docker", "image", "inspect", args.image]),
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (args.output_dir / filename).write_text(result.stdout)
    asyncio.run(run_controls(args, matches[0]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
