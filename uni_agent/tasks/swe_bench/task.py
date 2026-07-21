"""SWE-bench task: one problem family, solved by whichever agent you configure."""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


class SWEBenchTaskConfig(TaskConfig):
    name: str = "swe_bench"
    run_gold_patch: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )


@register_task("swe_bench")
class SWEBenchTask(Task):
    name = "swe_bench"
    config_model = SWEBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEBenchTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata  # the dataset sample is carried on the task config

        instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            f"starting swe_bench task (instance_id={instance_id}, run_gold_patch={cfg.run_gold_patch})\n"
            f"task config: {json.dumps(task_config_dump, indent=2)}"
        )
        async with self.build_sandbox() as sandbox:
            if cfg.run_gold_patch:
                logger.info("applying gold patch to /testbed")
                await sandbox.write_file("/tmp/gold_patch.patch", sample["patch"])
                await sandbox.exec(["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed")
            else:
                agent = self.build_agent()
                messages = cfg.prompt
                # The endpoint the agent calls lives on cfg.agent.model (the agent validates it).
                await agent.run(sandbox=sandbox, messages=messages)

            from .reward import compute_reward

            result = await compute_reward(sample, sandbox)

            logger.info(f"task done: resolved={result['resolved']}")
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                info=result,
            )
