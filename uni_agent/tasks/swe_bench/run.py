"""SWE-bench task: one problem family, solved by whichever agent you configure."""

from __future__ import annotations

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task


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
        sample = cfg.metadata  # the dataset sample now lives on the config

        async with self.build_sandbox() as sandbox:
            # run gold patch or agent
            if cfg.run_gold_patch:
                await sandbox.write_file(f"/tmp/gold_patch.patch", sample["patch"])
                await sandbox.exec(["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed")
            else:
                agent = self.build_agent()
                if cfg.model.base_url is None:
                    raise ValueError("swe_bench: cfg.model.base_url is not set (the endpoint the agent calls)")
                messages = [{"role": "user", "content": sample.get("problem_statement", "")}]

                result = await agent.run(
                    sandbox=sandbox,
                    base_url=cfg.model.base_url,
                    api_key=cfg.model.api_key,
                    messages=messages,
                )

            # compute reward
            from .reward import compute_reward
            result = await compute_reward(sample, sandbox)

            return TaskResult(
                reward=result["resolved"],
                info=result,
            )
