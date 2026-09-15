"""SWE-bench Multilingual task lifecycle."""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

# Remove tags and unreachable future history without touching the working tree:
# multilingual images may contain required, uncommitted build-time edits.
_GIT_CLEAN_HISTORY = "\n".join(
    [
        "git tag -d $(git tag -l) 2>/dev/null || true",
        "git reflog expire --expire=now --all || true",
        "git gc --prune=now || true",
    ]
)


class SWEBenchMultilingualTaskConfig(TaskConfig):
    name: str = "swe_bench_multilingual"
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )
    eval_timeout: float = Field(
        default=600.0,
        gt=0,
        description="Maximum number of seconds allowed for multilingual evaluation.",
    )


@register_task("swe_bench_multilingual")
class SWEBenchMultilingualTask(Task):
    name = "swe_bench_multilingual"
    config_model = SWEBenchMultilingualTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEBenchMultilingualTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata
        instance_id = sample.get("instance_id", "?")
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            "starting swe_bench_multilingual task "
            f"(instance_id={instance_id}, run_oracle_solution={cfg.run_oracle_solution})\n"
            f"task config: {json.dumps(task_config_dump, indent=2)}"
        )

        async with self.build_sandbox() as sandbox:
            # Preserve build-time worktree edits while removing history that could
            # expose the upstream solution to the agent.
            await sandbox.exec_shell(_GIT_CLEAN_HISTORY, workdir="/testbed")

            if cfg.run_oracle_solution:
                logger.info("applying gold patch to /testbed")
                patch_path = "/tmp/gold_patch.patch"
                await sandbox.write_file(patch_path, sample["patch"])
                apply_result = await sandbox.exec(
                    ["git", "apply", "--whitespace=fix", patch_path],
                    workdir="/testbed",
                )
                if apply_result.exit_code != 0:
                    error = apply_result.stderr.strip() or apply_result.stdout.strip()
                    raise RuntimeError(f"failed to apply gold patch for {instance_id}: {error}")
                finished = True
            else:
                agent = self.build_agent()
                agent_result = await agent.run(
                    sandbox=sandbox,
                    messages=cfg.prompt,
                    workdir="/testbed",
                )
                finished = agent_result.finished

            try:
                from .reward import compute_reward

                result = await compute_reward(
                    sample,
                    sandbox,
                    eval_timeout=cfg.eval_timeout,
                )
            except Exception:
                logger.exception("scoring failed for instance_id=%s", instance_id)
                raise

            logger.info("task done: resolved=%s", result["resolved"])
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
