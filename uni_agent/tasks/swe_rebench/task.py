"""SWE-rebench task (native framework loop).

Same shape as :mod:`uni_agent.tasks.swe_bench.task`, with two swe-rebench specifics:
scoring reads the eval config carried on the row (see :mod:`.reward`), and the
future git history is cleaned in-sandbox before the agent runs (this used to be a
data-preprocess ``post_setup_cmd``; owning it here keeps the dataset row declarative).
"""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

# Remove the repo's own tags + unreachable history so a later "future" tag/commit
# can't leak the fix to the agent. Best-effort (`|| true`); runs once in /testbed.
_GIT_CLEAN_HISTORY = " && ".join(
    [
        "git tag -d $(git tag -l) || true",
        "git reflog expire --expire=now --all || true",
        "git gc --prune=now || true",
    ]
)


class SWEREBenchTaskConfig(TaskConfig):
    name: str = "swe_rebench"
    run_gold_patch: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )


@register_task("swe_rebench")
class SWEREBenchTask(Task):
    name = "swe_rebench"
    config_model = SWEREBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEREBenchTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata  # the dataset sample is carried on the task config

        # Route this episode's logs (agent, tools, sandbox) to the run's log stream.
        async with self.episode_logging():
            instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
            task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
            logger.info(
                f"starting swe_rebench task (instance_id={instance_id}, run_gold_patch={cfg.run_gold_patch})\n"
                f"task config: {json.dumps(task_config_dump, indent=2)}"
            )
            async with self.build_sandbox() as sandbox:
                # Clean future history before anything reads the repo.
                await sandbox.exec_shell(_GIT_CLEAN_HISTORY, workdir="/testbed")

                if cfg.run_gold_patch:
                    logger.info("applying gold patch to /testbed")
                    await sandbox.write_file("/tmp/gold_patch.patch", sample["patch"])
                    await sandbox.exec(
                        ["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed"
                    )
                else:
                    agent = self.build_agent()
                    messages = cfg.prompt
                    # The endpoint the agent calls lives on cfg.agent.model (the agent validates it).
                    await agent.run(sandbox=sandbox, messages=messages)

                try:
                    from .reward import compute_reward

                    result = await compute_reward(sample, sandbox)
                except Exception:
                    # Scoring runs inside episode_logging; without surfacing it here the
                    # traceback escapes to the worker log and the eval silently "vanishes".
                    logger.exception(f"scoring failed for instance_id={instance_id}")
                    raise

                logger.info(f"task done: resolved={result['resolved']}")
                return TaskResult(
                    reward=float(result["resolved"]),
                    accuracy=float(result["resolved"]),
                    info=result,
                )
