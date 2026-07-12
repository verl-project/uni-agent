"""SWE-rebench task (native framework loop).

Same shape as :mod:`uni_agent.tasks.swe_bench.run`, with two swe-rebench specifics:
scoring reads the eval config carried on the row (see :mod:`.reward`), and the
future git history is cleaned in-sandbox before the agent runs (this used to be a
data-preprocess ``post_setup_cmd``; owning it here keeps the dataset row declarative).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from pydantic import Field

from ...logging import sample_logging
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
        run_id = str(uuid.uuid4())
        log_dir = os.path.expanduser(cfg.log_dir or f"/tmp/uni_agent_logs/{self.name}")

        # Route this episode's logs (agent, tools, sandbox) to <log_dir>/<run_id>.log.
        async with sample_logging(run_id, Path(log_dir) / f"{run_id}.log"):
            instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
            task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
            logger.info(
                f"starting swe_rebench task {run_id} (instance_id={instance_id}, run_gold_patch={cfg.run_gold_patch})\n"
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

                from .reward import compute_reward

                result = await compute_reward(sample, sandbox)

                logger.info(f"task {run_id} done: resolved={result['resolved']}")
                return TaskResult(
                    reward=result["resolved"],
                    info=result,
                )
