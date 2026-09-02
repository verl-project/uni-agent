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

from ...sandbox import SandboxBackend
from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


_GIT_CLEAN_HISTORY = """
set -euo pipefail

test -n "${BASE_COMMIT:-}"
base_commit="$(git rev-parse --verify "${BASE_COMMIT}^{commit}")"

git reset --hard "$base_commit"
git clean -ffd
git checkout --detach "$base_commit"

for remote in $(git remote); do
    git remote remove "$remote"
done
git for-each-ref --format='delete %(refname)' | git update-ref --stdin

git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now
git prune --expire=now

test "$(git rev-parse HEAD)" = "$base_commit"
test -z "$(git remote)"
test -z "$(git for-each-ref --format='%(refname)')"
test "$(git rev-list --count --all --reflog --not HEAD)" = "0"
test -z "$(git status --porcelain --untracked-files=all)"
""".strip()

_GIT_CLEAN_TIMEOUT_SECONDS = 600


async def _clean_git_history(sandbox: SandboxBackend, base_commit: str) -> None:
    result = await sandbox.exec_shell(
        _GIT_CLEAN_HISTORY,
        workdir="/testbed",
        env={"BASE_COMMIT": base_commit},
        timeout=_GIT_CLEAN_TIMEOUT_SECONDS,
    )
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"failed to sanitize git history at base_commit={base_commit}: {detail}")


class SWEREBenchTaskConfig(TaskConfig):
    name: str = "swe_rebench"
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )
    eval_timeout: float = Field(
        default=600.0,
        description="Per-sample reward-eval timeout (s) inside the sandbox.",
    )


@register_task("swe_rebench")
class SWEREBenchTask(Task):
    name = "swe_rebench"
    config_model = SWEREBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEREBenchTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata  # the dataset sample is carried on the task config

        instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            f"starting swe_rebench task (instance_id={instance_id}, run_oracle_solution={cfg.run_oracle_solution})\n"
            f"task config: {json.dumps(task_config_dump, indent=2)}"
        )
        base_commit = sample.get("base_commit") if isinstance(sample, dict) else None
        if not isinstance(base_commit, str) or not base_commit.strip():
            raise ValueError(f"missing base_commit for swe_rebench instance_id={instance_id}")

        async with self.build_sandbox() as sandbox:
            # Reset first, then remove every other local path to future commits.
            await _clean_git_history(sandbox, base_commit.strip())

            if cfg.run_oracle_solution:
                logger.info("applying gold patch to /testbed")
                await sandbox.write_file("/tmp/gold_patch.patch", sample["patch"])
                await sandbox.exec(["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed")
                finished = True
            else:
                agent = self.build_agent()
                messages = cfg.prompt
                # The endpoint the agent calls lives on cfg.agent.model (the agent validates it).
                agent_result = await agent.run(
                    sandbox=sandbox,
                    messages=messages,
                    workdir="/testbed",
                )
                finished = agent_result.finished

            try:
                from .reward import compute_reward

                result = await compute_reward(sample, sandbox, eval_timeout=cfg.eval_timeout)
            except Exception:
                logger.exception(f"scoring failed for instance_id={instance_id}")
                raise

            logger.info(f"task done: resolved={result['resolved']}")
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
