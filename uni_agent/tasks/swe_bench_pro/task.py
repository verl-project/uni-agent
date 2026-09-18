"""SWE-bench Pro task lifecycle."""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ...sandbox import SandboxBackend
from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

REPO_WORKDIR = "/app"
_GIT_PREPARE_TIMEOUT_SECONDS = 600
_GIT_PREPARE_SCRIPT = """
set -euo pipefail

test -n "${BASE_COMMIT:-}"
base_commit="$(git rev-parse --verify "${BASE_COMMIT}^{commit}")"

git restore .
git reset --hard "$base_commit"
git clean -fdq
git checkout --detach "$base_commit"

for remote in $(git remote); do
    git remote remove "$remote"
done
git for-each-ref --format='delete %(refname)' | git update-ref --stdin

rm -f .git/FETCH_HEAD .git/ORIG_HEAD
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now
git prune --expire=now

test "$(git rev-parse HEAD)" = "$base_commit"
test -z "$(git remote)"
test -z "$(git for-each-ref --format='%(refname)')"
""".strip()


async def _prepare_repository(sandbox: SandboxBackend, base_commit: str) -> None:
    """Reset the worktree and make commits newer than ``base_commit`` unreachable."""
    result = await sandbox.exec_shell(
        _GIT_PREPARE_SCRIPT,
        workdir=REPO_WORKDIR,
        env={"BASE_COMMIT": base_commit},
        timeout=_GIT_PREPARE_TIMEOUT_SECONDS,
    )
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
        raise RuntimeError(f"failed to prepare SWE-bench Pro repository at base_commit={base_commit}: {detail}")


class SWEBenchProTaskConfig(TaskConfig):
    name: str = "swe_bench_pro"
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )
    eval_timeout: float = Field(
        default=3600.0,
        gt=0,
        description="Maximum number of seconds allowed for SWE-bench Pro evaluation.",
    )


@register_task("swe_bench_pro")
class SWEBenchProTask(Task):
    name = "swe_bench_pro"
    config_model = SWEBenchProTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWEBenchProTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata
        instance_id = sample.get("instance_id", "?")
        base_commit = sample.get("base_commit")
        if not isinstance(base_commit, str) or not base_commit.strip():
            raise ValueError(f"missing base_commit for SWE-bench Pro instance_id={instance_id}")
        base_commit = base_commit.strip()

        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            "starting swe_bench_pro task (instance_id=%s, run_oracle_solution=%s)\ntask config: %s",
            instance_id,
            cfg.run_oracle_solution,
            json.dumps(task_config_dump, indent=2),
        )

        from .reward import compute_reward

        async with self.build_sandbox() as sandbox:
            logger.info("resetting repository and pruning future git history at %s", base_commit)
            await _prepare_repository(sandbox, base_commit)

            if cfg.run_oracle_solution:
                patch = sample.get("patch")
                if not isinstance(patch, str) or not patch.strip():
                    raise ValueError(f"missing gold patch for SWE-bench Pro instance_id={instance_id}")
                patch_path = "/tmp/swe_bench_pro_gold.patch"
                await sandbox.write_file(patch_path, patch)
                apply_result = await sandbox.exec(
                    ["git", "apply", "--whitespace=fix", patch_path],
                    workdir=REPO_WORKDIR,
                )
                if apply_result.exit_code != 0:
                    detail = apply_result.stderr.strip() or apply_result.stdout.strip()
                    raise RuntimeError(
                        f"failed to apply gold patch for SWE-bench Pro instance_id={instance_id}: "
                        f"{detail or f'exit code {apply_result.exit_code}'}"
                    )
                finished = True
            else:
                agent = self.build_agent()
                agent_result = await agent.run(
                    sandbox=sandbox,
                    messages=cfg.prompt,
                    workdir=REPO_WORKDIR,
                )
                finished = agent_result.finished

            result = await compute_reward(
                sample,
                sandbox,
                eval_timeout=cfg.eval_timeout,
            )

            logger.info("task done: resolved=%s", result["resolved"])
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
