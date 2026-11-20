"""R2E-Gym task for the unified Task / Agent / Sandbox runtime."""

from __future__ import annotations

import json
import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


_SETUP_SCRIPT = r"""
set -euo pipefail

mkdir -p /root/.local/bin /root/.r2e_eval
if [ -d /testbed/.venv ]; then
    ln -sfn /testbed/.venv /root/.venv
    if [ -x /testbed/.venv/bin/python ]; then
        ln -sfn /testbed/.venv/bin/python /root/.local/bin/python
        ln -sfn /testbed/.venv/bin/python /root/.local/bin/python3
    fi
    find /testbed/.venv/bin -maxdepth 1 -type f -executable -exec ln -sf {} /root/.local/bin/ \; 2>/dev/null || true
fi

if [ ! -f /root/.r2e_eval/run_tests.sh ]; then
    if [ -f /testbed/run_tests.sh ]; then
        cp -a /testbed/run_tests.sh /root/.r2e_eval/run_tests.sh
    elif [ -f /root/run_tests.sh ]; then
        cp -a /root/run_tests.sh /root/.r2e_eval/run_tests.sh
    elif [ -f /run_tests.sh ]; then
        cp -a /run_tests.sh /root/.r2e_eval/run_tests.sh
    else
        echo "R2E-Gym image is missing run_tests.sh" >&2
        exit 1
    fi
fi
chmod +x /root/.r2e_eval/run_tests.sh

if [ ! -d /root/.r2e_eval/r2e_tests ]; then
    if [ -d /testbed/r2e_tests ] && [ ! -L /testbed/r2e_tests ]; then
        cp -a /testbed/r2e_tests /root/.r2e_eval/r2e_tests
    elif [ -d /root/r2e_tests ] && [ ! -L /root/r2e_tests ]; then
        cp -a /root/r2e_tests /root/.r2e_eval/r2e_tests
    elif [ -d /r2e_tests ] && [ ! -L /r2e_tests ]; then
        cp -a /r2e_tests /root/.r2e_eval/r2e_tests
    else
        echo "R2E-Gym image is missing r2e_tests" >&2
        exit 1
    fi
fi

# Keep held-out evaluation scripts and tests hidden from the agent workspace.
rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh
find /testbed /root/.r2e_eval/r2e_tests -name '*.pyc' -delete
find /testbed /root/.r2e_eval/r2e_tests -name '__pycache__' -type d -prune -exec rm -rf {} +
git config --global --add safe.directory /testbed 2>/dev/null || true
git -C /testbed reset --hard 2>/dev/null || true
""".strip()


class R2EGymTaskConfig(TaskConfig):
    name: str = "r2e_gym"
    run_oracle_solution: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )
    setup_timeout: float = Field(default=300.0, description="R2E image normalization timeout in seconds.")
    eval_timeout: float = Field(default=300.0, description="R2E test timeout in seconds.")


@register_task("r2e_gym")
class R2EGymTask(Task):
    name = "r2e_gym"
    config_model = R2EGymTaskConfig

    async def run(self) -> TaskResult:
        cfg: R2EGymTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata
        if cfg.sandbox.image is None and isinstance(sample, dict) and sample.get("docker_image"):
            cfg.sandbox.image = str(sample["docker_image"])
        instance_id = sample.get("instance_id", "?") if isinstance(sample, dict) else "?"
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            f"starting r2e_gym task (instance_id={instance_id}, run_oracle_solution={cfg.run_oracle_solution})\n"
            f"task config: {json.dumps(task_config_dump, indent=2)}"
        )

        async with self.build_sandbox() as sandbox:
            setup = await sandbox.exec_shell(_SETUP_SCRIPT, workdir="/testbed", timeout=cfg.setup_timeout)
            if setup.exit_code != 0:
                detail = setup.stderr.strip() or setup.stdout.strip() or f"exit code {setup.exit_code}"
                raise RuntimeError(f"failed to prepare R2E-Gym sandbox for {instance_id}: {detail}")

            agent_error = None
            if cfg.run_oracle_solution:
                logger.info("applying R2E-Gym gold patch to /testbed")
                await sandbox.write_file("/tmp/gold_patch.patch", sample["patch"])
                applied = await sandbox.exec(
                    ["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"],
                    workdir="/testbed",
                )
                if applied.exit_code != 0:
                    detail = applied.stderr.strip() or applied.stdout.strip() or f"exit code {applied.exit_code}"
                    raise RuntimeError(f"failed to apply R2E-Gym gold patch for {instance_id}: {detail}")
                finished = True
            else:
                agent = self.build_agent()
                agent_result = await agent.run(sandbox=sandbox, messages=cfg.prompt, workdir="/testbed")
                finished = agent_result.finished
                agent_error = agent_result.info.get("error")

            from .reward import compute_reward

            result = await compute_reward(sample, sandbox, eval_timeout=cfg.eval_timeout)
            if isinstance(agent_error, str) and agent_error.strip():
                result = {**result, "agent_error": agent_error}
            logger.info(f"R2E-Gym task done: resolved={result['resolved']}")
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
