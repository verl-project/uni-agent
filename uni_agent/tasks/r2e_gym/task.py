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

mkdir -p /root/.local/bin
if [ -d /testbed/.venv ]; then
    ln -sfn /testbed/.venv /root/.venv
    if [ -x /testbed/.venv/bin/python ]; then
        ln -sfn /testbed/.venv/bin/python /root/.local/bin/python
        ln -sfn /testbed/.venv/bin/python /root/.local/bin/python3
    fi
    find /testbed/.venv/bin -maxdepth 1 -type f -executable -exec ln -sf {} /root/.local/bin/ \; 2>/dev/null || true
fi

runner=""
for candidate in /root/run_tests.sh /testbed/run_tests.sh /run_tests.sh; do
    if [ -f "$candidate" ]; then runner="$candidate"; break; fi
done
if [ -z "$runner" ]; then
    echo "R2E-Gym image is missing run_tests.sh" >&2
    exit 1
fi
tests_dir=""
for candidate in /root/r2e_tests /testbed/r2e_tests /r2e_tests; do
    if [ -d "$candidate" ]; then tests_dir="$candidate"; break; fi
done
if [ -z "$tests_dir" ]; then
    echo "R2E-Gym image is missing r2e_tests" >&2
    exit 1
fi

# Pack the held-out runner and tests (dereferencing symlinks, so a runner that points at a path removed below
# still has its contents), then remove every copy from the sandbox. The host keeps the archive during the agent
# run and writes it back only for scoring.
stage="$(mktemp -d /tmp/r2e_eval_stage.XXXXXX)"
cp -L "$runner" "$stage/run_tests.sh"
cp -RL "$tests_dir" "$stage/r2e_tests"
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$stage" -name '*.pyc' -delete
tar -czf /tmp/r2e_eval_assets.tar.gz -C "$stage" run_tests.sh r2e_tests
rm -rf "$stage"

# Also remove what the runner and the test tree resolve to, following symlinks transitively (links inside linked
# directories too), so no held-out file stays readable through a link target. Git-tracked repository files and the
# runtime environment (system and virtualenv paths) are kept; if the runner or the test tree itself resolves to such
# a path, setup fails instead of leaving the held-out tests readable.
git config --global --add safe.directory /testbed 2>/dev/null || true
testbed_is_git=0
if git -C /testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then testbed_is_git=1; fi
tracked() {
    [ "$testbed_is_git" = 1 ] && [ -n "$(git -C /testbed ls-files -- "${1#/testbed/}" 2>/dev/null | head -n 1)" ]
}
removable() {
    case "$1" in
        /testbed/r2e_tests | /testbed/r2e_tests/* | /testbed/run_tests.sh)
            ! tracked "$1" ;;
        "" | / | /root | /tmp | /opt | /home | /testbed | /testbed/.git | /testbed/.git/* | /testbed/.venv | \
            /testbed/.venv/* | /root/.venv* | /root/.local* | /opt/conda* | /usr | /usr/* | /bin* | /sbin* | /lib* | \
            /etc* | /proc* | /sys* | /dev* | /var*)
            return 1 ;;
        /testbed/*)
            [ "$testbed_is_git" = 1 ] && ! tracked "$1" ;;
        *) return 0 ;;
    esac
}
runner_target="$(readlink -f "$runner")"
tests_target="$(readlink -f "$tests_dir")"
declare -A seen=()
targets=()
queue=("$runner" "$tests_dir")
while ((${#queue[@]})); do
    path="${queue[0]}"
    queue=("${queue[@]:1}")
    target="$(readlink -f "$path" 2>/dev/null || true)"
    if [ -z "$target" ] || [ -n "${seen[$target]+x}" ]; then continue; fi
    seen[$target]=1
    targets+=("$target")
    if [ -d "$target" ]; then
        while IFS= read -r -d '' link; do queue+=("$link"); done < <(find "$target" -type l -print0)
    fi
done
for target in "${targets[@]}"; do
    if removable "$target"; then
        rm -rf "$target"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo "failed to remove held-out R2E-Gym path $target" >&2
            exit 1
        fi
    elif [ "$target" = "$runner_target" ] || [ "$target" = "$tests_target" ]; then
        echo "cannot hide held-out R2E-Gym tests: $target is a tracked or protected path" >&2
        exit 1
    fi
done
rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh

find /testbed -name '*.pyc' -delete
find /testbed -name '__pycache__' -type d -prune -exec rm -rf {} +
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
            from .reward import EVAL_ASSETS_PATH, compute_reward

            setup = await sandbox.exec_shell(_SETUP_SCRIPT, workdir="/testbed", timeout=cfg.setup_timeout)
            if setup.exit_code != 0:
                detail = setup.stderr.strip() or setup.stdout.strip() or f"exit code {setup.exit_code}"
                raise RuntimeError(f"failed to prepare R2E-Gym sandbox for {instance_id}: {detail}")
            # Held-out runner and tests live on the host during the agent run, out of the agent's reach.
            eval_assets = await sandbox.read_file(EVAL_ASSETS_PATH)
            removed = await sandbox.exec(["rm", "-f", EVAL_ASSETS_PATH])
            if removed.exit_code != 0:
                detail = removed.stderr.strip() or f"exit code {removed.exit_code}"
                raise RuntimeError(f"failed to remove staged R2E-Gym tests for {instance_id}: {detail}")

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

            result = await compute_reward(sample, sandbox, eval_timeout=cfg.eval_timeout, eval_assets=eval_assets)
            if isinstance(agent_error, str) and agent_error.strip():
                result = {**result, "agent_error": agent_error}
            logger.info(f"R2E-Gym task done: resolved={result['resolved']}")
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
