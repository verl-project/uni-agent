"""Version-independent Terminal-Bench task lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import Field

from uni_agent.sandbox import SandboxConfig, build_sandbox

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import install_archive, parse_json_mapping, resolve_env_mapping

logger = logging.getLogger(__name__)


class TerminalBenchTaskConfig(TaskConfig):
    name: str = "terminal_bench"
    run_oracle_solution: bool = Field(
        default=False,
        description="Skip the configured agent and run the task's official solution before verification.",
    )


def build_terminal_bench_sandbox_config(config: SandboxConfig, metadata: dict[str, Any]) -> SandboxConfig:
    """Apply provider-specific task environment settings to a SandboxConfig."""
    sandbox_config = config.model_copy(deep=True)
    environment = parse_json_mapping(metadata.get("environment_json"), field="environment_json")
    environment_env = resolve_env_mapping(environment.get("env", {}))
    workdir = environment.get("workdir")
    cpus = environment.get("cpus")
    memory_mb = environment.get("memory_mb")
    storage_mb = environment.get("storage_mb")
    gpus = int(environment.get("gpus") or 0)
    gpu_types = environment.get("gpu_types") or []
    allow_internet = bool(environment.get("allow_internet", True))
    kwargs = dict(sandbox_config.sandbox_kwargs)

    if sandbox_config.provider == "docker":
        run_args = list(kwargs.get("run_args", []))
        run_args.extend(["--platform", "linux/amd64"])
        if workdir:
            run_args.extend(["--workdir", str(workdir)])
        if cpus is not None:
            run_args.extend(["--cpus", f"{float(cpus):g}"])
        if memory_mb is not None:
            run_args.extend(["--memory", f"{int(memory_mb)}m"])
        if gpus > 0:
            run_args.extend(["--gpus", str(gpus)])
        if not allow_internet:
            run_args.extend(["--network", "none"])
        for name, value in sorted(environment_env.items()):
            run_args.extend(["--env", f"{name}={value}"])
        kwargs["run_args"] = run_args
    elif sandbox_config.provider == "modal":
        if workdir:
            kwargs.setdefault("workdir", str(workdir))
        if cpus is not None:
            kwargs.setdefault("cpu", float(cpus))
        if memory_mb is not None:
            kwargs.setdefault("memory", int(memory_mb))
        if gpus > 0:
            gpu_type = str(gpu_types[0]) if gpu_types else "any"
            kwargs.setdefault("gpu", f"{gpu_type}:{gpus}")
        kwargs.setdefault("block_network", not allow_internet)
        kwargs["env"] = {**dict(kwargs.get("env", {})), **environment_env}
    else:
        raise ValueError(
            f"Terminal-Bench supports sandbox providers 'docker' and 'modal', got {sandbox_config.provider!r}"
        )

    if storage_mb is not None:
        logger.debug("Ignoring advisory storage_mb=%s for %s", storage_mb, sandbox_config.provider)
    sandbox_config.sandbox_kwargs = kwargs
    return sandbox_config


async def _run_oracle(metadata: dict[str, Any], sandbox, *, timeout: float) -> dict[str, Any]:
    await install_archive(sandbox, metadata["solution_archive"], target_dir="/solution")
    chmod = await sandbox.exec(["chmod", "+x", "/solution/solve.sh"])
    if chmod.exit_code != 0:
        raise RuntimeError(f"failed to make Terminal-Bench oracle executable: {chmod.stderr.strip()}")

    solution_env = resolve_env_mapping(parse_json_mapping(metadata.get("solution_env_json"), field="solution_env_json"))
    solution_env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    response = await sandbox.exec_shell(
        "(/solution/solve.sh) > /logs/agent/oracle.txt 2>&1",
        env=solution_env,
        timeout=timeout,
    )
    return {
        "mode": "oracle",
        "exit_code": response.exit_code,
        "timed_out": response.exit_code == -1,
        "error": None if response.exit_code == 0 else (response.stderr or "").strip()[-2000:],
    }


@register_task("terminal_bench")
class TerminalBenchTask(Task):
    name = "terminal_bench"
    config_model = TerminalBenchTaskConfig

    async def run(self) -> TaskResult:
        cfg: TerminalBenchTaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        instance_id = metadata["instance_id"]
        dataset_version = metadata["dataset_version"]
        agent_timeout = float(metadata["agent_timeout"])
        verifier_timeout = float(metadata["verifier_timeout"])
        environment = parse_json_mapping(metadata.get("environment_json"), field="environment_json")
        workdir = str(environment["workdir"]) if environment.get("workdir") else None
        sandbox_config = build_terminal_bench_sandbox_config(cfg.sandbox, metadata)
        task_config_dump = cfg.model_dump(mode="json", exclude={"metadata", "prompt"})
        logger.info(
            "starting Terminal-Bench %s task (instance_id=%s, oracle=%s) "
            "agent_timeout=%gs verifier_timeout=%gs sandbox_runtime_timeout=%gs\n"
            "task config: %s\nsandbox config: %s",
            dataset_version,
            instance_id,
            cfg.run_oracle_solution,
            agent_timeout,
            verifier_timeout,
            sandbox_config.runtime_timeout,
            json.dumps(task_config_dump, indent=2),
            json.dumps(sandbox_config.model_dump(mode="json"), indent=2),
        )

        async with build_sandbox(sandbox_config) as sandbox:
            prepared = await sandbox.exec_shell(
                "mkdir -p /logs/agent /logs/verifier /logs/artifacts && "
                "chmod 777 /logs/agent /logs/verifier /logs/artifacts"
            )
            if prepared.exit_code != 0:
                raise RuntimeError(f"failed to prepare Terminal-Bench log directories: {prepared.stderr.strip()}")

            if cfg.run_oracle_solution:
                agent_info = await _run_oracle(
                    metadata,
                    sandbox,
                    timeout=agent_timeout,
                )
                finished: bool | None = agent_info["exit_code"] == 0
            else:
                agent = self.build_agent()
                try:
                    agent_result = await asyncio.wait_for(
                        agent.run(
                            sandbox=sandbox,
                            messages=cfg.prompt,
                            workdir=workdir,
                        ),
                        timeout=agent_timeout,
                    )
                except TimeoutError:
                    logger.warning("Terminal-Bench agent timed out for %s after %.0fs", instance_id, agent_timeout)
                    agent_info = {
                        "mode": "agent",
                        "timed_out": True,
                        "error": f"agent exceeded {agent_timeout:g}s",
                    }
                    finished = False
                except Exception as exc:  # score the resulting filesystem even when the agent fails
                    logger.exception("Terminal-Bench agent failed for %s; continuing to verifier", instance_id)
                    agent_info = {
                        "mode": "agent",
                        "timed_out": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    finished = False
                else:
                    agent_info = {
                        "mode": "agent",
                        "timed_out": False,
                        "error": None,
                        **agent_result.info,
                    }
                    finished = agent_result.finished

            from .reward import compute_reward

            result = await compute_reward(
                metadata,
                sandbox,
            )
            result["agent"] = agent_info

        score = float(result["reward"])
        logger.info(
            "Terminal-Bench task done: instance_id=%s reward=%.3f resolved=%s",
            instance_id,
            score,
            result["resolved"],
        )
        return TaskResult(
            reward=score,
            accuracy=score,
            finished=finished,
            extra_info=result,
        )
