"""Remote Docker provider and per-session host/NPU configuration."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import TYPE_CHECKING, Any

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.sandbox.registry import register_sandbox

if TYPE_CHECKING:
    from uni_agent.sandbox.base import SandboxConfig


@register_sandbox("triton_remote_docker")
class RemoteDockerSandbox(DockerSandbox):
    """Use the stock Docker sandbox through one remote Docker endpoint."""

    def __init__(
        self,
        *,
        docker_host: str,
        image: str,
        npu_lock_dir: str,
        runtime_timeout: float = 3600.0,
        docker_binary: str = "docker",
        run_args: list[str] | None = None,
        pull_policy: str = "never",
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if not docker_host:
            raise ValueError("docker_host is required")
        if not math.isfinite(runtime_timeout) or runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be finite and positive")

        args = list(run_args or [])
        args.extend(["--volume", f"{npu_lock_dir}:{npu_lock_dir}"])
        if cwd:
            args.extend(["--workdir", cwd])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])

        self.docker_host = docker_host
        self.runtime_timeout = runtime_timeout
        super().__init__(
            image=image,
            docker_binary=docker_binary,
            run_args=args,
            pull_policy=pull_policy,
            start_timeout=60,
            entrypoint="sleep",
            command=[str(math.ceil(runtime_timeout))],
        )

    @classmethod
    def from_config(cls, config: SandboxConfig) -> RemoteDockerSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        if timeout is None and (args[:2] == ("image", "inspect") or args[:1] == ("rm",)):
            timeout = 30
        return await super()._run_docker("--host", self.docker_host, *args, timeout=timeout)


def _copy_sandbox_kwargs(tools_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = copy.deepcopy(tools_kwargs)
    task_config = copied.get("task")
    if not isinstance(task_config, dict):
        raise ValueError("run_triton_task requires tools_kwargs['task']")
    sandbox_config = task_config.setdefault("sandbox", {})
    if not isinstance(sandbox_config, dict):
        raise TypeError("tools_kwargs['task']['sandbox'] must be a mapping")
    sandbox_kwargs = sandbox_config.setdefault("sandbox_kwargs", {})
    if not isinstance(sandbox_kwargs, dict):
        raise TypeError("tools_kwargs['task']['sandbox']['sandbox_kwargs'] must be a mapping")
    return copied, sandbox_kwargs


def parse_device_ids(value: str) -> tuple[str, ...]:
    raw_devices = tuple(part.strip() for part in value.split(","))
    if not raw_devices or any(not part for part in raw_devices):
        raise ValueError(f"evaluator_npu_device_ids cannot contain empty entries; received {value!r}")
    devices = raw_devices
    if len(set(devices)) != len(devices):
        raise ValueError("evaluator_npu_device_ids cannot contain duplicates")
    if any(not part.replace("-", "").replace("_", "").isalnum() for part in devices):
        raise ValueError("evaluator_npu_device_ids contains an unsafe device ID")
    return devices


def bind_remote_sandbox(
    tools_kwargs: dict[str, Any],
    *,
    hosts: str,
    session_id: str,
    devices: tuple[str, ...],
    lock_dir: str,
    lock_timeout: float,
) -> dict[str, Any]:
    """Copy sample config and bind its Docker host and NPU lease contract."""

    docker_hosts = tuple(part.strip() for part in hosts.split(","))
    if not docker_hosts or any(not part for part in docker_hosts):
        raise ValueError("remote_docker_hosts cannot contain empty entries")
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError("evaluator_npu_lock_timeout must be finite and positive")

    slot = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:8], "big") % len(docker_hosts)
    copied, sandbox_kwargs = _copy_sandbox_kwargs(tools_kwargs)
    sandbox_kwargs["docker_host"] = docker_hosts[slot]
    sandbox_kwargs["npu_lock_dir"] = lock_dir
    env = sandbox_kwargs.setdefault("env", {})
    if not isinstance(env, dict):
        raise TypeError("sandbox_kwargs.env must be a mapping")
    env.update(
        {
            "TRITON_EVAL_DEVICE_IDS": ",".join(devices),
            "TRITON_EVAL_LOCK_DIR": lock_dir,
            "TRITON_EVAL_LOCK_TIMEOUT": str(lock_timeout),
        }
    )
    return copied
