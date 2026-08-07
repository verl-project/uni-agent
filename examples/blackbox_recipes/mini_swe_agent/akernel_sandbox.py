"""AKernel sandbox provider used by the mini-swe-agent recipe."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from typing import Any

from uni_agent.sandbox import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import register_sandbox

logger = logging.getLogger(__name__)


def _to_akernel_image(image: str) -> str:
    if image.startswith("swebench/"):
        return (
            image.replace(
                "swebench/",
                "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/",
                1,
            )
            + ":v2"
        )
    if image.startswith("swerebench/"):
        return (
            image.replace(
                "swerebench/",
                "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/",
                1,
            )
            + ":latest"
        )
    return image


@register_sandbox("akernel")
class AKernelSandbox(Sandbox):
    """Run a task image on AKernel and expose the standard Sandbox interface."""

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float,
        upstream: str,
        proxy_port: int = 38197,
        sidecar_image: str | None = None,
        sidecar_target: str = "/opt/mini-swe-agent-venv",
        post_setup_cmd: str = "",
        cpu: int = 2000,
        memory: int = 4096,
        cpu_limit: int = 8000,
        mem_limit: int = 12288,
    ) -> None:
        self.image = _to_akernel_image(image)
        self.runtime_timeout = runtime_timeout
        self.upstream = upstream
        self.proxy_port = proxy_port
        self.sidecar_image = sidecar_image
        self.sidecar_target = sidecar_target
        self.post_setup_cmd = post_setup_cmd
        self.cpu = cpu
        self.memory = memory
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self._sandbox: Any | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> AKernelSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    async def start(self) -> None:
        server = os.getenv("AKERNEL_SERVER_ADDRESS")
        token = os.getenv("AKERNEL_TOKEN")
        if not server or not token:
            raise ValueError("AKERNEL_SERVER_ADDRESS and AKERNEL_TOKEN must be set")
        os.environ["TUNNEL_SSL_VERIFY"] = os.getenv(
            "TUNNEL_SSL_VERIFY",
            os.getenv("AKERNEL_TUNNEL_SSL_VERIFY", "0"),
        )

        from akernel_sdk import Mount
        from akernel_sdk import Sandbox as SDKAKernelSandbox

        kwargs: dict[str, Any] = {
            "image": self.image,
            "cpu": self.cpu,
            "memory": self.memory,
            "cpu_limit": self.cpu_limit,
            "mem_limit": self.mem_limit,
            "idle_timeout": int(self.runtime_timeout),
            "upstream": self.upstream,
            "proxy_port": self.proxy_port,
        }
        if self.sidecar_image:
            kwargs["mounts"] = [Mount(target=self.sidecar_target, image_url=self.sidecar_image)]
        if prefix := os.getenv("SANDBOX_NAME_PREFIX"):
            kwargs["name"] = f"{prefix}{uuid.uuid4().hex[:8]}"

        self._sandbox = await asyncio.to_thread(lambda: SDKAKernelSandbox(**kwargs))
        logger.info("AKernel sandbox created: %s", getattr(self._sandbox, "sandbox_id", "unknown"))
        if self.post_setup_cmd:
            setup = await self.exec_shell(self.post_setup_cmd, timeout=600)
            if setup.exit_code != 0:
                raise RuntimeError(f"AKernel post_setup_cmd failed: {(setup.stderr or setup.stdout)[-2000:]}")

    async def stop(self) -> None:
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is None:
            return
        try:
            if sandbox.is_running():
                await asyncio.to_thread(sandbox.kill)
        except Exception:
            logger.warning("Failed to stop AKernel sandbox", exc_info=True)
        logger.info("AKernel sandbox stopped: %s", getattr(sandbox, "sandbox_id", "unknown"))

    async def is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(await asyncio.to_thread(self._sandbox.is_running))
        except Exception:
            return False

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("AKernel sandbox has not started")
        command = shlex.join(argv)
        if env:
            assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
            command = f"env {assignments} {command}"
        if workdir:
            command = f"cd {shlex.quote(workdir)} && {command}"
        result = await asyncio.to_thread(
            self._sandbox.commands.run,
            command,
            timeout=int(timeout or 600),
        )
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", -1)),
            stdout=stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout),
            stderr=stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr),
        )
