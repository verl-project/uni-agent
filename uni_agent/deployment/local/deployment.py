import asyncio
import os
import re
import shlex
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import Command, CreateBashSessionRequest, IsAliveResponse, UploadRequest
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger

from .runtime import LocalRuntime, LocalRuntimeConfig


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return sanitized or "uni-agent-local"


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalDeploymentConfig(BaseModel):
    """Configuration for a local Docker/Podman sandbox."""

    image: str = "python:3.12"
    """Container image used for the sandbox."""
    command: str = "python3 -m pip install -q swerex && python3 -m swerex.server --auth-token {token}"
    """Command to run inside the sandbox."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    container_runtime: str = "docker"
    """Container runtime executable, typically docker or podman."""
    container_name: str | None = None
    """Optional container name override."""
    host: str | None = None
    """Override the runtime host. Defaults to localhost outside containers and container IP inside containers."""
    published_port: int | None = None
    """Host port mapped to the sandbox runtime port. If unset, a free local port is chosen."""
    runtime_port: int = 8000
    """Port exposed by the swerex server inside the sandbox."""
    network: str | None = None
    """Optional Docker network to attach the sandbox to."""
    shell: str = "/bin/bash"
    """Shell executable used as the container entrypoint."""
    extra_run_args: list[str] = Field(default_factory=list)
    """Extra args appended to the container runtime `run` command."""

    type: Literal["local"] = "local"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        return LocalDeployment.from_config(self, run_id)


class LocalDeployment(AbstractDeployment):
    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = LocalDeploymentConfig(**kwargs)
        self._runtime: LocalRuntime | None = None
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._container_name: str | None = None
        self._container_id: str | None = None
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: LocalDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float) -> IsAliveResponse:
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=0.5)
        except TimeoutError as e:
            self.logger.error("Local runtime did not start within timeout.")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _runtime_exec(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        self.logger.debug(f"Running container runtime command: {_shell_join(args)}")
        try:
            return subprocess.run(args, check=check, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Container runtime {self._config.container_runtime!r} was not found in PATH") from exc

    def _get_current_container_network(self) -> str | None:
        if not _is_running_in_container():
            return None

        container_id = os.getenv("HOSTNAME")
        if not container_id:
            return None

        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_id,
                    "--format",
                    "{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}",
                ]
            )
        except subprocess.CalledProcessError as exc:
            self.logger.warning(
                f"Failed to inspect current container network: {exc.stderr.strip() or exc.stdout.strip()}"
            )
            return None

        networks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not networks:
            return None
        return networks[0]

    def _get_container_ip(self, container_name: str) -> str | None:
        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_name,
                    "--format",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ]
            )
        except subprocess.CalledProcessError as exc:
            self.logger.warning(f"Failed to inspect sandbox IP: {exc.stderr.strip() or exc.stdout.strip()}")
            return None

        ip_address = result.stdout.strip()
        return ip_address or None

    def _get_runtime_host(self, container_name: str) -> str:
        if self._config.host:
            return self._config.host

        if _is_running_in_container() or self._config.network:
            container_ip = self._get_container_ip(container_name)
            if container_ip:
                return f"http://{container_ip}"

        return "http://127.0.0.1"

    def _build_run_command(self, container_name: str, published_port: int, command: str) -> list[str]:
        network = self._config.network or self._get_current_container_network()

        args = [
            self._config.container_runtime,
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            self._config.shell,
        ]
        if network:
            args.extend(["--network", network])
        args.extend(["-p", f"{published_port}:{self._config.runtime_port}"])
        args.extend(self._config.extra_run_args)
        args.extend([self._config.image, "-lc", command])
        return args

    def _get_container_logs(self, container_name: str) -> str:
        try:
            result = self._runtime_exec([self._config.container_runtime, "logs", container_name], check=False)
        except Exception as exc:
            return f"<failed to fetch logs: {exc}>"
        return (result.stdout or result.stderr).strip()

    async def start(self, max_retries: int = 5):
        token = self._get_token()
        command = self._config.command.format(token=token)
        published_port = self._config.published_port or _pick_free_port()
        container_name = self._config.container_name or f"uni-agent-{_sanitize_name(self.run_id)}"
        self._stopped = False

        last_error: Exception | None = None
        for attempt in range(max_retries):
            self.logger.info(
                f"Starting local deployment with runtime={self._config.container_runtime}, image={self._config.image}."
            )
            self._hooks.on_custom_step("Creating local sandbox")
            self._container_name = container_name

            try:
                result = await asyncio.to_thread(
                    self._runtime_exec,
                    self._build_run_command(container_name, published_port, command),
                )
                self._container_id = result.stdout.strip()
                runtime_config = LocalRuntimeConfig(
                    auth_token=token,
                    host=self._get_runtime_host(container_name),
                    port=self._config.runtime_port,
                    timeout=self._config.timeout,
                )
                self._runtime = LocalRuntime.from_config(runtime_config, run_id=self.run_id)

                await self._wait_until_alive(timeout=self._config.startup_timeout)
                await self.runtime.create_session(
                    CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
                )
                self._stopped = False
                return
            except Exception as exc:
                last_error = exc
                logs = self._get_container_logs(container_name)
                self.logger.error(f"Failed to start local sandbox: {exc}\nContainer logs:\n{logs}")
                await self.stop()
                if attempt < max_retries - 1:
                    sleep_time = min(30, 2**attempt)
                    self.logger.info(f"Retrying local deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create local sandbox after {max_retries} retries") from last_error

    async def copy_to_container(self, src: Path, tgt: Path):
        await self.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    async def stop(self):
        if self._stopped:
            return

        if self._runtime:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close local runtime within timeout: {exc}")
            self._runtime = None

        if self._container_name:
            try:
                await asyncio.to_thread(
                    self._runtime_exec,
                    [self._config.container_runtime, "rm", "-f", self._container_name],
                    False,
                )
            except Exception as exc:
                self.logger.error(f"Failed to delete local sandbox {self._container_name}: {exc}")
            finally:
                self._container_name = None
                self._container_id = None

        self._stopped = True

    @property
    def runtime(self) -> LocalRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def __del__(self):
        if hasattr(self, "_container_name") and self._container_name and not getattr(self, "_stopped", False):
            msg = "Ensuring local deployment is stopped because object is deleted"
            try:
                self.logger.debug(msg)
            except Exception:
                print(msg)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass
        self._stopped = True
