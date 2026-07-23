from __future__ import annotations

import asyncio
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig


@register_sandbox("docker")
class DockerSandbox(Sandbox):
    """Run an isolated sandbox from an image available to a local Docker daemon."""

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        docker_binary: str = "docker",
        container_name: str | None = None,
        run_args: list[str] | None = None,
        pull_policy: str = "missing",
        entrypoint: str = "sleep",
        command: list[str] | None = None,
    ) -> None:
        self.image = image
        self.docker_binary = docker_binary
        self.container_name = container_name
        self.run_args = list(run_args or [])
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be one of: 'always', 'missing', 'never'")
        self.pull_policy = pull_policy
        self.entrypoint = entrypoint
        self.command = list(command or ["infinity"])
        self._container_name: str | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> DockerSandbox:
        return cls(image=config.image, **config.sandbox_kwargs)

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_binary,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Docker executable {self.docker_binary!r} was not found") from exc

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.communicate()
            raise

        return ExecResult(
            exit_code=int(proc.returncode or 0),
            stdout=_to_str(stdout),
            stderr=_to_str(stderr),
        )

    async def start(self) -> None:
        if self._container_name is not None:
            return

        if self.pull_policy == "never":
            inspected = await self._run_docker("image", "inspect", self.image)
            if inspected.exit_code != 0:
                detail = inspected.stderr.strip() or inspected.stdout.strip()
                raise RuntimeError(f"Docker image {self.image!r} is not available locally: {detail}")

        name = self.container_name or f"uni-agent-{uuid.uuid4().hex[:12]}"
        args = ["run", "--rm", "-d", "--name", name, "--pull", self.pull_policy]
        if self.entrypoint:
            args.extend(["--entrypoint", self.entrypoint])
        args.extend(self.run_args)
        args.append(self.image)
        args.extend(self.command)

        started = await self._run_docker(*args)
        if started.exit_code != 0:
            detail = started.stderr.strip() or started.stdout.strip()
            raise RuntimeError(f"Failed to start Docker sandbox from {self.image!r}: {detail}")
        self._container_name = name

    async def stop(self) -> None:
        name, self._container_name = self._container_name, None
        if name is not None:
            await self._run_docker("rm", "-f", name)

    def _require_container(self) -> str:
        if self._container_name is None:
            raise RuntimeError("DockerSandbox not started; call start() first")
        return self._container_name

    async def is_alive(self) -> bool:
        if self._container_name is None:
            return False
        try:
            result = await self._run_docker(
                "inspect",
                "--format",
                "{{.State.Running}}",
                self._container_name,
                timeout=10.0,
            )
            return result.exit_code == 0 and result.stdout.strip() == "true"
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
        args = ["exec"]
        if workdir:
            args.extend(["--workdir", workdir])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.append(self._require_container())
        args.extend(argv)
        return await self._run_docker(*args, timeout=timeout)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        container = self._require_container()
        parent = str(PurePosixPath(remote_file).parent)
        if parent not in {"", "."}:
            created = await self.exec(["mkdir", "-p", parent])
            if created.exit_code != 0:
                raise RuntimeError(f"Failed to create Docker sandbox directory {parent!r}: {created.stderr.strip()}")
        result = await self._run_docker("cp", str(local_file), f"{container}:{remote_file}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to upload {local_file!s} to {remote_file!r}: {result.stderr.strip()}")

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        destination = Path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = await self._run_docker(
            "cp",
            f"{self._require_container()}:{remote_file}",
            str(destination),
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to download {remote_file!r}: {result.stderr.strip()}")
