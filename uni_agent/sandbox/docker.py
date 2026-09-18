from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig


async def _finish_cleanup(awaitable: Awaitable[Any]) -> Any:
    """Finish bounded resource cleanup even if the caller is cancelled again."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


def _positive_timeout(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds, got {value!r}")
    return value


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
        pull_timeout: float | None = None,
        start_timeout: float | None = None,
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
        self.pull_timeout = _positive_timeout("pull_timeout", pull_timeout)
        self.start_timeout = _positive_timeout("start_timeout", start_timeout)
        self.entrypoint = entrypoint
        self.command = list(command or ["infinity"])
        self._container_name: str | None = None
        self._cleanup_label: str | None = None

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
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await _finish_cleanup(proc.communicate())
            raise

        return ExecResult(
            exit_code=int(proc.returncode or 0),
            stdout=_to_str(stdout),
            stderr=_to_str(stderr),
        )

    async def _has_image(self) -> bool:
        return (await self._run_docker("image", "inspect", self.image)).exit_code == 0

    async def _pull_image(self) -> None:
        """Fetch the image up front so the pull is bounded by ``pull_timeout``, not by ``docker run``."""
        try:
            pulled = await self._run_docker("pull", self.image, timeout=self.pull_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Pulling Docker image {self.image!r} exceeded pull_timeout={self.pull_timeout:g}s"
            ) from exc
        if pulled.exit_code != 0:
            detail = pulled.stderr.strip() or pulled.stdout.strip()
            raise RuntimeError(f"Failed to pull Docker image {self.image!r}: {detail}")

    async def start(self) -> None:
        if self._container_name is not None:
            return
        if self._cleanup_label is not None:
            # A previous removal failed. Retry it before allocating another rollout.
            await self.stop()

        if self.pull_policy == "never":
            inspected = await self._run_docker("image", "inspect", self.image)
            if inspected.exit_code != 0:
                detail = inspected.stderr.strip() or inspected.stdout.strip()
                raise RuntimeError(f"Docker image {self.image!r} is not available locally: {detail}")

        # A separate `docker pull` to time-bound the pull on its own
        pull_policy = self.pull_policy
        if self.pull_timeout is not None and pull_policy != "never":
            if pull_policy == "always" or not await self._has_image():
                await self._pull_image()
            pull_policy = "never"

        name = self.container_name or f"uni-agent-{uuid.uuid4().hex[:12]}"
        args = ["run", "--rm", "-d", "--name", name, "--pull", pull_policy]
        if self.entrypoint:
            args.extend(["--entrypoint", self.entrypoint])
        args.extend(self.run_args)
        # Names may collide with containers we do not own. A per-attempt label lets
        # cleanup find only our container, including when `run` never returns its ID.
        self._cleanup_label = f"uni-agent.sandbox={uuid.uuid4().hex}"
        args.extend(["--label", self._cleanup_label])
        args.append(self.image)
        args.extend(self.command)

        try:
            started = await self._run_docker(*args, timeout=self.start_timeout)
            if started.exit_code != 0:
                detail = started.stderr.strip() or started.stdout.strip()
                raise RuntimeError(f"Failed to start Docker sandbox from {self.image!r}: {detail}")
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise TimeoutError(
                f"Starting Docker sandbox from {self.image!r} exceeded start_timeout={self.start_timeout:g}s"
            ) from exc
        except (Exception, asyncio.CancelledError):
            await self.stop()
            raise
        self._container_name = name

    async def stop(self) -> None:
        try:
            await _finish_cleanup(self._stop())
        except asyncio.TimeoutError as exc:
            # A removal timeout is an infrastructure error, not a command timeout
            # that Sandbox.exec may turn into an ordinary tool observation.
            raise RuntimeError(f"Docker sandbox cleanup timed out (owner={self._cleanup_label!r})") from exc

    async def _stop(self) -> None:
        # Invalidate the data plane immediately; retain the ownership label until
        # removal succeeds so callers can retry a failed cleanup.
        self._container_name = None
        if self._cleanup_label is None:
            return
        found = await self._run_docker(
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={self._cleanup_label}",
            timeout=30.0,
        )
        if found.exit_code != 0:
            raise RuntimeError(f"Failed to locate Docker sandbox for cleanup: {found.stderr.strip()}")
        for container_id in found.stdout.split():
            removed = await self._run_docker("rm", "-f", container_id, timeout=30.0)
            if removed.exit_code != 0:
                raise RuntimeError(f"Failed to remove Docker sandbox {container_id!r}: {removed.stderr.strip()}")
        self._cleanup_label = None

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
        try:
            return await self._run_docker(*args, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Killing the Docker CLI does not terminate the command in the container.
            # Retire this rollout before returning so its processes cannot keep
            # modifying the workspace or race a subsequent verifier.
            await self.stop()
            raise

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
