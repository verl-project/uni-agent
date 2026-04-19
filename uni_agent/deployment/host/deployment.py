"""Host deployment: runs tool scripts directly on the host machine without containers."""

import asyncio
import os
import shutil
import signal
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import CommandTimeoutError, DeploymentNotStartedError
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    BashAction,
    BashInterruptAction,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)

from uni_agent.async_logging import get_logger


class HostRuntime(AbstractRuntime):
    """Runtime that executes commands in a persistent local bash session."""

    def __init__(self, run_id: str, env: dict[str, str] | None = None):
        self.logger = get_logger("host-runtime", run_id)
        self._env = dict(env or os.environ)
        self._process: asyncio.subprocess.Process | None = None

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        self._process = await asyncio.create_subprocess_exec(
            "bash",
            "--norc",
            "--noprofile",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._env,
        )
        setup = "export PS1='' PS2='' PROMPT_COMMAND=''\n"
        self._process.stdin.write(setup.encode())
        await self._process.stdin.drain()
        # Drain any startup output
        marker = f"__UNIAGENT_READY_{uuid.uuid4().hex[:12]}__"
        self._process.stdin.write(f"echo '{marker}'\n".encode())
        await self._process.stdin.drain()
        startup_timeout = request.startup_timeout or 10
        await self._read_until_marker(marker, timeout=startup_timeout)
        self.logger.info("Host bash session created")
        return CreateSessionResponse()

    async def _read_until_marker(self, marker: str, timeout: float) -> tuple[str, int]:
        """Read stdout until the marker line appears. Returns (output, exit_code)."""
        lines: list[str] = []
        try:
            async with asyncio.timeout(timeout):
                while True:
                    line_bytes = await self._process.stdout.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace")
                    if marker in line:
                        after = line.split(marker, 1)[1].strip()
                        exit_code = int(after) if after else 0
                        return "".join(lines), exit_code
                    lines.append(line)
        except (asyncio.TimeoutError, TimeoutError):
            partial = "".join(lines)
            rc = self._process.returncode if self._process is not None else "no_process"
            self.logger.error(
                f"_read_until_marker timed out after {timeout}s "
                f"(bash returncode={rc}, partial stdout repr, first 500 chars)={partial[:500]!r}"
            )
            raise CommandTimeoutError(f"Command timed out after {timeout}s")
        return "".join(lines), 1

    async def run_in_session(self, action: Action) -> Observation:
        if isinstance(action, BashInterruptAction):
            if self._process and self._process.returncode is None:
                self._process.send_signal(signal.SIGINT)
                await asyncio.sleep(0.5)
            return Observation(output="", exit_code=130)

        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")

        if self._process is None or self._process.returncode is not None:
            raise RuntimeError("No active bash session")

        marker = f"__UNIAGENT_{uuid.uuid4().hex[:16]}__"
        wrapped = f"{action.command}\n__ua_ec=$?\necho '{marker}'\"$__ua_ec\"\n"

        self._process.stdin.write(wrapped.encode())
        await self._process.stdin.drain()

        timeout = getattr(action, "timeout", 60) or 60
        output, exit_code = await self._read_until_marker(marker, timeout)

        if output.endswith("\n"):
            output = output[:-1]

        return Observation(output=output, exit_code=exit_code)

    async def execute(self, command: Command) -> CommandResponse:
        proc = await asyncio.create_subprocess_exec(
            *command.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        timeout = getattr(command, "timeout", 60) or 60
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return CommandResponse(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        )

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        path = Path(request.path)
        encoding = getattr(request, "encoding", None) or "utf-8"
        errors = getattr(request, "errors", None) or "replace"
        content = path.read_text(encoding=encoding, errors=errors)
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        path = Path(request.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(request.content)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        src = Path(request.source_path)
        tgt = Path(request.target_path)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, tgt, dirs_exist_ok=True)
        else:
            shutil.copy2(src, tgt)
        return UploadResponse()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        alive = self._process is not None and self._process.returncode is None
        return IsAliveResponse(is_alive=alive)

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        return CloseSessionResponse()

    async def close(self) -> CloseResponse:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        return CloseResponse()


class HostDeploymentConfig(BaseModel):
    """Configuration for host-local execution (no container)."""

    type: Literal["host"] = "host"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake.

    During parameter-sync weight reloads, fork()/exec() and even the asyncio event loop can be
    starved for tens of seconds.
    """

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        return HostDeployment.from_config(self, run_id)


class HostDeployment(AbstractDeployment):
    """Deployment that runs tool scripts directly on the host machine."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = HostDeploymentConfig(**kwargs)
        self._runtime: HostRuntime | None = None
        self.logger = get_logger("host-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._tool_dir: Path | None = None
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: HostDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            return IsAliveResponse(is_alive=False)
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self, max_retries: int = 5):
        self._tool_dir = Path(tempfile.mkdtemp(prefix=f"uni-agent-host-{self.run_id[:8]}-"))
        self.logger.info(f"Created tool directory: {self._tool_dir}")

        env = dict(os.environ)
        env["PATH"] = f"{self._tool_dir}:{env.get('PATH', '')}"

        self._runtime = HostRuntime(run_id=self.run_id, env=env)
        await self._runtime.create_session(
            CreateSessionRequest(startup_source=[], startup_timeout=self._config.startup_timeout)
        )
        self._stopped = False
        self.logger.info("Host deployment started")

    async def stop(self):
        if self._stopped:
            return

        if self._runtime:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close host runtime: {exc}")
            self._runtime = None

        if self._tool_dir and self._tool_dir.exists():
            try:
                shutil.rmtree(self._tool_dir)
            except Exception as exc:
                self.logger.error(f"Failed to clean up tool directory: {exc}")

        self._stopped = True
        self.logger.info("Host deployment stopped")

    async def copy_to_container(self, src: Path, tgt: Path):
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
        self.logger.debug(f"Copied tool {src.name} -> {tgt}")

    @property
    def tool_install_dir(self) -> Path:
        """Directory on the host where tool scripts are installed. Already on PATH."""
        if self._tool_dir is None:
            raise DeploymentNotStartedError()
        return self._tool_dir

    @property
    def runtime(self) -> HostRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
