"""Host deployment: runs tool scripts directly on the host machine without containers."""

import asyncio
import os
import shutil
import signal
import uuid
from pathlib import Path
from typing import Any, Self

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
from uni_agent.deployment.config import HostDeploymentConfig


class HostRuntime(AbstractRuntime):
    """Runtime that executes commands in a persistent local bash session."""

    def __init__(self, run_id: str, env: dict[str, str] | None = None):
        self.logger = get_logger("host-runtime", run_id)
        self._env = dict(env or os.environ)
        self._process: asyncio.subprocess.Process | None = None
        # Serialize all bash stdin/stdout IO. interrupt_session intentionally
        # does NOT acquire this lock: it sends SIGINT out-of-band so the holder
        # of the lock (a blocked `_read_until_marker`) gets unblocked when bash
        # finishes the interrupted command and prints the trailing marker.
        self._io_lock = asyncio.Lock()
        # Session start time + a generation counter so a stale rebuild does
        # not clobber a fresh session created by a concurrent path.
        self._session_startup_timeout: float = 10.0
        self._broken = False

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        self._session_startup_timeout = request.startup_timeout or 10
        await self._spawn_bash(self._session_startup_timeout)
        return CreateSessionResponse()

    async def _spawn_bash(self, startup_timeout: float) -> None:
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
        marker = f"__UNIAGENT_READY_{uuid.uuid4().hex[:12]}__"
        self._process.stdin.write(f"echo '{marker}'\n".encode())
        await self._process.stdin.drain()
        await self._read_until_marker(marker, timeout=startup_timeout)
        self._broken = False
        self.logger.info("Host bash session created")

    async def _ensure_session(self) -> None:
        """Rebuild bash if the previous one died or was marked broken."""
        if self._process is None or self._process.returncode is not None or self._broken:
            if self._process is not None and self._process.returncode is None:
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=2)
                except Exception:
                    pass
            self.logger.warning("Rebuilding host bash session")
            await self._spawn_bash(self._session_startup_timeout)

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
            raise CommandTimeoutError(f"Command timed out after {timeout}s") from None
        return "".join(lines), 1

    async def run_in_session(self, action: Action) -> Observation:
        if isinstance(action, BashInterruptAction):
            # Fire SIGINT WITHOUT taking the IO lock so we can interrupt a
            # command whose run_in_session is currently waiting on stdout.
            # bash will abort the foreground command and resume reading stdin,
            # so the original `_read_until_marker` call will eventually see
            # its marker (with exit_code != 0) and return cleanly.
            if self._process and self._process.returncode is None:
                self._process.send_signal(signal.SIGINT)
            return Observation(output="", exit_code=130)

        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")

        async with self._io_lock:
            await self._ensure_session()

            marker = f"__UNIAGENT_{uuid.uuid4().hex[:16]}__"
            wrapped = f"{action.command}\n__ua_ec=$?\necho '{marker}'\"$__ua_ec\"\n"

            self._process.stdin.write(wrapped.encode())
            await self._process.stdin.drain()

            timeout = getattr(action, "timeout", 60) or 60
            try:
                output, exit_code = await self._read_until_marker(marker, timeout)
            except CommandTimeoutError:
                # The user's command is still running inside bash. Try to
                # interrupt it and drain stdout up to the original marker so
                # the next command starts on a clean stream. If that fails,
                # mark the session broken so the next call rebuilds it.
                await self._recover_after_timeout(marker)
                raise

            if output.endswith("\n"):
                output = output[:-1]

            return Observation(output=output, exit_code=exit_code)

    async def _recover_after_timeout(self, pending_marker: str) -> None:
        """Best-effort: SIGINT the running command and consume residual output
        up to its marker. Marks the session broken on failure."""
        if self._process is None or self._process.returncode is not None:
            self._broken = True
            return
        try:
            self._process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            self._broken = True
            return
        try:
            # Short timeout: bash should produce the trailing marker quickly
            # once the foreground command stops. If not, the shell is wedged.
            await self._read_until_marker(pending_marker, timeout=5)
            self.logger.info("Drained residual output after command timeout")
        except CommandTimeoutError:
            self.logger.warning(
                "Failed to drain stdout after SIGINT; marking session as broken"
            )
            self._broken = True

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


class HostDeployment(AbstractDeployment):
    """Deployment that runs tool scripts directly on the host machine."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = HostDeploymentConfig(**kwargs)
        self._runtime: HostRuntime | None = None
        self.logger = get_logger("host-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
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
        env = dict(os.environ)

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

        self._stopped = True
        self.logger.info("Host deployment stopped")

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
