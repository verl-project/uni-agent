"""Host deployment: runs tool scripts directly on the host machine without containers."""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import (
    CommandTimeoutError,
    DeploymentNotStartedError,
    SessionNotInitializedError,
)
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


def _list_child_pids(parent_pid: int) -> list[int]:
    """Return direct child PIDs of `parent_pid`.

    On Linux we read /proc/PID/task/PID/children, which is cheap and exact.
    Elsewhere (macOS, etc.) we fall back to `pgrep -P`.
    """
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{parent_pid}/task/{parent_pid}/children") as f:
                return [int(p) for p in f.read().split() if p]
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    try:
        out = subprocess.check_output(["pgrep", "-P", str(parent_pid)], stderr=subprocess.DEVNULL)
        return [int(p) for p in out.decode().split() if p]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


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
        # Once the session is unusable (bash died, or stdout cannot be drained
        # after a timeout), we don't try to rebuild it: state would be lost
        # silently. Instead we mark it dead and surface errors so the upper
        # layer can fail the episode, matching the sandbox-based deployments.
        self._dead = False

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        startup_timeout = request.startup_timeout or 10
        self._process = await asyncio.create_subprocess_exec(
            "bash",
            "--norc",
            "--noprofile",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._env,
            # Job control (`set -m` below) puts each foreground command in
            # its own process group, so we can SIGINT just the user command
            # without killing the bash shell itself.
            start_new_session=True,
        )
        # `set -m` enables job control; PS1/PS2/PROMPT_COMMAND are zeroed so
        # nothing accidentally injects bytes between commands.
        setup = "set -m\nexport PS1='' PS2='' PROMPT_COMMAND=''\n"
        self._process.stdin.write(setup.encode())
        await self._process.stdin.drain()
        marker = f"__UNIAGENT_READY_{uuid.uuid4().hex[:12]}__"
        self._process.stdin.write(f"echo '{marker}'\n".encode())
        await self._process.stdin.drain()
        await self._read_until_marker(marker, timeout=startup_timeout)
        self._dead = False
        self.logger.info("Host bash session created")
        return CreateSessionResponse()

    def _check_session_alive(self) -> None:
        """Raise if the session is unusable. Callers are expected to surface
        this so the upper layer can fail the episode instead of silently
        running on a partially-recovered or fresh session with lost state."""
        if self._dead:
            raise SessionNotInitializedError("Host bash session is no longer usable")
        if self._process is None:
            raise SessionNotInitializedError("Host bash session has not been started")
        if self._process.returncode is not None:
            self._dead = True
            raise SessionNotInitializedError(f"Host bash session exited (returncode={self._process.returncode})")

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
            # SIGINT the foreground child of bash, NOT bash itself. With
            # `set -m`, bash puts each command in its own process group, so
            # this kills only the user command. bash then resumes reading
            # stdin, the trailing marker line gets printed, and the original
            # `_read_until_marker` call returns cleanly with exit_code=130.
            # We do this outside the IO lock so we can interrupt a command
            # whose run_in_session is currently waiting on stdout.
            self._signal_foreground_child(signal.SIGINT)
            return Observation(output="", exit_code=130)

        if not isinstance(action, BashAction):
            raise TypeError(f"Unsupported action type: {type(action)}")

        async with self._io_lock:
            self._check_session_alive()

            marker = f"__UNIAGENT_{uuid.uuid4().hex[:16]}__"
            wrapped = f"{action.command}\n__ua_ec=$?\necho '{marker}'\"$__ua_ec\"\n"

            self._process.stdin.write(wrapped.encode())
            await self._process.stdin.drain()

            timeout = getattr(action, "timeout", 60) or 60
            try:
                output, exit_code = await self._read_until_marker(marker, timeout)
            except CommandTimeoutError:
                # User command is still running. Try to interrupt it and drain
                # stdout up to the original marker so the next command sees a
                # clean stream. If draining fails, mark the session dead so
                # subsequent calls raise SessionNotInitializedError and the
                # upper layer fails the episode.
                await self._drain_after_timeout(marker)
                raise

            if output.endswith("\n"):
                output = output[:-1]

            return Observation(output=output, exit_code=exit_code)

    async def _drain_after_timeout(self, pending_marker: str) -> None:
        """SIGINT the running user command and consume residual output up to
        its marker so the next command starts on a clean stream. If we
        cannot drain, mark the session as dead."""
        if self._process is None or self._process.returncode is not None:
            self._dead = True
            return
        # No child to signal is fine (race: command may have just finished).
        # Either way, try to drain whatever the marker is following.
        self._signal_foreground_child(signal.SIGINT)
        try:
            await self._read_until_marker(pending_marker, timeout=5)
            self.logger.info("Drained residual output after command timeout")
        except CommandTimeoutError:
            self.logger.warning("Failed to drain stdout after SIGINT; session is dead")
            self._dead = True

    def _signal_foreground_child(self, sig: int) -> bool:
        """Send `sig` to bash's foreground job. With `set -m`, bash places
        each job (including a multi-process pipeline) in its own process
        group, so we look up each direct child's pgid and signal the group.
        Signaling bash itself would either be ignored or kill the shell;
        we want to terminate only the user command, including every stage
        of a pipeline. Returns True if at least one group was signaled."""
        if self._process is None or self._process.returncode is not None:
            return False
        signaled_pgids: set[int] = set()
        delivered = False
        for pid in _list_child_pids(self._process.pid):
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if pgid in signaled_pgids or pgid == self._process.pid:
                # Skip duplicates and refuse to signal bash's own group,
                # which would happen if job control somehow wasn't enabled.
                continue
            signaled_pgids.add(pgid)
            try:
                os.killpg(pgid, sig)
                delivered = True
            except (ProcessLookupError, PermissionError):
                # Fall back to per-pid kill if the group is gone or we
                # don't have permission for killpg.
                try:
                    os.kill(pid, sig)
                    delivered = True
                except ProcessLookupError:
                    continue
        return delivered

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
        alive = not self._dead and self._process is not None and self._process.returncode is None
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
