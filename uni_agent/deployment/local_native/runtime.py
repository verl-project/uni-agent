"""LocalNative runtime: pexpect-backed bash session, adapted from ``swerex.runtime.local``.

Why a separate runtime from ``HostRuntime``?

``HostRuntime`` builds the bash subprocess with ``asyncio.create_subprocess_exec``,
which means the resulting stdin/stdout streams are bound to the event loop that
created them. The framework's ``auto_await`` runs sync-style calls via
``asyncio.run(coro)``, creating a fresh loop each time -- after the loop that
``env.start()`` ran on closes, the next ``env.communicate()`` call lands on a
new loop and asyncio raises "got Future attached to a different loop" trying to
read from those stale streams.

``LocalNativeRuntime`` sidesteps the problem by driving the bash session
through ``pexpect``, which uses synchronous PTY I/O and holds no loop-bound
state. The async methods are thin shells (``await asyncio.sleep(...)`` for
pacing), so they run inside any loop. This mirrors how SWE-ReX's
``LocalRuntime`` is used directly when the rest of the framework lives in the
same process (i.e. without the HTTP layer).

Code is a direct port of
https://github.com/SWE-agent/SWE-ReX/blob/main/src/swerex/runtime/local.py
with uni-agent's logger conventions and minor cleanups; see ``LocalDeployment``
for the matching deployment wrapper.
"""

# ruff: noqa: E501
import asyncio
import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Self

import bashlex
import bashlex.ast
import bashlex.errors
import pexpect
from swerex.exceptions import (
    BashIncorrectSyntaxError,
    CommandTimeoutError,
    NoExitCodeError,
    NonZeroExitCodeError,
    SessionDoesNotExistError,
    SessionExistsError,
    SessionNotInitializedError,
)
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseBashSessionResponse,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateBashSessionRequest,
    CreateBashSessionResponse,
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

__all__ = ["LocalNativeRuntime", "BashSession"]


def _split_bash_command(input: str) -> list[str]:
    r"""Split a bash command with linebreaks, escaped newlines, and heredocs into a list of
    individual commands.

    Examples:
    - "cmd1\ncmd2" are two commands
    - "cmd1\\\n asdf" is one command (linebreak is escaped)
    - "cmd1<<EOF\na\nb\nEOF" is one command (heredoc)
    """
    input = input.strip()
    if not input or all(line.strip().startswith("#") for line in input.splitlines()):
        # bashlex can't deal with empty strings or the like.
        return []
    parsed = bashlex.parse(input)
    cmd_strings = []

    def find_range(cmd: bashlex.ast.node) -> tuple[int, int]:
        start = cmd.pos[0]  # type: ignore
        end = cmd.pos[1]  # type: ignore
        for part in getattr(cmd, "parts", []):
            part_start, part_end = find_range(part)
            start = min(start, part_start)
            end = max(end, part_end)
        return start, end

    for cmd in parsed:
        start, end = find_range(cmd)
        cmd_strings.append(input[start:end])
    return cmd_strings


def _strip_control_chars(s: str) -> str:
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s).replace("\r\n", "\n")


def _check_bash_command(command: str) -> None:
    """Check if a bash command is valid. Raises BashIncorrectSyntaxError if it's not."""
    _unique_string = "SOUNIQUEEOF"
    cmd = f"/usr/bin/env bash -n << '{_unique_string}'\n{command}\n{_unique_string}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode == 0:
        return
    stdout = result.stdout.decode(errors="backslashreplace")
    stderr = result.stderr.decode(errors="backslashreplace")
    msg = (
        f"Error (exit code {result.returncode}) while checking bash command \n{command!r}\n"
        f"---- Stderr ----\n{stderr}\n---- Stdout ----\n{stdout}"
    )
    raise BashIncorrectSyntaxError(msg, extra_info={"bash_stdout": stdout, "bash_stderr": stderr})


class Session(ABC):
    @abstractmethod
    async def start(self) -> CreateSessionResponse: ...

    @abstractmethod
    async def run(self, action: Action) -> Observation: ...

    @abstractmethod
    async def close(self) -> CloseSessionResponse: ...


class BashSession(Session):
    """A persistent bash REPL controlled via pexpect (PTY-based)."""

    _UNIQUE_STRING = "UNIQUESTRING29234"

    def __init__(self, request: CreateBashSessionRequest, *, run_id: str):
        self.request = request
        self._ps1 = "SHELLPS1PREFIX"
        self._shell: pexpect.spawn | None = None
        self.logger = get_logger("local-native-session", run_id)

    @property
    def shell(self) -> pexpect.spawn:
        if self._shell is None:
            raise RuntimeError("shell not initialized")
        return self._shell

    def _get_reset_commands(self) -> list[str]:
        """Commands that reset PS1/PS2/PS0 to known values."""
        return [
            f"export PS1='{self._ps1}'",
            "export PS2=''",
            "export PS0=''",
        ]

    async def start(self) -> CreateBashSessionResponse:
        """Spawn the bash REPL, source startup files, and set the prompt."""
        self._shell = pexpect.spawn(
            "/usr/bin/env bash",
            encoding="utf-8",
            codec_errors="backslashreplace",
            echo=False,
            env=dict(os.environ.copy(), **{"PS1": self._ps1, "PS2": "", "PS0": ""}),  # type: ignore
        )
        await asyncio.sleep(0.3)
        cmds = []
        if self.request.startup_source:
            cmds += [f"source {path}" for path in self.request.startup_source] + ["sleep 0.3"]
        cmds += self._get_reset_commands()
        cmd = " ; ".join(cmds)
        self.shell.sendline(cmd)
        self.shell.expect(self._ps1, timeout=self.request.startup_timeout)
        output = _strip_control_chars(self.shell.before)  # type: ignore
        return CreateBashSessionResponse(output=output)

    def _eat_following_output(self, timeout: float = 0.5) -> str:
        """Drain output for ``timeout`` seconds so the next command starts clean."""
        time.sleep(timeout)
        try:
            output = self.shell.read_nonblocking(timeout=0.1)
        except pexpect.TIMEOUT:
            return ""
        return _strip_control_chars(output)

    async def interrupt(self, action: BashInterruptAction) -> BashObservation:
        """SIGINT the running foreground command; fall back to background-and-kill."""
        output = ""
        for _ in range(action.n_retry):
            self.shell.sendintr()
            expect_strings = action.expect + [self._ps1]
            try:
                expect_index = self.shell.expect(expect_strings, timeout=action.timeout)  # type: ignore
                matched_expect_string = expect_strings[expect_index]
            except Exception:
                await asyncio.sleep(0.2)
                continue
            output += _strip_control_chars(self.shell.before)  # type: ignore
            output += self._eat_following_output()
            output = output.strip()
            return BashObservation(output=output, exit_code=0, expect_string=matched_expect_string)
        # Last resort: stop the job, kill it in background.
        try:
            self.shell.sendcontrol("z")
            self.shell.expect(expect_strings, timeout=action.timeout)
            output += self.shell.before
            self.shell.sendline("kill -9 %1")
            expect_index = self.shell.expect(expect_strings, timeout=action.timeout)  # type: ignore
            matched_expect_string = expect_strings[expect_index]
            output += self.shell.before
            output += self._eat_following_output()
            output = output.strip()
            return BashObservation(output=output, exit_code=0, expect_string=matched_expect_string)
        except pexpect.TIMEOUT:
            raise pexpect.TIMEOUT("Failed to interrupt session") from None

    async def run(self, action: BashAction | BashInterruptAction) -> BashObservation:
        """Dispatch a bash action."""
        if self.shell is None:
            raise SessionNotInitializedError("shell not initialized")
        if isinstance(action, BashInterruptAction):
            return await self.interrupt(action)
        if action.is_interactive_command or action.is_interactive_quit:
            return await self._run_interactive(action)
        r = await self._run_normal(action)
        if action.check == "raise" and r.exit_code != 0:
            msg = f"Command {action.command!r} failed with exit code {r.exit_code}. Here is the output:\n{r.output!r}"
            if action.error_msg:
                msg = f"{action.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return r

    async def _run_interactive(self, action: BashAction) -> BashObservation:
        """Run an interactive action: skip exit-code retrieval and PS1 reseeking."""
        assert self.shell is not None
        self.shell.sendline(action.command)
        expect_strings = action.expect + [self._ps1]
        try:
            expect_index = self.shell.expect(expect_strings, timeout=action.timeout)  # type: ignore
            matched_expect_string = expect_strings[expect_index]
        except pexpect.TIMEOUT as e:
            raise CommandTimeoutError(
                f"timeout after {action.timeout} seconds while running command {action.command!r}"
            ) from e
        output: str = _strip_control_chars(self.shell.before)  # type: ignore
        if action.is_interactive_quit:
            assert not action.is_interactive_command
            self.shell.setecho(False)
            self.shell.waitnoecho()
            self.shell.sendline(f"stty -echo; echo '{self._UNIQUE_STRING}'")
            self.shell.expect(self._UNIQUE_STRING, timeout=1)
            self.shell.expect(self._ps1, timeout=1)
        else:
            # Interactive command often turns echo back on inside the shell.
            output = output.lstrip().removeprefix(action.command).strip()

        return BashObservation(output=output, exit_code=0, expect_string=matched_expect_string)

    async def _run_normal(self, action: BashAction) -> BashObservation:
        """Run a normal bash action: syntax check -> execute -> capture exit code."""
        action = deepcopy(action)

        assert self.shell is not None
        _check_bash_command(action.command)

        # 1. Send the command (joined with ; to avoid PS1 noise between subcommands).
        fallback_terminator = False
        try:
            individual_commands = _split_bash_command(action.command)
        except Exception as e:
            self.logger.debug(
                f"bashlex could not split command (falling back to tail-marker mode): {type(e).__name__}: {e}"
            )
            action.command += f"\n TMPEXITCODE=$? ; sleep 0.1; echo -n '{self._UNIQUE_STRING}' ; (exit $TMPEXITCODE)"
            fallback_terminator = True
        else:
            action.command = " ; ".join(individual_commands)
        self.shell.sendline(action.command)
        if not fallback_terminator:
            expect_strings = action.expect + [self._ps1]
        else:
            expect_strings = [self._UNIQUE_STRING]
        try:
            expect_index = self.shell.expect(expect_strings, timeout=action.timeout)  # type: ignore
            matched_expect_string = expect_strings[expect_index]
        except pexpect.TIMEOUT as e:
            raise CommandTimeoutError(
                f"timeout after {action.timeout} seconds while running command {action.command!r}"
            ) from e
        output: str = _strip_control_chars(self.shell.before)  # type: ignore

        # 2. Capture exit code.
        if action.check == "ignore":
            return BashObservation(output=output, exit_code=None, expect_string=matched_expect_string)

        try:
            _exit_code_prefix = "EXITCODESTART"
            _exit_code_suffix = "EXITCODEEND"
            self.shell.sendline(f"\necho {_exit_code_prefix}$?{_exit_code_suffix}")
            try:
                self.shell.expect(_exit_code_suffix, timeout=1)
            except pexpect.TIMEOUT:
                raise NoExitCodeError("timeout while getting exit code") from None
            exit_code_raw: str = _strip_control_chars(self.shell.before)  # type: ignore
            exit_code = re.findall(f"{_exit_code_prefix}([0-9]+)", exit_code_raw)
            if len(exit_code) != 1:
                raise NoExitCodeError(
                    f"failed to parse exit code from output {exit_code_raw!r} "
                    f"(command: {action.command!r}, matches: {exit_code})"
                )
            output += exit_code_raw.split(_exit_code_prefix)[0]
            exit_code = int(exit_code[0])
            # Drain the trailing PS1.
            try:
                self.shell.expect(self._ps1, timeout=0.1)
            except pexpect.TIMEOUT:
                raise CommandTimeoutError("Timeout while getting PS1 after exit code extraction") from None
            output = output.replace(self._UNIQUE_STRING, "").replace(self._ps1, "")
        except Exception:
            if action.check == "raise":
                raise
            exit_code = None
        return BashObservation(output=output, exit_code=exit_code, expect_string=matched_expect_string)

    async def close(self) -> CloseSessionResponse:
        if self._shell is None:
            return CloseBashSessionResponse()
        self.shell.close()
        self._shell = None
        return CloseBashSessionResponse()

    def interact(self) -> None:
        """Enter interactive mode."""
        self.shell.interact()


class LocalNativeRuntime(AbstractRuntime):
    """In-process runtime that drives bash via pexpect, no event-loop binding.

    This is the uni-agent flavored mirror of ``swerex.runtime.local.LocalRuntime``.
    """

    def __init__(self, *, run_id: str):
        self.run_id = run_id
        self._sessions: dict[str, Session] = {}
        self.logger = get_logger("local-native-runtime", run_id)

    @classmethod
    def from_config(cls, run_id: str) -> Self:
        return cls(run_id=run_id)

    @property
    def sessions(self) -> dict[str, Session]:
        return self._sessions

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        return IsAliveResponse(is_alive=True)

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        if request.session in self.sessions:
            raise SessionExistsError(f"session {request.session} already exists")
        if isinstance(request, CreateBashSessionRequest):
            session = BashSession(request, run_id=self.run_id)
        else:
            raise ValueError(f"unknown session type: {request!r}")
        self.sessions[request.session] = session
        return await session.start()

    async def run_in_session(self, action: Action) -> Observation:
        if action.session not in self.sessions:
            raise SessionDoesNotExistError(f"session {action.session!r} does not exist")
        return await self.sessions[action.session].run(action)

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        if request.session not in self.sessions:
            raise SessionDoesNotExistError(f"session {request.session!r} does not exist")
        out = await self.sessions[request.session].close()
        del self.sessions[request.session]
        return out

    async def execute(self, command: Command) -> CommandResponse:
        """Execute a one-shot command (no session). Subprocess-based."""
        try:
            result = subprocess.run(
                command.command,
                shell=command.shell,
                timeout=command.timeout,
                env=command.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if command.merge_output_streams else subprocess.PIPE,
                cwd=command.cwd,
            )
            r = CommandResponse(
                stdout=result.stdout.decode(errors="backslashreplace"),
                stderr=result.stderr.decode(errors="backslashreplace") if result.stderr is not None else "",
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as e:
            raise CommandTimeoutError(f"Timeout ({command.timeout}s) exceeded while running command") from e
        if command.check and result.returncode != 0:
            msg = (
                f"Command {command.command!r} failed with exit code {result.returncode}. "
                f"Stdout:\n{r.stdout!r}\nStderr:\n{r.stderr!r}"
            )
            if command.error_msg:
                msg = f"{command.error_msg}: {msg}"
            raise NonZeroExitCodeError(msg)
        return r

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        content = Path(request.path).read_text(encoding=request.encoding, errors=request.errors)
        return ReadFileResponse(content=content)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        Path(request.path).parent.mkdir(parents=True, exist_ok=True)
        Path(request.path).write_text(request.content)
        return WriteFileResponse()

    async def upload(self, request: UploadRequest) -> UploadResponse:
        if Path(request.source_path).is_dir():
            shutil.copytree(request.source_path, request.target_path, dirs_exist_ok=True)
        else:
            shutil.copy(request.source_path, request.target_path)
        return UploadResponse()

    async def close(self) -> CloseResponse:
        for session in self.sessions.values():
            await session.close()
        return CloseResponse()
