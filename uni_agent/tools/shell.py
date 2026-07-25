"""``shell`` tool + the stateful shell session it owns.

The agent-facing unit is :class:`ShellTool` (registry key ``stateful_shell``, seen
by the model as ``shell``): it holds a live :class:`Shell` -- either
:class:`SandboxShell` (sandbox ``open_shell``, e.g. openyuanrong) or
:class:`TmuxShell` (tmux over one-shot exec) -- so cwd / exports persist
across calls.

:class:`TmuxShell` is the fallback for backends that only expose one-shot
exec. Providers with ``supports_shell`` skip tmux entirely.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import re
import shlex
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from uni_agent.sandbox import Sandbox, SandboxBackend
from .base import Tool, ToolError, ToolResult, register_tool

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CommandResult:
    """Outcome of one command run in a shell session."""

    command_id: int
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    start_time: float
    end_time: float
    timed_out: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@runtime_checkable
class Shell(Protocol):
    """Minimal session surface used by :class:`ShellTool`."""

    async def start(self) -> None: ...

    async def run(self, command: str, *, timeout: float = 120.0) -> CommandResult: ...

    async def close(self) -> None: ...


class SandboxShell:
    """:class:`Shell` backed by sandbox ``open_shell()`` (native provider session)."""

    def __init__(self, handle: Any):
        self._handle = handle
        self._counter = 0

    async def start(self) -> None:
        return None

    async def run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        start = time.monotonic()
        self._counter += 1
        cid = self._counter
        timed_out = False
        code: int | None = None
        stdout = ""
        stderr = ""
        try:
            res = await self._handle.run(command, timeout=timeout)
            code = res.exit_code
            stdout = res.stdout or ""
            stderr = res.stderr or ""
        except Exception as exc:
            name = type(exc).__name__
            if name == "CommandTimeoutError" or isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                timed_out = True
                stderr = str(exc)
            else:
                raise
        end = time.monotonic()
        return CommandResult(
            command_id=cid,
            command=command,
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            start_time=start,
            end_time=end,
            timed_out=timed_out,
        )

    async def close(self) -> None:
        await self._handle.close()


async def open_shell_session(
    backend: SandboxBackend,
    *,
    env: dict[str, str] | None = None,
    width: int = 120,
    height: int = 40,
) -> Shell:
    """Prefer a native shell; fall back to tmux-over-exec."""
    if isinstance(backend, Sandbox) and backend.supports_shell:
        handle = await backend.open_shell(env=env)
        logger.info("stateful_shell using native shell (%s)", type(backend).__name__)
        session: Shell = SandboxShell(handle)
        await session.start()
        return session

    logger.info("stateful_shell using tmux shell (%s)", type(backend).__name__)
    session = TmuxShell(backend, width=width, height=height, env=env)
    await session.start()
    return session


def _capture_wrapper(command: str, out: str, err: str, rc: str, *, signal: str | None, sock: str) -> str:
    """Build the shell line running ``command`` under the file-capture protocol."""
    b64 = base64.b64encode(command.encode()).decode("ascii")
    line = (
        f"eval \"$(printf %s '{b64}' | base64 -d)\" "
        f"> {shlex.quote(out)} 2> {shlex.quote(err)}; "
        f'__rc=$?; printf %s "$__rc" > {shlex.quote(rc)}.part '
        f"&& mv {shlex.quote(rc)}.part {shlex.quote(rc)}"
    )
    if signal is not None:
        line += f"; tmux -S {shlex.quote(sock)} wait -S {shlex.quote(signal)}"
    return line


# Best-effort tmux install when the image lacks it: first package manager on PATH,
# non-interactive, one exec. apt needs an index refresh first (minimal images ship
# an empty lists dir); `|| true` tolerates a flaky mirror if tmux still resolves.
_INSTALL_TMUX = r"""
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux
elif command -v yum >/dev/null 2>&1; then yum install -y tmux
elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux
elif command -v pacman >/dev/null 2>&1; then pacman -Sy --noconfirm tmux
elif command -v zypper >/dev/null 2>&1; then zypper install -y -n tmux
else
  echo "no supported package manager to install tmux" >&2
  exit 127
fi
""".strip()


class TmuxShell:
    """:class:`Shell` backed by a detached tmux session, driven via one-shot exec.

    Fallback for providers without ``open_shell``. Every tmux verb goes through
    :meth:`SandboxBackend.exec`; only ``tmux`` needs to be present in the image.
    """

    def __init__(
        self,
        backend: SandboxBackend,
        *,
        session_id: str | None = None,
        width: int = 120,
        height: int = 40,
        shell: str = "bash",
        env: dict[str, str] | None = None,
    ):
        self.backend = backend
        self.session_id = session_id or f"uni-{uuid.uuid4().hex[:8]}"
        self.width = width
        self.height = height
        self._shell = shell
        self._env = dict(env or {})
        self._dir = f"/tmp/uni-agent-shell/{self.session_id}"
        self._sock = f"{self._dir}/tmux.sock"
        self._counter = 0

    # ----- helpers -----
    def _tmux(self, *args: str) -> list[str]:
        """A tmux argv pinned to our private server socket via ``-S``."""
        return ["tmux", "-S", self._sock, *args]

    def _paths(self, cid: int) -> tuple[str, str, str]:
        base = f"{self._dir}/cmd_{cid}"
        return f"{base}.out", f"{base}.err", f"{base}.rc"

    def _chan(self, cid: int) -> str:
        return f"uniagent-{self.session_id}-{cid}"

    async def _read_text(self, path: str) -> str:
        res = await self.backend.exec(["cat", path])
        return res.stdout if res.exit_code == 0 else ""

    # ----- lifecycle -----
    async def start(self) -> None:
        # tmux backs the channel; install it on first use if the image lacks it.
        if (await self.backend.exec(["tmux", "-V"])).exit_code != 0:
            res = await self.backend.exec_shell(_INSTALL_TMUX, timeout=300.0)
            if (await self.backend.exec(["tmux", "-V"])).exit_code != 0:
                raise RuntimeError(
                    "tmux is not available and could not be installed in the "
                    f"sandbox (installer exit {res.exit_code}): "
                    f"{(res.stderr or '').strip()[:500]}"
                )
        await self.backend.exec(["mkdir", "-p", self._dir])
        # Launch the shell under ``env K=V ... <shell>`` so the channel inherits its
        # env without echoing exports into the pane.
        launch = [
            "env",
            *(f"{key}={value}" for key, value in self._env.items()),
            self._shell,
        ] if self._env else [self._shell]
        res = await self.backend.exec(
            self._tmux(
                "new-session", "-d",
                "-s", self.session_id,
                "-x", str(self.width),
                "-y", str(self.height),
                *launch,
            )
        )
        if res.exit_code != 0:
            raise RuntimeError(f"failed to start tmux session: {res.stderr.strip()}")
        # Large scrollback so capture_pane(entire=True) can return full history.
        await self.backend.exec(
            self._tmux("set-option", "-g", "history-limit", "1000000")
        )

    async def close(self) -> None:
        await self.backend.exec(self._tmux("kill-session", "-t", self.session_id))
        await self.backend.exec(["rm", "-rf", self._dir])

    async def observe(self) -> ToolResult:
        return ToolResult(text=await self.capture_pane())

    # ----- shell actions -----
    async def start_command(self, command: str) -> int:
        cid = self._counter + 1
        self._counter = cid
        out, err, rc = self._paths(cid)
        line = _capture_wrapper(command, out, err, rc, signal=self._chan(cid), sock=self._sock)
        # Type the wrapped line then press Enter. `--` ends option parsing so a
        # command starting with `-` is still typed literally.
        res = await self.backend.exec(
            self._tmux("send-keys", "-t", self.session_id, "--", line, "Enter")
        )
        if res.exit_code != 0:
            raise RuntimeError(f"failed to inject command: {res.stderr.strip()}")
        return cid

    async def poll(self, command_id: int) -> int | None:
        _, _, rc = self._paths(command_id)
        res = await self.backend.exec(["cat", rc])
        if res.exit_code != 0:
            return None  # exit-code file not written yet -> still running
        text = res.stdout.strip()
        return int(text) if text else None

    async def run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        start = time.monotonic()
        cid = await self.start_command(command)
        out, err, rc = self._paths(cid)
        chan = self._chan(cid)
        timed_out = False
        code: int | None = None

        while True:
            code = await self.poll(cid)
            if code is not None:
                break
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                await self.interrupt()
                code = await self.poll(cid)
                timed_out = code is None
                break
            # Event-driven wakeup: block on the command's tmux wait channel to
            # return the instant it signals. poll() stays the source of truth, so a
            # lost signal costs one bounded slice, not a hang.
            slice_s = max(0.1, min(2.0, timeout - elapsed))
            await self.backend.exec_shell(
                f"timeout {slice_s} tmux -S {shlex.quote(self._sock)} wait {shlex.quote(chan)} 2>/dev/null || true",
                timeout=slice_s + 10,
            )

        end = time.monotonic()
        return CommandResult(
            command_id=cid,
            command=command,
            exit_code=code,
            stdout=await self._read_text(out),
            stderr=await self._read_text(err),
            start_time=start,
            end_time=end,
            timed_out=timed_out,
        )

    async def send_keys(self, keys: str | list[str]) -> None:
        keys_list = [keys] if isinstance(keys, str) else list(keys)
        res = await self.backend.exec(
            self._tmux("send-keys", "-t", self.session_id, "--", *keys_list)
        )
        if res.exit_code != 0:
            raise RuntimeError(f"send_keys failed: {res.stderr.strip()}")

    async def interrupt(self) -> None:
        await self.backend.exec(
            self._tmux("send-keys", "-t", self.session_id, "C-c")
        )

    async def capture_pane(self, *, entire: bool = False) -> str:
        args = self._tmux("capture-pane", "-p")
        if entire:
            args += ["-S", "-"]
        args += ["-t", self.session_id]
        res = await self.backend.exec(args)
        return res.stdout

    async def resize(self, *, width: int, height: int) -> None:
        res = await self.backend.exec(
            self._tmux("resize-window", "-t", self.session_id,
                       "-x", str(width), "-y", str(height))
        )
        if res.exit_code != 0:
            raise RuntimeError(f"resize failed: {res.stderr.strip()}")
        self.width, self.height = width, height


# ===================== the agent-facing tool =====================

DESCRIPTION = "Execute a bash command in the terminal."

# CSI escape sequences (colors, cursor moves, line erases). Tools like pytest/pip emit
# these even into a pipe, so strip them (plus \r redraws) from what the model sees.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text).replace("\r", "")


def _format(stdout: str, stderr: str, exit_code: int | None) -> str:
    out = _clean(stdout or "").rstrip()
    err = _clean(stderr or "").rstrip()
    header = f"[exit code: {exit_code if exit_code is not None else 'unknown'}]"
    parts = [header, "[stdout]", out if out else "(empty)"]
    if err:
        parts += ["[stderr]", err]
    return "\n".join(parts)


def _format_timeout(timeout: float, stdout: str, stderr: str) -> str:
    """Observation for a command the channel had to interrupt (soft timeout).

    Mirrors the training-time message: say what happened, steer toward a faster /
    non-interactive command, and still surface partial output. The length cap is
    applied centrally later by :meth:`ToolResult.to_observation`.
    """
    parts = [
        f"The command was cancelled because it took more than {timeout:g} seconds. "
        "Please try a different command that completes more quickly. A common cause "
        "is a command that is interactive or waits for input -- this environment "
        "cannot provide input, so such a command never completes."
    ]
    out = _clean(stdout or "").rstrip()
    err = _clean(stderr or "").rstrip()
    if out:
        parts += ["[partial stdout]", out]
    if err:
        parts += ["[partial stderr]", err]
    return "\n".join(parts)


class ShellArguments(BaseModel):
    command: str = Field(description="The command to execute.")


class ShellToolConfig(BaseModel):
    """Construction kwargs for the shell tool (the ``shell`` entry's kwargs)."""

    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables exported in the shell channel "
        "(shell-ergonomic defaults like PAGER=cat; not cross-cutting secrets).",
    )
    command_timeout: float = Field(
        default=180.0, description="Per-command timeout in seconds."
    )
    width: int = Field(default=120, description="Terminal width (columns).")
    height: int = Field(default=40, description="Terminal height (rows).")

    model_config = ConfigDict(extra="forbid")


@register_tool("stateful_shell")
class ShellTool(Tool):
    # Registry key is ``stateful_shell`` (config / TOOL_REGISTRY); the model
    # still sees this tool as ``shell`` via the explicit ``name`` below.
    name = "shell"
    description = DESCRIPTION
    args_model = ShellArguments
    config_model = ShellToolConfig

    config: ShellToolConfig

    def __init__(self, sandbox: SandboxBackend, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        self._shell: Shell | None = None

    async def _ensure_shell(self) -> Shell:
        if self._shell is None:
            self._shell = await open_shell_session(
                self.sandbox,
                width=self.config.width,
                height=self.config.height,
                env=self.config.env_vars,
            )
        return self._shell

    async def start(self) -> None:
        # Open the session now so first-use cost (native shell create, or tmux
        # install/session setup) happens up front rather than on the first command.
        await self._ensure_shell()

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        command = args.get("command")
        if not command or not str(command).strip():
            raise ToolError("Parameter `command` is required for shell.")

        command_timeout = timeout if timeout is not None else self.config.command_timeout
        shell = await self._ensure_shell()
        result = await shell.run(command, timeout=command_timeout)
        if result.timed_out:
            return ToolResult(
                text=_format_timeout(command_timeout, result.stdout, result.stderr),
                status="timeout",
            )
        return ToolResult(
            text=_format(result.stdout, result.stderr, result.exit_code),
            status="ok",
        )

    async def close(self) -> None:
        if self._shell is not None:
            await self._shell.close()
            self._shell = None
