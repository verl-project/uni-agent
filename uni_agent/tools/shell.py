"""``shell`` tool + the stateful shell channel it owns, in one place.

The agent-facing unit is :class:`ShellTool` (registered as ``shell``): it holds a
live shell channel -- a detached tmux shell -- opened lazily on first use and torn
down in :meth:`ShellTool.close`. Commands run in that one shell, so cwd / exports /
background jobs persist across calls (a ``cd`` sticks). The agent runs on the host;
only the command text crosses into the container.

The channel itself (:class:`ShellChannel`) is an implementation detail of the
tool, not an agent-facing layer:

* **File-capture protocol.** Every command redirects its ``stdout`` / ``stderr``
  to its own files and records an exit code; the shell is held by
  ``tmux new-session -d`` so the same pane can be screen-scraped (``capture-pane``),
  keyed (``send-keys``) and ``resize``-d.
* **Driven through one-shot exec.** Every tmux verb is issued through the backend's
  exec primitive (:class:`SandboxBackend`), so nothing resident is installed in the
  image beyond ``tmux``. A ``tmux wait -S`` completion signal, consumed by a
  blocking ``timeout … tmux wait``, wakes the waiter the instant a command
  finishes; a ``tmux -V`` preflight and a ``--`` end-of-options guard on
  ``send-keys`` keep commands starting with ``-`` literal.
"""

from __future__ import annotations

import base64
import dataclasses
import shlex
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..sandbox import SandboxBackend
from .base import Observation, Tool, ToolError, register_tool


@dataclasses.dataclass
class CommandResult:
    """Outcome of one command run in a shell channel."""

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


def _capture_wrapper(command: str, out: str, err: str, rc: str, *, signal: str | None) -> str:
    """Build the shell line that runs ``command`` under the file-capture protocol.

    The command body is base64-encoded so arbitrary quoting / newlines survive
    being typed into a live shell, then ``eval``-ed *in that same shell* so cwd
    and exported variables persist across calls (this statefulness is the whole
    point of the channel). ``stdout`` / ``stderr`` are redirected to their own
    files and the exit code is written last and atomically (``.part`` + ``mv``)
    so its presence is an unambiguous completion marker for :meth:`poll`.

    ``signal`` (tmux only) appends a ``tmux wait -S`` so a blocking waiter can be
    woken the instant the command finishes.
    """
    b64 = base64.b64encode(command.encode()).decode("ascii")
    line = (
        f"eval \"$(printf %s '{b64}' | base64 -d)\" "
        f"> {shlex.quote(out)} 2> {shlex.quote(err)}; "
        f'__rc=$?; printf %s "$__rc" > {shlex.quote(rc)}.part '
        f"&& mv {shlex.quote(rc)}.part {shlex.quote(rc)}"
    )
    if signal is not None:
        line += f"; tmux wait -S {shlex.quote(signal)}"
    return line


# Best-effort tmux install for ShellChannel.start() when the image lacks it: pick
# the first package manager on PATH and install non-interactively, all in a single
# exec to keep round-trips low.
_INSTALL_TMUX = r"""
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
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


class ShellChannel:
    """Stateful shell held by a detached ``tmux`` session, driven via exec.

    The persistent handle the shell tool owns -- not an agent-facing layer. Works
    on any provider that offers a one-shot exec (Modal, docker, ...). Every tmux
    verb (``new-session`` / ``send-keys`` / ``capture-pane`` / ``resize-window`` /
    ``wait``) is issued through :meth:`SandboxBackend.exec`, so nothing resident
    is installed into the task image beyond ``tmux`` itself.
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
        self._counter = 0

    # ----- helpers -----
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
        # Launch the pane's shell under ``env K=V ... <shell>`` so the whole
        # channel (and every command it spawns) inherits this channel's env,
        # without echoing exports into the pane.
        launch = [
            "env",
            *(f"{key}={value}" for key, value in self._env.items()),
            self._shell,
        ] if self._env else [self._shell]
        res = await self.backend.exec(
            [
                "tmux", "new-session", "-d",
                "-s", self.session_id,
                "-x", str(self.width),
                "-y", str(self.height),
                *launch,
            ]
        )
        if res.exit_code != 0:
            raise RuntimeError(f"failed to start tmux session: {res.stderr.strip()}")
        # Large scrollback so capture_pane(entire=True) can return full history.
        await self.backend.exec(
            ["tmux", "set-option", "-g", "history-limit", "1000000"]
        )

    async def close(self) -> None:
        await self.backend.exec(["tmux", "kill-session", "-t", self.session_id])
        await self.backend.exec(["rm", "-rf", self._dir])

    async def observe(self) -> Observation:
        text = await self.capture_pane()
        return Observation(
            text=text,
            structured={"session_id": self.session_id, "last_command_id": self._counter},
            meta={"transport": "tmux"},
        )

    # ----- shell actions -----
    async def start_command(self, command: str) -> int:
        cid = self._counter + 1
        self._counter = cid
        out, err, rc = self._paths(cid)
        line = _capture_wrapper(command, out, err, rc, signal=self._chan(cid))
        # Type the wrapped line then press Enter. `--` ends option parsing so a
        # command starting with `-` is still typed literally.
        res = await self.backend.exec(
            ["tmux", "send-keys", "-t", self.session_id, "--", line, "Enter"]
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

    async def run(self, command: str, *, timeout: float = 180.0) -> CommandResult:
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
            # Event-driven wakeup: block on the command's tmux wait channel so we
            # return the instant it signals. poll() above remains the source of
            # truth, so a wait-signal lost to a race only costs one bounded slice
            # instead of hanging.
            slice_s = max(0.1, min(2.0, timeout - elapsed))
            await self.backend.exec_shell(
                f"timeout {slice_s} tmux wait {shlex.quote(chan)} 2>/dev/null || true",
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
            ["tmux", "send-keys", "-t", self.session_id, "--", *keys_list]
        )
        if res.exit_code != 0:
            raise RuntimeError(f"send_keys failed: {res.stderr.strip()}")

    async def interrupt(self) -> None:
        await self.backend.exec(
            ["tmux", "send-keys", "-t", self.session_id, "C-c"]
        )

    async def capture_pane(self, *, entire: bool = False) -> str:
        args = ["tmux", "capture-pane", "-p"]
        if entire:
            args += ["-S", "-"]
        args += ["-t", self.session_id]
        res = await self.backend.exec(args)
        return res.stdout

    async def resize(self, *, width: int, height: int) -> None:
        res = await self.backend.exec(
            ["tmux", "resize-window", "-t", self.session_id,
             "-x", str(width), "-y", str(height)]
        )
        if res.exit_code != 0:
            raise RuntimeError(f"resize failed: {res.stderr.strip()}")
        self.width, self.height = width, height


# ===================== the agent-facing tool =====================

DESCRIPTION = "Execute a bash command in the terminal."
MAX_OUTPUT_LEN = 16000
_CLIP = "\n<response clipped: output truncated>"


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_LEN:
        return text
    return text[:MAX_OUTPUT_LEN] + _CLIP


def _format(stdout: str, stderr: str, exit_code: int | None, *, timed_out: bool = False) -> str:
    out = (stdout or "").rstrip()
    err = (stderr or "").rstrip()
    header = f"[exit code: {exit_code if exit_code is not None else 'unknown'}]"
    if timed_out:
        header += " [timed out]"
    parts = [header, "[stdout]", out if out else "(empty)"]
    if err:
        parts += ["[stderr]", err]
    return _truncate("\n".join(parts))


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


@register_tool("shell")
class ShellTool(Tool):
    name = "shell"
    description = DESCRIPTION
    args_model = ShellArguments
    config_model = ShellToolConfig

    config: ShellToolConfig

    def __init__(self, sandbox: SandboxBackend, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        self._shell: ShellChannel | None = None

    async def _ensure_shell(self) -> ShellChannel:
        if self._shell is None:
            shell = ShellChannel(
                self.sandbox,
                width=self.config.width,
                height=self.config.height,
                env=self.config.env_vars,
            )
            await shell.start()
            self._shell = shell
        return self._shell

    async def run(self, args: dict[str, Any]) -> Observation:
        command = args.get("command")
        if not command or not str(command).strip():
            raise ToolError("Parameter `command` is required for shell.")

        shell = await self._ensure_shell()
        result = await shell.run(command, timeout=self.config.command_timeout)
        return Observation(
            text=_format(
                result.stdout,
                result.stderr,
                result.exit_code,
                timed_out=result.timed_out,
            ),
            structured={"exit_code": result.exit_code, "timed_out": result.timed_out},
        )

    async def close(self) -> None:
        if self._shell is not None:
            await self._shell.close()
            self._shell = None
