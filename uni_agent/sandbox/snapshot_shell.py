"""Stateful shell that runs each command in one exec, persisting cwd and exported env.

Every command starts a fresh bash through ``snapshot_runner.sh`` inside the sandbox:
the runner restores the last committed snapshot, runs the command with bounded
output, and commits a new snapshot only when the command returns normally.
"""

from __future__ import annotations

import asyncio
import math
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ExecResult

if TYPE_CHECKING:
    from .base import Sandbox

#: Per-stream cap applied inside the sandbox. Observations keep the first 100k
#: characters and UTF-8 needs at most 4 bytes each, so the model sees the same text.
OUTPUT_LIMIT = 512 * 1024
#: Larger commands are uploaded instead of risking the single-argv length limit.
_ARGV_LIMIT = 16_000
_RUNNER = Path(__file__).with_name("snapshot_runner.sh").read_text()
_TRAILER = "\nUASH1 "


class ShellTimeoutError(TimeoutError):
    """The command or its transport timed out; ``result`` holds the partial output."""

    def __init__(self, message: str, result: ExecResult):
        super().__init__(message)
        self.result = result


class SnapshotStateError(RuntimeError):
    """The state could not be restored or confirmed; the handle remains usable."""

    def __init__(self, message: str, result: ExecResult):
        super().__init__(message)
        self.result = result


class SnapshotShell:
    """Serialize calls; an uncertain outcome fences the state directory, not the session.

    After a transport failure, cancellation or lost runner the previous command may
    still be running, so the next command moves to a fresh directory seeded with the
    last committed state. A stale runner can then only write to the abandoned one.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        generation_attr: str | None = None,
        output_limit: int = OUTPUT_LIMIT,
        interactive: bool = False,
        prelude: str = "",
    ):
        """``generation_attr`` names a sandbox attribute that changes when the sandbox is
        stopped or replaced; the handle then refuses to run. ``interactive`` captures the
        initial state from ``bash -i`` so bashrc applies as in a terminal; ``prelude``
        runs before that capture."""
        self._sandbox = sandbox
        self._env = dict(env or {})
        self._cwd = cwd
        self._generation_attr = generation_attr
        self._generation = self._current_generation()
        self._limit = output_limit
        self._interactive = interactive
        self._prelude = f"{prelude}; " if prelude else ""
        self._root: str | None = None
        self._stale_roots: list[str] = []
        self._bash: str | None = None
        self._lock = asyncio.Lock()
        self._status = "new"  # new | open | fenced | broken | closed
        self._strict = True
        self._notice = ""

    def _current_generation(self) -> object:
        return getattr(self._sandbox, self._generation_attr, None) if self._generation_attr else None

    async def open(self) -> None:
        """Capture the initial state now instead of on the first :meth:`run`."""
        async with self._lock:
            if self._status == "new":
                await self._prepare()
            elif self._status != "open":
                raise RuntimeError(f"shell is {self._status}; open a new handle")

    async def _prepare(self) -> None:
        previous = self._root if self._status == "fenced" else None
        root = f"/tmp/.uni-agent-shell-{uuid.uuid4().hex}"
        q = shlex.quote(root)
        exports = "".join(f"export {shlex.quote(k)}={shlex.quote(v)}; " for k, v in self._env.items())
        cd = f"cd -- {shlex.quote(self._cwd)} && " if self._cwd else ""
        capture = f"{{ builtin export -p; builtin printf 'cd -- %q || return\\n' \"$PWD\"; }} > {q}/state"
        if self._interactive:
            # bashrc functions and aliases (e.g. conda's) go to a static profile the
            # runner sources before the state; only exports and cwd change afterwards.
            capture += (
                f" && {{ builtin alias -p; builtin declare -f; "
                f"builtin echo 'builtin shopt -s expand_aliases'; }} > {q}/profile"
            )
            capture = f'"$BASH" -ic {shlex.quote(capture)} </dev/null >/dev/null 2>&1; [ -s {q}/profile ]'
        fresh = f"( {self._prelude}{cd}{exports}{capture} ) && echo fresh"
        sweep: list[str] = []
        if previous:
            # Renaming first is the fence: a stale runner addresses files by the old
            # path, so it can no longer commit, and the copy below cannot race with it.
            # The dead copy is kept until this rotation is confirmed, so a retry after a
            # lost reply still finds the committed state; earlier leftovers go now.
            p, dead = shlex.quote(previous), shlex.quote(previous + ".dead")
            sweep = list(self._stale_roots)
            rm = f" && {{ rm -rf {' '.join(map(shlex.quote, sweep))} 2>/dev/null || true; }}" if sweep else ""
            state = (
                f"{{ if {{ mv -- {p} {dead} 2>/dev/null || [ -d {dead} ]; }} && cp {dead}/state {q}/state 2>/dev/null "
                f"&& {{ [ ! -e {dead}/profile ] || cp {dead}/profile {q}/profile; }}; "
                f"then echo restored; else {fresh}; fi{rm}; }}"
            )
        else:
            state = fresh
        script = f'umask 077; mkdir -p {q} && printf %s "$1" > {q}/runner.sh && {state} && printf %s "$BASH"'
        try:
            result = await asyncio.wait_for(self._sandbox._exec(["bash", "-c", script, "_", _RUNNER], timeout=30), 30)
            if result.exit_code:
                raise RuntimeError(f"failed to initialize shell: {result.stderr}")
            origin, _, bash = result.stdout.partition("\n")
            if origin not in ("fresh", "restored") or not bash.startswith("/"):
                raise RuntimeError("invalid shell initialization output")
        except BaseException:
            self._stale_roots.append(root)
            if previous is None:
                self._status = "broken"
            raise
        if previous:
            self._stale_roots = [r for r in self._stale_roots if r not in sweep] + [previous + ".dead"]
            if origin == "fresh":
                self._notice = "[shell state was lost and has been reinitialized]\n"
        self._root, self._bash, self._status = root, bash, "open"

    async def run(self, command: str, *, timeout: float = 120.0) -> ExecResult:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        return await asyncio.wait_for(self._run(command, timeout=timeout), timeout + 5)

    async def _run(self, command: str, *, timeout: float) -> ExecResult:
        if self._status == "new":
            await self.open()
        # Queue wait is bounded, but timing out before dispatch leaves state intact.
        async with self._lock:
            if self._status in ("broken", "closed"):
                raise RuntimeError(f"shell is {self._status}; open a new handle")
            if self._current_generation() is not self._generation:
                self._status = "broken"
                raise RuntimeError("sandbox was stopped or replaced; open a new handle")
            try:
                if self._status == "fenced":
                    await self._prepare()
                raw = await self._dispatch(command, timeout)
            except asyncio.CancelledError:
                self._status = "fenced"
                raise
            except (ShellTimeoutError, SnapshotStateError):
                raise
            except Exception as exc:
                self._status = "fenced"
                partial = getattr(exc, "result", None) or ExecResult(-1, "", "")
                if isinstance(exc, TimeoutError) or self._sandbox._is_timeout_error(exc):
                    raise ShellTimeoutError("shell execution deadline expired; remote state unknown", partial) from exc
                if not await self._sandbox_alive():
                    self._status = "broken"
                    raise
                raise SnapshotStateError(
                    f"shell transport failed; remote state unknown: {exc}", ExecResult(-1, "", str(exc))
                ) from exc
            return self._interpret(raw, timeout)

    async def _sandbox_alive(self) -> bool:
        try:
            return bool(await asyncio.wait_for(self._sandbox.is_alive(), 30))
        except Exception:
            return False

    async def _dispatch(self, command: str, timeout: float) -> ExecResult:
        assert self._root is not None and self._bash is not None
        rid = uuid.uuid4().hex[:16]
        argv = [
            self._bash,
            f"{self._root}/runner.sh",
            self._root,
            rid,
            f"{timeout:.3f}",
            str(self._limit),
            "1" if self._strict else "0",
        ]
        if len(command.encode()) > _ARGV_LIMIT:
            await self._sandbox.write_file(f"{self._root}/cmd.{rid}", command)
        else:
            argv.append(command)
        raw = await self._sandbox._exec(argv, timeout=timeout + 3)
        if raw.exit_code == -1:
            self._status = "fenced"
            raise ShellTimeoutError("shell execution deadline expired; remote state unknown", raw)
        if raw.exit_code != 0 or _TRAILER not in raw.stderr:
            self._status = "fenced"
            raise SnapshotStateError("shell runner failed; remote state unknown", raw)
        return raw

    def _interpret(self, raw: ExecResult, timeout: float) -> ExecResult:
        idx = raw.stderr.rfind(_TRAILER)
        rc, committed, timed, ran, out_bytes, err_bytes = (int(x) for x in raw.stderr[idx + 1 :].split()[1:7])
        stdout, stderr = raw.stdout, raw.stderr[:idx]
        if out_bytes > self._limit:
            stdout += f"\n[stdout truncated: first {self._limit} of {out_bytes} bytes shown]"
        if err_bytes > self._limit:
            stderr += f"\n[stderr truncated: first {self._limit} of {err_bytes} bytes shown]"
        result = ExecResult(rc, stdout, self._notice + stderr)
        self._notice = ""
        if timed:
            # The runner killed the command's process group before any commit.
            raise ShellTimeoutError(f"command timed out after {timeout}s", result)
        if not ran:
            self._strict = False
            raise SnapshotStateError(
                "could not restore the shell state (its working directory may have been removed); "
                "the command was not executed and the next one starts in the home directory",
                result,
            )
        self._strict = True
        if not committed:
            result.stderr += (
                "\n[shell state not saved: the command replaced the shell; "
                "cwd and exports from the previous command are kept]"
            )
        return result

    async def close(self) -> None:
        async with self._lock:
            if self._status == "closed":
                return
            self._status = "closed"
            roots = [r for r in (self._root, *self._stale_roots) if r]
            if not roots:
                return
            result = await asyncio.wait_for(self._sandbox._exec(["rm", "-rf", *roots], timeout=10), 10)
            if result.exit_code:
                raise RuntimeError(f"shell cleanup failed: {result.stderr}")
