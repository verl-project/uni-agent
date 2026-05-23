"""Async wrapper around ``lark-cli event consume <EventKey>``.

Spawns the consumer as a subprocess and exposes it as an async iterator
of decoded JSON events. Honours the lark-event subprocess contract:

- Wait for ``[event] ready event_key=<key>`` on stderr in ``start()``
  before iteration begins.
- Keep stdin open via an unclosed pipe; ``lark-cli`` treats stdin EOF
  as a graceful shutdown trigger.
- Stop via stdin close → SIGTERM → SIGKILL ladder so the server-side
  subscription unsubscribes cleanly.

``command_prefix`` (e.g. ``["docker", "exec", "-i", "<container>"]``)
lets us call a containerised ``lark-cli`` so the lark identity matches
the one the agent uses to reply.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence

_READY_MARKER = b"[event] ready"


class LarkEventListenerError(RuntimeError):
    """Raised when the underlying ``lark-cli event consume`` subprocess
    fails to start, never becomes ready, or exits abnormally.
    """


class LarkEventListener:
    """Async iterator over decoded events from a single ``EventKey``."""

    def __init__(
        self,
        event_key: str,
        *,
        as_identity: str = "bot",
        jq: str | None = None,
        command_prefix: Sequence[str] | None = None,
    ):
        self.event_key = event_key
        self.as_identity = as_identity
        self.jq = jq
        self.command_prefix: list[str] = list(command_prefix) if command_prefix else []
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_pump_task: asyncio.Task | None = None

    async def start(self, *, ready_timeout: float = 30.0) -> None:
        """Spawn ``lark-cli event consume`` and block until ready."""
        argv = [
            *self.command_prefix,
            "lark-cli",
            "event",
            "consume",
            self.event_key,
            "--as",
            self.as_identity,
        ]
        if self.jq:
            argv.extend(["--jq", self.jq])
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        ready_wait = asyncio.create_task(self._await_ready())
        try:
            await asyncio.wait_for(ready_wait, timeout=ready_timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise LarkEventListenerError(
                f"`lark-cli event consume {self.event_key}` did not emit "
                f"'[event] ready' within {ready_timeout}s. Check "
                f"`lark-cli auth status --as {self.as_identity}` and that "
                f"the bot is subscribed to {self.event_key}."
            ) from None

        self._stderr_pump_task = asyncio.create_task(self._pump_stderr())

    async def _await_ready(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                rc = await self.proc.wait()
                raise LarkEventListenerError(
                    f"`lark-cli event consume {self.event_key}` exited before becoming ready (rc={rc})"
                )
            _write_stderr(line)
            if _READY_MARKER in line:
                return

    async def _pump_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            _write_stderr(line)

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[dict]:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event

    async def stop(self) -> None:
        """Shut the consumer down cleanly (stdin close → SIGTERM → SIGKILL)."""
        if self.proc is None:
            return
        if self.proc.returncode is None:
            if self.proc.stdin is not None and not self.proc.stdin.is_closing():
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
        if self._stderr_pump_task is not None:
            self._stderr_pump_task.cancel()
            try:
                await self._stderr_pump_task
            except asyncio.CancelledError:
                pass
        self.proc = None


def _write_stderr(b: bytes) -> None:
    try:
        sys.stderr.write(b.decode("utf-8", errors="replace"))
        sys.stderr.flush()
    except Exception:
        pass


async def fetch_bot_open_id(command_prefix: Sequence[str] | None = None) -> str:
    """Resolve the bot's own ``open_id`` via the Lark Open API.

    Used to build the jq filter that drops the bot's own reply messages
    (the platform may re-deliver them as ``im.message.receive_v1`` events)
    before they reach the agent loop. ``command_prefix`` mirrors
    :class:`LarkEventListener` so we resolve the id with the same
    lark-cli identity that subscribes events.
    """
    prefix = list(command_prefix) if command_prefix else []
    argv = [*prefix, "lark-cli", "api", "GET", "/open-apis/bot/v3/info", "--as", "bot"]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise LarkEventListenerError(
            f"`lark-cli api GET /open-apis/bot/v3/info --as bot` failed "
            f"(rc={proc.returncode}):\n"
            f"{err.decode('utf-8', errors='replace')}"
        )
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        preview = out[:300].decode("utf-8", errors="replace")
        raise LarkEventListenerError(f"could not parse bot info JSON: {e}\noutput preview: {preview!r}") from e
    for path in (("bot", "open_id"), ("data", "bot", "open_id"), ("open_id",)):
        cur = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.startswith("ou_"):
            return cur
    raise LarkEventListenerError(f"could not find bot.open_id in `lark-cli` response: {data!r}")
