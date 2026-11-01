"""Isolate blocking frontend command I/O while the owning process monitors lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

_TOKEN_ENV = "UNI_AGENT_SANDBOX_COMMAND_TOKEN"
_MAX_PACKET = 512 * 1024 * 1024
_CANCEL_GRACE = 2.0
logger = logging.getLogger(__name__)


def _encode(message: dict) -> bytes:
    payload = json.dumps(message, ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_PACKET:
        raise ValueError("Sandbox command response exceeds the transport limit")
    return struct.pack("!I", len(payload)) + payload


def _read_sync(stream) -> dict | None:
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError("Incomplete command-worker header")
    size = struct.unpack("!I", header)[0]
    if size > _MAX_PACKET:
        raise ValueError("Command-worker packet exceeds the transport limit")
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("Incomplete command-worker payload")
    return json.loads(payload)


async def _read_async(stream) -> dict:
    size = struct.unpack("!I", await stream.readexactly(4))[0]
    if size > _MAX_PACKET:
        raise ValueError("Command-worker packet exceeds the transport limit")
    return json.loads(await stream.readexactly(size))


def _worker_main() -> int:
    # This file is executed directly, so the helper imports only the pinned SDK,
    # not uni-agent's trainer/GPU stack. Credentials never enter argv or stdout.
    logging.disable(logging.CRITICAL)
    request = _read_sync(sys.stdin.buffer)
    if request is None:
        return 0
    token = os.environ[_TOKEN_ENV]
    secrets = {token}
    state_lock = threading.Lock()
    output_lock = threading.Lock()
    ready = threading.Event()
    finished = threading.Event()
    cancel_requested = threading.Event()
    cancel_done = threading.Event()
    handle = None
    sandbox = None

    def redact(message):
        with state_lock:
            values = tuple(secrets)
        for value in values:
            message = message.replace(value, "<REDACTED>")
        return message

    def emit(message):
        packet = _encode(message)
        with output_lock:
            sys.stdout.buffer.write(packet)
            sys.stdout.buffer.flush()

    def current_token():
        with state_lock:
            return os.environ[_TOKEN_ENV]

    def control_reader():
        while True:
            try:
                message = _read_sync(sys.stdin.buffer)
            except (OSError, ValueError):
                # Parent death can truncate a packet, not only close the pipe
                # between messages. It must terminate the blocking worker too.
                os._exit(1)
            if message is None:
                # The helper owns no sandbox. Parent death must not leave a
                # blocked SDK client process behind, even during a route wait.
                os._exit(0)
            if message["kind"] == "token":
                with state_lock:
                    os.environ[_TOKEN_ENV] = message["value"]
                    secrets.add(message["value"])
            elif message["kind"] == "cancel":
                cancel_requested.set()

    def cancel_command():
        cancel_requested.wait()
        while not ready.wait(0.05):
            if finished.is_set():
                return
        try:
            killed = handle.kill()
            emit({"kind": "cancel_ack", "killed": bool(killed)})
        except Exception as error:
            emit(
                {
                    "kind": "cancel_ack",
                    "killed": False,
                    "error_type": type(error).__name__,
                    "message": redact(str(error)),
                }
            )
        finally:
            cancel_done.set()

    threading.Thread(target=control_reader, daemon=True).start()
    threading.Thread(target=cancel_command, daemon=True).start()
    try:
        from importlib.metadata import version

        import yr_sandbox

        if version("openyuanrong-sandbox") != "0.10.2rc9":
            raise RuntimeError("The command worker requires openyuanrong-sandbox==0.10.2rc9")
        connection = yr_sandbox.ConnectionConfig(**request["connection"], token=token, token_provider=current_token)
        sandbox = yr_sandbox.Sandbox.from_id(request["sandbox_id"], connection=connection)
        handle = sandbox.commands.run(
            request["command"], background=True, envs=request["env"], cwd=request["workdir"], timeout=request["timeout"]
        )
        ready.set()
        emit({"kind": "started", "command_id": handle.id, "pid": handle.pid})
        deadline = time.monotonic() + request["timeout"]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cancel_requested.set()
                cancel_done.wait(_CANCEL_GRACE)
                raise TimeoutError(f"Command timed out after {request['timeout']}s")
            try:
                result = handle.wait(min(10.0, remaining))
            except TimeoutError:
                continue
            emit(
                {
                    "kind": "result",
                    "exit_code": result.exit_code,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                    "status": getattr(result.status, "value", str(result.status)),
                }
            )
            return 0
    except BaseException as error:
        emit(
            {
                "kind": "error",
                "error_type": type(error).__name__,
                "message": redact(str(error)),
                "is_timeout": isinstance(error, TimeoutError),
            }
        )
        return 1
    finally:
        finished.set()
        if sandbox is not None:
            sandbox.close()  # from_id is detached; the parent owns instance deletion.


async def _stop_process(process):
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def execute_command(sandbox, connection, command, *, env, workdir, timeout, check_interval, is_current):
    """Execute one command, reap its SDK worker, and preserve the provider result.

    The SDK's synchronous command waits can outlive instance deletion. Keeping
    those waits out of the event loop's thread pool lets confirmed deletion end
    both the await and the blocking client process. The parent remains the sole
    owner of the sandbox and its tunnel.
    """
    from .base import ExecResult, SandboxExecutionError, SandboxTerminatedError

    token = await asyncio.to_thread(connection.resolved_token)
    worker_env = dict(os.environ)
    worker_env[_TOKEN_ENV] = token
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=worker_env,
    )
    cancel_ack = asyncio.Event()
    cancel_result = {}
    stderr_tail = deque(maxlen=8)

    async def send(message):
        process.stdin.write(_encode(message))
        await process.stdin.drain()

    async def drain_stderr():
        while chunk := await process.stderr.read(8192):
            stderr_tail.append(chunk)

    async def receive():
        while True:
            message = await _read_async(process.stdout)
            if message["kind"] == "cancel_ack":
                cancel_result.update(message)
                cancel_ack.set()
            elif message["kind"] in {"result", "error"}:
                return message

    async def confirm_liveness():
        if not is_current():
            raise SandboxExecutionError(f"Sandbox {sandbox.sandbox_id} owner was closed")
        try:
            info = await asyncio.to_thread(sandbox.get_info)
        except Exception as error:
            if type(error).__name__ in {"SandboxNotFound", "SandboxNotFoundError"}:
                raise SandboxTerminatedError(f"Sandbox {sandbox.sandbox_id} was deleted") from error
            # A transport/authentication error does not prove instance death.
        else:
            if info.state != "running":
                raise SandboxTerminatedError(f"Sandbox {sandbox.sandbox_id} is {info.state}")

    reader = asyncio.create_task(receive())
    stderr_reader = asyncio.create_task(drain_stderr())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async def cancel_owned_command():
        if reader.done() or process.returncode is not None:
            return
        try:
            await send({"kind": "cancel"})
            acknowledgement = asyncio.create_task(cancel_ack.wait())
            try:
                await asyncio.wait(
                    {reader, acknowledgement}, timeout=_CANCEL_GRACE, return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_result.get("error_type"):
                    logger.warning(
                        "Command cancellation was not acknowledged in %s: %s",
                        sandbox.sandbox_id,
                        cancel_result["error_type"],
                    )
                elif not reader.done() and not cancel_ack.is_set():
                    logger.warning("Command cancellation acknowledgement timed out in %s", sandbox.sandbox_id)
            finally:
                acknowledgement.cancel()
                await asyncio.gather(acknowledgement, return_exceptions=True)
        except (BrokenPipeError, ConnectionResetError):
            pass

    try:
        fields = {
            key: getattr(connection, key)
            for key in ("server_address", "use_tls", "gateway_address", "gateway_use_tls", "verify_tls")
        }
        await send(
            {
                "kind": "run",
                "connection": fields,
                "sandbox_id": sandbox.sandbox_id,
                "command": command,
                "env": env,
                "workdir": workdir,
                "timeout": timeout,
            }
        )
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                await cancel_owned_command()
                raise TimeoutError(f"Command timed out after {timeout}s")
            await asyncio.wait({reader}, timeout=min(check_interval, remaining))
            if reader.done():
                try:
                    message = reader.result()
                except (asyncio.IncompleteReadError, ValueError) as error:
                    await confirm_liveness()
                    detail = b"".join(stderr_tail).decode("utf-8", errors="replace").replace(token, "<REDACTED>")
                    raise SandboxExecutionError(
                        "Sandbox command worker exited without a valid response: " + detail
                    ) from error
                if message["kind"] == "error":
                    if message["error_type"] in {"SandboxNotFound", "SandboxNotFoundError"}:
                        raise SandboxTerminatedError(f"Sandbox {sandbox.sandbox_id} was deleted")
                    await confirm_liveness()
                    if message.get("is_timeout"):
                        raise TimeoutError(message["message"])
                    raise SandboxExecutionError(f"{message['error_type']}: {message['message']}")
                if message["status"] == "TIMED_OUT":
                    raise TimeoutError(message["stderr"] or f"Command timed out after {timeout}s")
                return ExecResult(
                    exit_code=int(message["exit_code"]), stdout=message["stdout"], stderr=message["stderr"]
                )
            await confirm_liveness()
            if connection.token_provider is not None:
                refreshed = await asyncio.to_thread(connection.resolved_token)
                await send({"kind": "token", "value": refreshed})
    except asyncio.CancelledError:
        await cancel_owned_command()
        raise
    finally:
        await _stop_process(process)
        reader.cancel()
        stderr_reader.cancel()
        await asyncio.gather(reader, stderr_reader, return_exceptions=True)
        if process.stdin is not None:
            process.stdin.close()


if __name__ == "__main__":
    raise SystemExit(_worker_main())
