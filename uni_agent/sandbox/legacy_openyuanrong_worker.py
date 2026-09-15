"""Small Python 3.11 bridge for the legacy OpenYuanRong SDK.

The current Uni-Agent runtime uses Python 3.12 for Ray/vLLM, while the
remote186 OpenYuanRong service used by the existing Codex/OpenClaw runs is
served by the legacy ``akernel_sdk`` API.  Its native ``yr`` extension is
Python-3.11-only, so this process keeps the legacy SDK isolated behind a
newline-delimited JSON RPC protocol.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import json
import os
import sys
from types import SimpleNamespace
from typing import Any


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _double_quote(value: str) -> str:
    """Quote one argv item for the legacy service's outer shell parser."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def _command_line(argv: list[str]) -> str:
    return " ".join(_double_quote(str(item)) for item in argv)


def _redacted_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for key in ("OPENYUANRONG_SERVER_ADDRESS", "OPENYUANRONG_TOKEN", "AKERNEL_SERVER_ADDRESS", "AKERNEL_TOKEN"):
        secret = os.environ.get(key)
        if secret:
            message = message.replace(secret, "<redacted>")
    return message[:4000]


def _load_sdk() -> Any:
    server = os.environ.get("OPENYUANRONG_SERVER_ADDRESS", "").strip()
    token = os.environ.get("OPENYUANRONG_TOKEN", "").strip()
    if not server or not token:
        raise ValueError("OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN are required")

    # The legacy yr/fnruntime extension resolves libffi symbols at import time.
    # Load the host's libffi 7 globally when available, matching the existing
    # Codex/OpenClaw launcher; this process never hosts CUDA/vLLM.
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token
    # akernel_sdk.Sandbox's reverse-tunnel code reads the YR_* names when it
    # builds the Traefik WebSocket URL.  The old Codex/OpenClaw launchers set
    # these implicitly through akernel_sdk, but an isolated helper must make
    # the mapping explicit before importing the SDK.
    gateway = (
        os.environ.get("OPENYUANRONG_GATEWAY_ADDRESS", "").strip()
        or os.environ.get("AKERNEL_GATEWAY_ADDRESS", "").strip()
        or server
    )
    os.environ["YR_SERVER_ADDRESS"] = server
    os.environ["YR_GATEWAY_ADDRESS"] = gateway
    os.environ["TUNNEL_SSL_VERIFY"] = os.environ.get("OPENYUANRONG_TUNNEL_SSL_VERIFY", "0")
    libffi = os.environ.get("OPENYUANRONG_LIBFFI_PATH", "/usr/lib/x86_64-linux-gnu/libffi.so.7")
    if os.path.exists(libffi):
        ctypes.CDLL(libffi, mode=getattr(ctypes, "RTLD_GLOBAL", 0x100))

    from akernel_sdk import Mount, Sandbox

    return SimpleNamespace(Mount=Mount, Sandbox=Sandbox)


def _command_result(result: object) -> dict[str, object]:
    return {
        "exit_code": int(getattr(result, "exit_code", -1)),
        "stdout": _text(getattr(result, "stdout", "")),
        "stderr": _text(getattr(result, "stderr", "")),
    }


class LegacyWorker:
    def __init__(self) -> None:
        self.sdk = _load_sdk()
        self.sandbox: Any = None
        self.shells: dict[str, Any] = {}
        self.next_shell_id = 0

    async def handle(self, request: dict[str, Any]) -> object:
        op = request.get("op")
        if op == "start":
            return await self._start(request.get("kwargs") or {})
        if op == "stop":
            return await self._stop()
        if op == "is_alive":
            return bool(self.sandbox is not None and self.sandbox.is_running())
        if op == "exec":
            return await self._exec(request)
        if op == "exec_shell":
            return await self._exec_shell(request)
        if op == "read_file":
            return await self._read_file(request["path"])
        if op == "write_file":
            return await self._write_file(request["path"], request["data"])
        if op == "upload":
            return await self._upload(request["local_path"], request["remote_path"])
        if op == "download":
            return await self._download(request["remote_path"], request["local_path"])
        if op == "shell_create":
            return await self._shell_create(request)
        if op == "shell_run":
            return await self._shell_run(request)
        if op == "shell_close":
            return await self._shell_close(request)
        if op == "get_port_url":
            return self.sandbox.get_port_url(int(request["port"]))
        if op == "get_tunnel_url":
            return self.sandbox.get_tunnel_url()
        raise ValueError(f"unknown legacy sandbox RPC operation: {op!r}")

    async def _start(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.sandbox is not None:
            return {"sandbox_id": str(getattr(self.sandbox, "sandbox_id", ""))}
        options = dict(kwargs)
        mounts = options.get("mounts")
        if mounts:
            options["mounts"] = [self.sdk.Mount(**mount) if isinstance(mount, dict) else mount for mount in mounts]
        self.sandbox = self.sdk.Sandbox(**options)
        return {"sandbox_id": str(getattr(self.sandbox, "sandbox_id", ""))}

    async def _stop(self) -> dict[str, bool]:
        if self.sandbox is not None:
            self.sandbox.kill()
            self.sandbox = None
        self.shells.clear()
        return {"stopped": True}

    async def _exec(self, request: dict[str, Any]) -> dict[str, object]:
        command = _command_line(request["argv"])
        result = self.sandbox.commands.run(
            command,
            envs=request.get("env"),
            cwd=request.get("workdir"),
            timeout=int(request.get("timeout") or 60),
        )
        return _command_result(result)

    async def _exec_shell(self, request: dict[str, Any]) -> dict[str, object]:
        result = self.sandbox.commands.run(
            request["command"],
            envs=request.get("env"),
            cwd=request.get("workdir"),
            timeout=int(request.get("timeout") or 60),
        )
        return _command_result(result)

    async def _read_file(self, path: str) -> dict[str, str]:
        data = self.sandbox.files.read(path, format="bytes")
        return {"data": base64.b64encode(bytes(data)).decode("ascii")}

    async def _write_file(self, path: str, data: str) -> dict[str, bool]:
        self.sandbox.files.write(path, base64.b64decode(data))
        return {"written": True}

    async def _upload(self, local_path: str, remote_path: str) -> dict[str, bool]:
        self.sandbox.files.copy_from_local(local_path, remote_path)
        return {"uploaded": True}

    async def _download(self, remote_path: str, local_path: str) -> dict[str, bool]:
        self.sandbox.files.copy_to_local(remote_path, local_path)
        return {"downloaded": True}

    async def _shell_create(self, request: dict[str, Any]) -> dict[str, str]:
        shell = await self.sandbox.shells.create(cwd=request.get("cwd"), envs=request.get("env"))
        self.next_shell_id += 1
        shell_id = f"shell-{self.next_shell_id}"
        self.shells[shell_id] = shell
        return {"shell_id": shell_id}

    async def _shell_run(self, request: dict[str, Any]) -> dict[str, object]:
        shell = self.shells[request["shell_id"]]
        result = await shell.run(
            request["command"],
            envs=request.get("env"),
            cwd=request.get("workdir"),
            timeout=int(request.get("timeout") or 60),
        )
        return _command_result(result)

    async def _shell_close(self, request: dict[str, Any]) -> dict[str, bool]:
        shell = self.shells.pop(request["shell_id"], None)
        if shell is not None:
            await shell.kill()
        return {"closed": True}


async def main() -> int:
    protocol_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            worker = LegacyWorker()
    except BaseException as exc:
        print(_redacted_error(exc), file=sys.stderr, flush=True)
        return 2

    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("legacy sandbox RPC request must be an object")
            with contextlib.redirect_stdout(sys.stderr):
                result = await worker.handle(request)
            response = {"id": request.get("id"), "ok": True, "result": result}
        except BaseException as exc:
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": _redacted_error(exc),
            }
        protocol_stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        protocol_stdout.flush()

    await worker._stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
