"""OpenYuanRong (AKernel) remote sandbox command execution.

Uses ``akernel_sdk.Sandbox`` with sidecar ``Mount`` to inject the
mini-swe-agent tool image.  Supports upstream tunnel so the agent
inside the sandbox can reach the gateway via ``http://127.0.0.1:<proxy_port>``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass
class CommandResult:
    """Result of a command executed inside a sandbox."""

    stdout: str
    stderr: str
    exit_code: int

logger = logging.getLogger(__name__)

DEFAULT_PROXY_PORT = 38197


def _configure_akernel_env() -> None:
    """Map OPENYUANRONG_* env vars to AKERNEL_* before importing akernel_sdk."""
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    tunnel_ssl_verify = os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", "0")
    if not server or not token:
        raise ValueError(
            "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN "
            "environment variables must be set for YR sandbox"
        )
    os.environ["AKERNEL_SERVER_ADDRESS"] = server
    os.environ["AKERNEL_TOKEN"] = token
    os.environ["TUNNEL_SSL_VERIFY"] = tunnel_ssl_verify


def extract_upstream(gateway_url: str) -> str:
    """Extract host:port from a gateway URL for upstream tunnel config.

    Example: "http://8.92.9.155:40169/sessions/abc/v1" -> "8.92.9.155:40169"
    """
    parsed = urlparse(gateway_url)
    return f"{parsed.hostname}:{parsed.port}"


def rewrite_gateway_url(
    gateway_url: str,
    proxy_port: int = DEFAULT_PROXY_PORT,
    *,
    strip_v1: bool = False,
) -> str:
    """Rewrite gateway URL to use the sandbox-internal tunnel.

    Replaces host:port with 127.0.0.1:<proxy_port>, keeps path intact.

    Example:
        "http://8.92.9.155:40169/sessions/abc/v1"
        -> "http://127.0.0.1:8766/sessions/abc/v1"
    """
    parsed = urlparse(gateway_url)
    path = parsed.path.removesuffix("/v1") if strip_v1 else parsed.path
    return f"http://127.0.0.1:{proxy_port}{path}"


class YRSandbox:
    """Command execution via OpenYuanRong (AKernel) remote sandbox."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return getattr(self._sandbox, "sandbox_id", "unknown")


    @classmethod
    async def create(
        cls,
        *,
        image: str,
        sidecar_image: str,
        upstream: str = "",
        proxy_port: int = DEFAULT_PROXY_PORT,
        env: dict[str, str] | None = None,
        cpu: int = 1000,
        memory: int = 2048,
        cpu_limit: int = 4000,
        mem_limit: int = 8192,
        idle_timeout: int = 7200,
        sidecar_target: str = "/opt/mini-swe-agent",
        max_retries: int = 5,
        **sandbox_kwargs: Any,
    ) -> "YRSandbox":
        """Create an OpenYuanRong sandbox with sidecar tool mounted.

        The sidecar image is mounted at ``sidecar_target`` inside the
        sandbox via ``akernel_sdk.Mount``.

        If ``upstream`` is provided, a tunnel is set up so the sandbox can
        reach the local gateway via ``http://127.0.0.1:<proxy_port>``.
        """
        _configure_akernel_env()
        from akernel_sdk import Mount, Sandbox

        sb_kwargs: dict[str, Any] = {
            "image": image,
            "cpu": cpu,
            "memory": memory,
            "cpu_limit": cpu_limit,
            "mem_limit": mem_limit,
            "idle_timeout": idle_timeout,
            "mounts": [
                Mount(target=sidecar_target, image_url=sidecar_image),
            ],
        }
        if upstream:
            sb_kwargs["upstream"] = upstream
            sb_kwargs["proxy_port"] = proxy_port
        if env:
            sb_kwargs["env"] = env
        sb_kwargs.update(sandbox_kwargs)

        logger.info(
            "Creating YR sandbox (image=%s, cpu=%d, memory=%d, sidecar=%s:%s, upstream=%s)",
            image, cpu, memory, sidecar_image, sidecar_target, upstream or "none",
        )
        last_error: Exception | None = None
        for retry in range(max_retries):
            sandbox = None
            try:
                sandbox = await asyncio.to_thread(lambda: Sandbox(**sb_kwargs))
                logger.info("YR sandbox created: %s", getattr(sandbox, "sandbox_id", "?"))
                return cls(sandbox=sandbox)
            except Exception as exc:
                last_error = exc
                sandbox_id = getattr(sandbox, "sandbox_id", None)
                logger.critical(
                    "Failed to create YR sandbox (sandbox_id=%s): %s",
                    sandbox_id or "n/a", exc,
                )
                if sandbox is not None:
                    try:
                        await asyncio.to_thread(sandbox.kill)
                    except Exception:
                        pass
                if retry < max_retries - 1:
                    sleep_time = min(30, 2 ** retry)
                    logger.info("Retrying YR sandbox creation in %d seconds...", sleep_time)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create YR sandbox after {max_retries} retries") from last_error

    async def run(self, cmd: str, *, timeout: int = 600) -> CommandResult:
        """Execute *cmd* inside the OpenYuanRong sandbox via ``sandbox.commands.run``."""
        try:
            result = await asyncio.to_thread(
                self._sandbox.commands.run, cmd, timeout=timeout,
            )
            return CommandResult(
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
                exit_code=getattr(result, "exit_code", -1),
            )
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Kill the OpenYuanRong sandbox if still running."""
        if self._sandbox is not None:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "?")
            try:
                if self._sandbox.is_running():
                    await asyncio.to_thread(self._sandbox.kill)
                    logger.info("YR sandbox %s killed", sandbox_id)
                else:
                    logger.info("YR sandbox %s already stopped", sandbox_id)
            except Exception as e:
                logger.warning("Failed to kill YR sandbox %s: %s", sandbox_id, e)
            self._sandbox = None
