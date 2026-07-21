"""openYuanrong remote sandbox command execution.

This sandbox infra is developed collaboratively by the OpenYuanrong team and the Ant AKernel team.

Wraps remote sandbox lifecycle (create, run commands, cleanup) for blackbox agent
recipes such as mini-swe-agent, including sidecar tool mounts and gateway tunnel helpers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
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


def _resolve_sandbox_name() -> str | None:
    """Return ``{prefix}{random}`` when ``SANDBOX_NAME_PREFIX`` env is set."""
    prefix = os.getenv("SANDBOX_NAME_PREFIX")
    if not prefix:
        return None
    return f"{prefix}{uuid.uuid4().hex[:8]}"


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


class SandboxClient:
    """Command execution via remote sandbox."""

    _sdk_initialized = False

    @classmethod
    def init(cls) -> None:
        """Configure env and select sandbox SDK via ``sys.modules`` injection."""
        if cls._sdk_initialized:
            return

        # Require OpenYuanrong credentials from the caller environment.
        server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
        token = os.getenv("OPENYUANRONG_TOKEN")
        if not server or not token:
            raise ValueError(
                "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN "
                "environment variables must be set for sandbox"
            )
        # Reverse tunnel TLS verify
        os.environ["TUNNEL_SSL_VERIFY"] = os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", "0")

        if os.getenv("USE_OPENYUANRONG_SDK", "0") == "1":
            try:
                import openyuanrong_sandbox_sdk as _sdk_module
            except ImportError:
                raise ImportError(
                    "USE_OPENYUANRONG_SDK=1 but openyuanrong_sandbox_sdk is not installed. "
                    "Please install openyuanrong_sandbox_sdk or set USE_OPENYUANRONG_SDK=0."
                ) from None
        else:
            os.environ["AKERNEL_SERVER_ADDRESS"] = server
            os.environ["AKERNEL_TOKEN"] = token
            try:
                import akernel_sdk as _sdk_module
            except ImportError:
                raise ImportError(
                    "USE_OPENYUANRONG_SDK=0 but akernel_sdk is not installed. "
                    "Please install akernel_sdk or set USE_OPENYUANRONG_SDK=1."
                ) from None
        sys.modules["openyuanrong_sandbox_sdk"] = _sdk_module
        cls._sdk_initialized = True

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
        max_retries: int = 10,
        **sandbox_kwargs: Any,
    ) -> SandboxClient:
        """Create an sandbox client with sidecar tool mounted.

        The sidecar image is mounted at ``sidecar_target`` inside the
        sandbox via the ``openyuanrong_sandbox_sdk.Mount`` class.

        If ``upstream`` is provided, a tunnel is set up so the sandbox can
        reach the local gateway via ``http://127.0.0.1:<proxy_port>``.
        """
        if not cls._sdk_initialized:
            cls.init()

        from openyuanrong_sandbox_sdk import Mount, Sandbox

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
            "Creating sandbox (image=%s, cpu=%d, memory=%d, sidecar=%s:%s, upstream=%s)",
            image,
            cpu,
            memory,
            sidecar_image,
            sidecar_target,
            upstream or "none",
        )
        last_error: Exception | None = None
        for retry in range(max_retries):
            # Regenerate name each attempt so retries do not collide on the same name.
            name = _resolve_sandbox_name()
            if name is not None:
                sb_kwargs["name"] = name
            try:
                sandbox = await asyncio.to_thread(lambda: Sandbox(**sb_kwargs))
                logger.info(
                    "sandbox created: %s (image=%s)",
                    getattr(sandbox, "sandbox_id", "?"),
                    image,
                )
                return cls(sandbox=sandbox)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to create sandbox (image=%s, name=%s, attempt=%d/%d): %s",
                    image,
                    name or "auto",
                    retry + 1,
                    max_retries,
                    exc,
                )
                # Create may fail after remote allocation while the local handle
                # was never assigned; clean up by name when available.
                if name is not None:
                    try:
                        await asyncio.to_thread(lambda n=name: Sandbox.delete(n))
                    except Exception:
                        pass
                if retry < max_retries - 1:
                    sleep_time = min(30, 2**retry)
                    logger.info("Retrying sandbox creation in %d seconds...", sleep_time)
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create sandbox after {max_retries} retries") from last_error

    async def run(self, cmd: str, *, timeout: int = 600) -> CommandResult:
        """Execute *cmd* inside the sandbox via ``sandbox.commands.run``."""
        try:
            result = await asyncio.to_thread(
                self._sandbox.commands.run,
                cmd,
                timeout=timeout,
            )
            return CommandResult(
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
                exit_code=getattr(result, "exit_code", -1),
            )
        except Exception as e:
            return CommandResult(stdout="", stderr=str(e), exit_code=-1)

    async def cleanup(self) -> None:
        """Kill the sandbox if still running."""
        if self._sandbox is not None:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "?")
            try:
                if self._sandbox.is_running():
                    await asyncio.to_thread(self._sandbox.kill)
                    logger.info("sandbox %s killed", sandbox_id)
                else:
                    logger.info("sandbox %s already stopped", sandbox_id)
            except Exception as e:
                logger.warning("Failed to kill sandbox %s: %s", sandbox_id, e)
            self._sandbox = None