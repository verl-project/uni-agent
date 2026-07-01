from __future__ import annotations

import asyncio
import logging
import os
import uuid
from time import monotonic
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from swerex.runtime.abstract import Command

    from .base import SandboxConfig

logger = logging.getLogger(__name__)

#: swerex server port inside the sandbox (veFaaS routes the function URL here).
_RUNTIME_PORT = 8000


class _VefaasRuntime:
    """Minimal async swerex client that speaks veFaaS routing.

    Covers just what the sandbox needs -- ``execute`` for exec, liveness, and
    ``close`` -- posting to the function-route base URL with the veFaaS headers
    (``X-API-Key`` + ``X-Faas-Instance-Name``) and swerex's own pydantic models.
    """

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str,
        instance_name: str,
        timeout: float = 60.0,
        proxy: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._instance_name = instance_name
        self._timeout = timeout
        self._proxy = proxy

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._auth_token:
            headers["X-API-Key"] = self._auth_token
        if self._instance_name:
            headers["X-Faas-Instance-Name"] = str(self._instance_name)
        return headers

    async def _post(self, endpoint: str, payload: Any, output_cls: type, *, timeout: float | None = None):
        import aiohttp

        total = timeout if timeout is not None else self._timeout
        headers = {**self._headers, "X-Request-ID": uuid.uuid4().hex}
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(connector=connector, proxy=self._proxy) as session:
            async with session.post(
                f"{self._base_url}/{endpoint}",
                json=payload.model_dump() if payload is not None else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=total),
            ) as resp:
                resp.raise_for_status()
                return output_cls(**(await resp.json()))

    async def is_alive(self, *, timeout: float | None = None) -> bool:
        import aiohttp

        total = timeout if timeout is not None else self._timeout
        try:
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector, proxy=self._proxy) as session:
                async with session.get(
                    f"{self._base_url}/is_alive",
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=total),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def wait_until_alive(self, *, timeout: float = 120.0, interval: float = 2.0) -> None:
        deadline = monotonic() + timeout
        probe_timeout = min(self._timeout, 10.0)
        while True:
            if await self.is_alive(timeout=probe_timeout):
                return
            if monotonic() >= deadline:
                raise TimeoutError(f"veFaaS runtime not alive within {timeout}s at {self._base_url}")
            await asyncio.sleep(interval)

    async def execute(self, command: Command):
        from swerex.runtime.abstract import CommandResponse

        # A long command timeout must not be cut off by the shorter client timeout.
        cmd_timeout = getattr(command, "timeout", None)
        http_timeout = max(self._timeout, cmd_timeout + 30) if cmd_timeout else self._timeout
        return await self._post("execute", command, CommandResponse, timeout=http_timeout)

    async def close(self) -> None:
        try:
            from swerex.runtime.abstract import CloseResponse

            await self._post("close", None, CloseResponse)
        except Exception:
            logger.debug("veFaaS runtime close() failed", exc_info=True)


def _get_vefaas_client():
    """Build a Volcengine veFaaS API client (blocking SDK; call off the event loop)."""
    import volcenginesdkcore
    import volcenginesdkvefaas

    access_key = os.getenv("VOLCE_ACCESS_KEY") or os.getenv("VOLCENGINE_ACCESS_KEY")
    secret_key = os.getenv("VOLCE_SECRET_KEY") or os.getenv("VOLCENGINE_SECRET_KEY")
    region = os.getenv("VEFAAS_REGION", "cn-beijing")
    proxy = os.getenv("SANDBOX_PROXY")

    if not (access_key and secret_key):
        raise ValueError(
            "VefaasSandbox needs Volcengine credentials: set VOLCE_ACCESS_KEY / VOLCE_SECRET_KEY."
        )

    configuration = volcenginesdkcore.Configuration()
    configuration.ak = access_key
    configuration.sk = secret_key
    configuration.read_timeout = 120
    configuration.connect_timeout = 120
    configuration.auto_retry = False
    configuration.region = region
    configuration.client_side_validation = True
    if proxy:
        configuration.proxy = proxy
    return volcenginesdkvefaas.VEFAASApi(volcenginesdkcore.ApiClient(configuration))


@register_sandbox("vefaas")
class VefaasSandbox(Sandbox):
    """Creates a Volcengine veFaaS sandbox and drives it over swerex."""

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600.0,
        startup_timeout: float = 120.0,
    ) -> None:
        self.image = image
        self.runtime_timeout = runtime_timeout
        self.startup_timeout = startup_timeout
        self._function_id: str = os.getenv("VEFAAS_FUNCTION_ID")
        self._function_route: str = os.getenv("VEFAAS_FUNCTION_ROUTE")
        self._client: Any | None = None
        self._sandbox_id: str | None = None
        self._runtime: _VefaasRuntime | None = None

        assert self._function_id is not None, "VEFAAS_FUNCTION_ID is not set"
        assert self._function_route is not None, "VEFAAS_FUNCTION_ROUTE is not set"

    @classmethod
    def from_config(cls, config: SandboxConfig) -> VefaasSandbox:
        # Standard fields map to constructor args; veFaaS specifics (function_id,
        # function_route, proxy, ...) ride along in sandbox_kwargs.
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    # ----- control plane -----
    async def start(self) -> None:
        if self._runtime is not None:
            return  # already started

        import volcenginesdkvefaas

        self._client = _get_vefaas_client()

        token = uuid.uuid4().hex
        command = f"curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}"
        instance_image_info = volcenginesdkvefaas.InstanceImageInfoForCreateSandboxInput(
            image=self.image, port=_RUNTIME_PORT, command=command,
        )
        request = volcenginesdkvefaas.CreateSandboxRequest(
            function_id=self._function_id,
            instance_image_info=instance_image_info,
            timeout=int(self.runtime_timeout / 60),
        )
        # create_sandbox is a blocking SDK call; run it off the event loop.
        resp = await asyncio.to_thread(self._client.create_sandbox, request)
        sandbox_id = resp.sandbox_id
        if not sandbox_id:
            raise RuntimeError("veFaaS create_sandbox returned no sandbox id")
        self._sandbox_id = sandbox_id

        runtime = _VefaasRuntime(
            base_url=self._function_route,
            auth_token=token,
            instance_name=sandbox_id,
            proxy=os.getenv("SANDBOX_PROXY"),
        )
        await runtime.wait_until_alive(timeout=self.startup_timeout)
        self._runtime = runtime

    async def stop(self) -> None:
        # Idempotent via the None checks below: a second call finds nothing to do.
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None

        if self._sandbox_id is not None and self._client is not None:
            import volcenginesdkvefaas

            request = volcenginesdkvefaas.KillSandboxRequest(
                function_id=self._function_id, sandbox_id=self._sandbox_id
            )
            # kill_sandbox is a blocking SDK call; run it off the event loop.
            await asyncio.to_thread(self._client.kill_sandbox, request)
            self._sandbox_id = None
        self._client = None

    def _require_runtime(self) -> _VefaasRuntime:
        if self._runtime is None:
            raise RuntimeError("VefaasSandbox not started; call start() first")
        return self._runtime

    # ----- data plane -----
    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # ``execute`` runs argv once in a fresh session on the swerex server (no
        # implicit shell). Command's cwd / env / timeout map straight onto our args.
        from swerex.runtime.abstract import Command

        resp = await self._require_runtime().execute(
            Command(command=list(argv), shell=False, cwd=workdir, env=env or None, timeout=timeout)
        )
        return ExecResult(
            exit_code=int(resp.exit_code or 0),
            stdout=_to_str(resp.stdout),
            stderr=_to_str(resp.stderr),
        )
