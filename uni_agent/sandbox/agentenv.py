"""AgentEnv sandbox provider: an E2B-protocol deployment served by AgentEnv.

AgentEnv exposes an E2B-compatible control plane and ``envd`` data plane, so this
provider reuses :class:`~uni_agent.sandbox.e2b.E2BSandbox` (exec, file transfer,
port exposure, template handling) with native AgentEnv image mounts. The
endpoint/credential triplet is resolved per sandbox rather than written into the
process environment, which lets ``e2b`` and ``agentenv`` sandboxes coexist in one
worker (and keeps the API key out of task YAML, which is logged).

Connection settings, in priority order:

======================  ===================================  ==========================
setting                 explicit (``sandbox_kwargs``)        environment fallback
======================  ===================================  ==========================
``api_url``             ``sandbox_kwargs.api_url``           ``AGENTENV_API_URL`` > ``E2B_API_URL``
``sandbox_url``         ``sandbox_kwargs.sandbox_url``       ``AGENTENV_SANDBOX_URL`` > ``E2B_SANDBOX_URL``
``api_key``             ``sandbox_kwargs.api_key``           ``AGENTENV_API_KEY`` > ``E2B_API_KEY``
======================  ===================================  ==========================

All three are required; a blank value counts as unset. When one is missing startup
fails with a :class:`ValueError` that names the setting and the environment
variables to set (values themselves are never logged or echoed).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .e2b import E2BSandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount

logger = logging.getLogger(__name__)

#: setting -> its environment fallbacks, highest priority first.
_CONNECTION_ENV_VARS: dict[str, tuple[str, ...]] = {
    "api_url": ("AGENTENV_API_URL", "E2B_API_URL"),
    "sandbox_url": ("AGENTENV_SANDBOX_URL", "E2B_SANDBOX_URL"),
    "api_key": ("AGENTENV_API_KEY", "E2B_API_KEY"),
}


_NATIVE_NETWORK_FIELDS = {
    "allow_public_traffic": "allowPublicTraffic",
    "allow_out": "allowOut",
    "deny_out": "denyOut",
    "mask_request_host": "maskRequestHost",
}


def _resolve_connection_value(explicit: str | None, env_names: tuple[str, ...]) -> str | None:
    """Resolve one connection setting: explicit kwarg, then env vars, else ``None``.

    Blank/whitespace-only values count as unset at every level; values are
    stripped and never logged.
    """
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


@register_sandbox("agentenv")
class AgentEnvSandbox(E2BSandbox):
    """E2B-protocol sandbox served by an AgentEnv deployment.

    Reuses :class:`E2BSandbox`'s data plane and lifecycle; only the connection
    triplet differs. Prefer ``sandbox_kwargs.template`` over ``image``:

    - **template first**: an AgentEnv deployment is not required to implement
      E2B's template-build API. When ``sandbox_kwargs.template`` names an existing
      template alias, startup uses it as-is and no ``Template.exists`` /
      ``Template.build`` call is made.
    - **image needs template builds**: when only ``image`` is given, startup goes
      through the E2B template API (``Template.exists`` then ``Template.build``,
      with the same connection kwargs). The AgentEnv deployment must implement
      that API; otherwise pass ``template`` instead.
    - **expose_port needs domain**: the E2B SDK derives the exposed host as
      ``{port}-{sandbox_id}.{domain}`` (or from ``sandbox_url``). Without a
      configured/reachable domain the returned URL is not routable, so
      ``expose_port`` is only usable when the deployment's domain is known.
    - image mounts use native AgentEnv attached drives. With a template, the
      snapshot needs a free drive slot per mount. Without a template, the task
      image is cold-started directly.

    Everything else (``runtime_timeout``, ``executable_paths``, ``cpu_count``,
    ``memory_mb``, ``envs``, ``metadata``, ``proxy``, ``from_config``) is
    inherited from :class:`E2BSandbox`.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        sandbox_url: str | None = None,
        api_key: str | None = None,
        proxy_port: int | None = None,
        image_mounts: list[ImageMount] | None = None,
        **kwargs: Any,
    ) -> None:
        resolved = {
            name: _resolve_connection_value(value, _CONNECTION_ENV_VARS[name])
            for name, value in (("api_url", api_url), ("sandbox_url", sandbox_url), ("api_key", api_key))
        }
        missing = [name for name, value in resolved.items() if value is None]
        if missing:
            hints = "; ".join(
                f"{name}: pass sandbox_kwargs.{name} or set {' / '.join(_CONNECTION_ENV_VARS[name])}"
                for name in missing
            )
            raise ValueError(
                f"AgentEnvSandbox is missing required connection settings (values are never logged). Configure {hints}."
            )
        super().__init__(
            api_url=resolved["api_url"], sandbox_url=resolved["sandbox_url"], api_key=resolved["api_key"], **kwargs
        )

        self.image_mounts = list(image_mounts or [])
        if self.image_mounts:
            if self.template is not None and not self.secure:
                raise ValueError("AgentEnv warm image mounts require secure sandbox communication")
            if self.network and self.network.keys() - _NATIVE_NETWORK_FIELDS.keys():
                raise ValueError("AgentEnv image mounts received unsupported sandbox network settings")
            lifecycle = self.lifecycle or {}
            if lifecycle.keys() - {"on_timeout"} or lifecycle.get("on_timeout") not in (None, "kill", "pause"):
                raise ValueError("AgentEnv image mounts support only lifecycle.on_timeout=kill or pause")

    async def _create_sandbox(self) -> Any:
        if not self.image_mounts:
            return await super()._create_sandbox()

        import httpx
        from e2b import AsyncSandbox

        request: dict[str, Any] = {
            "timeout": max(1, int(self.runtime_timeout)),
            "autoPause": (self.lifecycle or {}).get("on_timeout") == "pause",
            "metadata": self.metadata,
            "envVars": self.envs,
            "attachedDrives": [
                {
                    "driveID": f"image_mount_{index}",
                    "source": {"image": mount.image},
                    "mountPath": mount.mount_path,
                    "readOnly": True,
                }
                for index, mount in enumerate(self.image_mounts)
            ],
        }
        if self.network is not None:
            request["network"] = {_NATIVE_NETWORK_FIELDS[key]: value for key, value in self.network.items()}
        if self.template is not None:
            path = "/sandboxes"
            request.update(templateID=self.template, allow_internet_access=self.allow_internet_access)
        else:
            path = "/sandboxes-cold"
            request.update(
                image=self.image,
                secure=self.secure,
                allowInternetAccess=self.allow_internet_access,
                cpuCount=self.cpu_count,
                memoryMB=self.memory_mb,
            )

        base_url = self._connection_kwargs["api_url"].rstrip("/")
        async with httpx.AsyncClient(
            headers={"X-API-Key": self._connection_kwargs["api_key"]},
            proxy=self.proxy,
            http2=self._connection_kwargs.get("http_version") == "2",
            timeout=httpx.Timeout(600.0, connect=10.0),
        ) as client:
            response = await client.post(f"{base_url}{path}", json=request)
            response.raise_for_status()
            sandbox_id = response.json()["sandboxID"]
            try:
                return await AsyncSandbox.connect(sandbox_id, **self._connection_kwargs)
            except BaseException:
                try:
                    cleanup = await client.delete(f"{base_url}/sandboxes/{quote(sandbox_id, safe='')}", timeout=10)
                    cleanup.raise_for_status()
                except Exception:
                    logger.warning("Failed to clean up AgentEnv sandbox %s after connect failed", sandbox_id)
                raise

    async def start(self) -> None:
        try:
            await super().start()
        except BaseException:
            if self.image_mounts:
                await self.stop()
            raise
