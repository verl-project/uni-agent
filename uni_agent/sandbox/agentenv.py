"""AgentEnv sandbox provider: an E2B-protocol deployment served by AgentEnv.

AgentEnv exposes an E2B-compatible control plane and ``envd`` data plane, so this
provider reuses :class:`~uni_agent.sandbox.e2b.E2BSandbox` (exec, file transfer,
port exposure, template handling) and only fills in the connection settings. The
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

import os
from typing import Any

from .e2b import E2BSandbox
from .registry import register_sandbox

#: setting -> its environment fallbacks, highest priority first.
_CONNECTION_ENV_VARS: dict[str, tuple[str, ...]] = {
    "api_url": ("AGENTENV_API_URL", "E2B_API_URL"),
    "sandbox_url": ("AGENTENV_SANDBOX_URL", "E2B_SANDBOX_URL"),
    "api_key": ("AGENTENV_API_KEY", "E2B_API_KEY"),
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
    - image mounts are not supported; build a composite AgentEnv/E2B template
      and reference it through ``template``.

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
        if kwargs.get("image_mounts"):
            raise NotImplementedError(
                "AgentEnvSandbox does not support image mounts; build a composite AgentEnv/E2B template "
                "and reference it with sandbox_kwargs.template instead"
            )
        super().__init__(
            api_url=resolved["api_url"], sandbox_url=resolved["sandbox_url"], api_key=resolved["api_key"], **kwargs
        )
