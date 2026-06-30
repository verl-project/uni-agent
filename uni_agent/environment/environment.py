"""Environment: the composition root over a sandbox + a toolbox.

Turns an :class:`EnvironmentConfig` into a runnable world: pick a sandbox
provider, build a :class:`~uni_agent.tools.Toolbox` from the selected tools (each
tool bound to the sandbox and owning its own state), and expose a small agent-loop
surface (:meth:`reset` / :meth:`call` / :meth:`close`). Teardown is ordered --
the toolbox closes (releasing any open channels) before the sandbox stops.

There is no separate session layer: each ``tools`` entry is a ``{name, ...kwargs}``
mapping, and a stateful tool (e.g. ``shell``) opens and owns its channel
internally. A tool entry's kwargs are auto-parsed into its config_model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..sandbox import LocalSandbox, ModalSandbox, Sandbox
from ..tools import TOOL_REGISTRY, Toolbox
from .config import EnvironmentConfig, SandboxConfig


def _tool_entry(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Split a tools config entry ``{name, ...kwargs}`` into ``(name, kwargs)``."""
    if not isinstance(entry, dict) or not entry.get("name"):
        raise ValueError(f"each tools entry must be a mapping with a 'name': {entry!r}")
    name = entry["name"]
    kwargs = {k: v for k, v in entry.items() if k != "name"}
    return name, kwargs

if TYPE_CHECKING:
    from ..sandbox import SandboxBackend
    from ..tools import Observation, Tool


def build_sandbox(config: SandboxConfig) -> Sandbox:
    """Instantiate the sandbox backend named by ``config.provider``."""
    provider = config.provider
    if provider == "local":
        return LocalSandbox()
    if provider == "modal":
        if ModalSandbox is None:
            raise RuntimeError("modal is not installed; `pip install modal` to use provider='modal'")
        # sandbox_kwargs is forwarded to the ModalSandbox constructor, so app_name /
        # modal_sandbox_kwargs (or an image / runtime_timeout override) all flow
        # through it; explicit fields are the defaults it may override.
        kwargs: dict[str, Any] = {
            "image": config.image,
            "runtime_timeout": config.runtime_timeout,
            **config.sandbox_kwargs,
        }
        return ModalSandbox(**kwargs)
    if provider == "vefaas":
        raise NotImplementedError(
            "provider='vefaas' is not yet ported to uni_agent.sandbox; available providers: local, modal"
        )
    raise ValueError(f"unknown sandbox provider: {provider!r}")


def build_tool(
    name: str,
    sandbox: SandboxBackend,
    *,
    kwargs: dict[str, Any] | None = None,
) -> Tool:
    """Instantiate a tool by name, bound to ``sandbox`` and tuned by ``kwargs``.

    ``kwargs`` is auto-parsed into the tool's ``config_model`` by the tool
    constructor, so a tool's own timeout (e.g. the shell's ``command_timeout``)
    lives in its kwargs.
    """
    cls = TOOL_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown tool: {name!r}")
    return cls(sandbox, **(kwargs or {}))


class Environment:
    """A runnable agent environment: one sandbox + a toolbox of stateful tools.

    Typical use::

        env = Environment.from_config(yaml_dict)
        async with env:                       # reset() on enter, close() on exit
            schemas = env.tool_schemas()      # hand to the model
            obs = await env.call("shell", {"command": "ls"})
            print(obs.text)
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        tools: list[dict[str, Any]],
    ):
        self.sandbox = sandbox
        # Each entry is a {name, ...kwargs} mapping -> (name, kwargs).
        self._tool_specs = [_tool_entry(entry) for entry in tools]
        self.toolbox: Toolbox | None = None
        self._started = False

    @classmethod
    def from_config(cls, config: EnvironmentConfig | dict[str, Any]) -> Environment:
        """Build an environment from an :class:`EnvironmentConfig` or a raw dict.

        A dict may be the inner config or a full document with a top-level
        ``environment:`` key (as loaded straight from YAML).
        """
        if isinstance(config, dict):
            config = EnvironmentConfig(**config.get("environment", config))
        sandbox = build_sandbox(config.sandbox)
        return cls(sandbox=sandbox, tools=config.tools)

    # ----- lifecycle -----
    async def reset(self) -> list[dict]:
        """Boot the sandbox and build the toolbox; return the tool schemas.

        Tools are (re)built fresh against the running sandbox, so a stateful tool
        never carries a channel over from a previous episode. The schemas are what
        a policy passes to the model.
        """
        if self._started:
            await self.close()
        await self.sandbox.start()
        tools = [
            build_tool(name, self.sandbox, kwargs=kwargs)
            for name, kwargs in self._tool_specs
        ]
        self.toolbox = Toolbox(tools)
        self._started = True
        return self.toolbox.schemas()

    def tool_schemas(self) -> list[dict]:
        """OpenAI function schemas for the exposed tools (requires :meth:`reset`)."""
        if self.toolbox is None:
            raise RuntimeError("Environment not started; call reset() first")
        return self.toolbox.schemas()

    async def call(self, name: str, args: dict[str, Any] | None = None) -> Observation:
        """Dispatch one tool call and return its :class:`Observation` (one step)."""
        if self.toolbox is None:
            raise RuntimeError("Environment not started; call reset() first")
        return await self.toolbox.call(name, args)

    async def close(self) -> None:
        """Close the toolbox (release open channels), then stop the sandbox."""
        if self.toolbox is not None:
            await self.toolbox.close()
            self.toolbox = None
        try:
            await self.sandbox.stop()
        finally:
            self._started = False

    async def __aenter__(self) -> Environment:
        await self.reset()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
