"""Agent registry: register an agent by name and build it by name.

Mirrors the sandbox / reward / tools / tasks registries. Each agent is an
:class:`~uni_agent.agents.base.Agent` subclass living under ``agents/<name>/``
that registers itself with :func:`register_agent`. :func:`build_agent` resolves
an agent by ``config.name``, importing its module on first use
(:data:`AGENT_MODULES`) so an agent whose extra deps aren't installed never
blocks importing this package -- you only pay for the agent you actually select.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from .base import Agent, AgentConfig

AGENT_REGISTRY: dict[str, type[Agent]] = {}

#: agent name -> module that defines (and registers) it, for lazy loading.
AGENT_MODULES: dict[str, str] = {
    "code_act": "uni_agent.agents.code_act.agent",
    "claude_code": "uni_agent.agents.claude_code.agent",
}


def register_agent(name: str) -> Callable[[type[Agent]], type[Agent]]:
    """Class decorator: register an :class:`Agent` under ``name`` (and stamp ``cls.name``)."""

    def decorator(cls: type[Agent]) -> type[Agent]:
        if name in AGENT_REGISTRY and AGENT_REGISTRY[name] is not cls:
            raise ValueError(f"Agent {name!r} already registered: {AGENT_REGISTRY[name]!r} vs {cls!r}")
        cls.name = name
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def _load_agent_module(name: str) -> None:
    """Import the module that registers agent ``name`` (no-op if unknown)."""
    module_name = AGENT_MODULES.get(name)
    if module_name is None:
        return
    try:
        import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import agent {name!r} from {module_name!r}. Install the optional dependencies it needs."
        ) from exc


def get_agent_cls(name: str) -> type[Agent]:
    """Return a registered agent class by name, importing its module on first use."""
    if name not in AGENT_REGISTRY:
        _load_agent_module(name)
    if name not in AGENT_REGISTRY:
        available = sorted(set(AGENT_REGISTRY) | set(AGENT_MODULES))
        raise ValueError(f"Unknown agent: {name!r}. Available: {available}")
    return AGENT_REGISTRY[name]


def build_agent(config: AgentConfig) -> Agent:
    """Instantiate the agent named by ``config.name`` from its config."""
    return get_agent_cls(config.name).from_config(config)
