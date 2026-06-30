"""Sandbox registry: register a provider by name and build it by name.

Mirrors the reward / tools / tasks registries. Each provider is a
:class:`~uni_agent.sandbox.base.Sandbox` subclass living in its own module that
registers itself with :func:`register_sandbox`. :func:`build_sandbox` resolves a
provider by name, importing its module on first use (:data:`SANDBOX_MODULES`) so
a provider whose third-party SDK isn't installed never blocks importing this
package -- you only pay for the provider you actually select.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from .base import Sandbox, SandboxConfig

SANDBOX_REGISTRY: dict[str, type[Sandbox]] = {}

#: provider name -> module that defines (and registers) it, for lazy loading.
SANDBOX_MODULES: dict[str, str] = {
    "local": "uni_agent.sandbox.local",
    "modal": "uni_agent.sandbox.modal",
}


def register_sandbox(name: str) -> Callable[[type[Sandbox]], type[Sandbox]]:
    """Class decorator: register a :class:`Sandbox` provider under ``name`` (and stamp ``cls.provider``)."""

    def decorator(cls: type[Sandbox]) -> type[Sandbox]:
        if name in SANDBOX_REGISTRY and SANDBOX_REGISTRY[name] is not cls:
            raise ValueError(f"Sandbox provider {name!r} already registered: {SANDBOX_REGISTRY[name]!r} vs {cls!r}")
        cls.provider = name
        SANDBOX_REGISTRY[name] = cls
        return cls

    return decorator


def _load_sandbox_module(name: str) -> None:
    """Import the module that registers provider ``name`` (no-op if unknown)."""
    module_name = SANDBOX_MODULES.get(name)
    if module_name is None:
        return
    try:
        import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import sandbox provider {name!r} from {module_name!r}. "
            f"Install the optional dependencies it needs (e.g. `pip install modal` for provider='modal')."
        ) from exc


def get_sandbox_cls(name: str) -> type[Sandbox]:
    """Return a registered provider class by name, importing its module on first use."""
    if name not in SANDBOX_REGISTRY:
        _load_sandbox_module(name)
    if name not in SANDBOX_REGISTRY:
        available = sorted(set(SANDBOX_REGISTRY) | set(SANDBOX_MODULES))
        raise ValueError(f"Unknown sandbox provider: {name!r}. Available: {available}")
    return SANDBOX_REGISTRY[name]


def build_sandbox(config: SandboxConfig) -> Sandbox:
    """Instantiate the sandbox provider named by ``config.provider`` from its config."""
    return get_sandbox_cls(config.provider).from_config(config)
