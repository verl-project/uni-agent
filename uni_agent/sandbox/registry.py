"""Sandbox registry: register a provider by name, build it by name.

Providers live in their own module and self-register via
:func:`register_sandbox`; :func:`build_sandbox` imports that module lazily on
first use, so an uninstalled provider SDK never blocks importing this package.

External overlays (private backends) can attach extra providers
without adding them to :data:`SANDBOX_MODULES`. Set ``UNI_AGENT_SANDBOX_PLUGINS``
to a comma-separated list of importable module names. Each module must call
:func:`register_sandbox` at import time. Missing modules fail closed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module

from .base import Sandbox, SandboxConfig

SANDBOX_REGISTRY: dict[str, type[Sandbox]] = {}

#: Comma-separated importable modules that call :func:`register_sandbox`.
SANDBOX_PLUGIN_MODULES_ENV = "UNI_AGENT_SANDBOX_PLUGINS"

#: provider name -> module that defines (and registers) it, for lazy loading.
SANDBOX_MODULES: dict[str, str] = {
    "local": "uni_agent.sandbox.local",
    "docker": "uni_agent.sandbox.docker",
    "modal": "uni_agent.sandbox.modal",
    "vefaas": "uni_agent.sandbox.vefaas",
    "openyuanrong": "uni_agent.sandbox.openyuanrong",
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


def _plugin_module_names() -> tuple[str, ...]:
    raw = os.environ.get(SANDBOX_PLUGIN_MODULES_ENV, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _load_plugin_modules() -> None:
    """Import overlay modules so they can self-register. ``import_module`` is cached."""
    for module_name in _plugin_module_names():
        try:
            import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import sandbox plugin module {module_name!r} from {SANDBOX_PLUGIN_MODULES_ENV}."
            ) from exc


def _load_sandbox_module(name: str) -> None:
    """Import the in-tree module that registers provider ``name`` (no-op if unknown)."""
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
        _load_plugin_modules()
    if name not in SANDBOX_REGISTRY:
        available = sorted(set(SANDBOX_REGISTRY) | set(SANDBOX_MODULES))
        raise ValueError(f"Unknown sandbox provider: {name!r}. Available: {available}")
    return SANDBOX_REGISTRY[name]


def build_sandbox(config: SandboxConfig) -> Sandbox:
    """Instantiate the sandbox provider named by ``config.provider`` from its config."""
    return get_sandbox_cls(config.provider).from_config(config)
