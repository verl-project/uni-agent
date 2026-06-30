"""Sandbox providers -- one module per provider, one class per provider.

Each provider is a single :class:`Sandbox` subclass that owns its lifecycle and
*is* the data plane (exec + file transfer + optional port tunnel). Tools and the
channels they hold depend only on the narrow :class:`SandboxBackend` protocol
(the data-plane subset), never on lifecycle.

Providers self-register by name via :func:`register_sandbox`, and are resolved by
name through :func:`build_sandbox` (mirrors the tools / rewards / tasks
registries). Provider modules are imported *lazily* -- :func:`build_sandbox`
imports the selected provider's module on first use (see ``SANDBOX_MODULES``) --
so importing this package never forces every provider's third-party SDK to be
installed; you only pay for the provider you select. ``base`` and host-local
providers pull in no third-party *provider* SDK (only the stdlib + pydantic) and
are imported eagerly here.

Add a provider by dropping ``<name>.py`` next to this file, decorating its class
with ``@register_sandbox("<name>")``, and adding it to ``SANDBOX_MODULES``.
"""

from __future__ import annotations

from .base import ExecResult, Sandbox, SandboxBackend, SandboxConfig
from .registry import (
    SANDBOX_MODULES,
    SANDBOX_REGISTRY,
    build_sandbox,
    get_sandbox_cls,
    register_sandbox,
)

# Host-local provider: stdlib-only, so import (and register) it eagerly and
# export the class for direct use. Heavier providers (e.g. ``modal``) stay lazy.
from .local import LocalSandbox

__all__ = [
    "ExecResult",
    "SandboxBackend",
    "Sandbox",
    "SandboxConfig",
    "LocalSandbox",
    "build_sandbox",
    "register_sandbox",
    "get_sandbox_cls",
    "SANDBOX_REGISTRY",
    "SANDBOX_MODULES",
]
