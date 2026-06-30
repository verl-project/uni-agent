"""Sandbox providers -- one module per provider, one class per provider.

Each provider is a single :class:`Sandbox` subclass that owns its lifecycle and
*is* the data plane (exec + file transfer + optional port tunnel). Tools and the
channels they hold depend only on the narrow :class:`SandboxBackend` protocol
(the data-plane subset), never on lifecycle.
"""

from __future__ import annotations

from .base import ExecResult, Sandbox, SandboxBackend, SandboxConfig

# Host-local provider: stdlib-only, so import (and register) it eagerly and
# export the class for direct use. Heavier providers (e.g. ``modal``) stay lazy.
from .local import LocalSandbox
from .registry import (
    SANDBOX_MODULES,
    SANDBOX_REGISTRY,
    build_sandbox,
    get_sandbox_cls,
    register_sandbox,
)

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
