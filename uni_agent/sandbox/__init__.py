"""Sandbox providers -- one module per provider, one :class:`Sandbox` subclass each.

A provider owns its lifecycle and is the data plane (exec + file transfer +
optional port tunnel); tools depend only on the narrow :class:`SandboxBackend`
protocol, never on lifecycle.
"""

from __future__ import annotations

from .base import ExecResult, Sandbox, SandboxBackend, SandboxConfig

# Host-local provider is stdlib-only: import (and register) it eagerly. Heavier
# providers (e.g. ``modal``) stay lazy.
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
