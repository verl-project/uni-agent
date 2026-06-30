"""Environment layer: compose a sandbox + a toolbox into one runnable world.

The agent loop talks to an :class:`Environment` (reset / call / close); the
environment owns provider selection, ordered lifecycle, and tool dispatch. Each
tool is bound to the sandbox and owns its own state (a stateful tool keeps its
channel internally), so there is no separate session layer. Build one from
config::

    from uni_agent.environment import Environment

    env = Environment.from_config({
        "sandbox": {"provider": "modal", "runtime_timeout": 3600},
        "tools": [{"name": "shell", "command_timeout": 120}, {"name": "str_replace_editor"}],
    })
    async with env:
        schemas = env.tool_schemas()
        obs = await env.call("shell", {"command": "echo hi"})
"""

from __future__ import annotations

from .config import EnvironmentConfig, SandboxConfig
from .environment import Environment, build_sandbox, build_tool

__all__ = [
    "Environment",
    "EnvironmentConfig",
    "SandboxConfig",
    "build_sandbox",
    "build_tool",
]
