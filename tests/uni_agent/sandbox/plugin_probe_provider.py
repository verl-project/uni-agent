"""Test-only overlay provider imported via ``UNI_AGENT_SANDBOX_PLUGINS``."""

from __future__ import annotations

from uni_agent.sandbox.base import ExecResult, Sandbox
from uni_agent.sandbox.registry import register_sandbox


@register_sandbox("plugin_probe")
class PluginProbeSandbox(Sandbox):
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        return ExecResult(exit_code=0, stdout="", stderr="")
