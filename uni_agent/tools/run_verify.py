"""``run_verify``: AST check, correctness, then a performance benchmark.

The command is the image-owned verifier. It takes no model arguments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import Tool, ToolResult, register_tool

DESCRIPTION = """
Run AST check + correctness verify + performance benchmark (speedup_vs_torch).
No arguments needed. Call it after writing code or after any optimization.
If AST fails or any case fails, the benchmark is skipped.
""".strip()


class RunVerifyArguments(BaseModel):
    max_error_chars: int = Field(default=2000, description="Unused. Kept so older prompts stay valid.")


@register_tool("run_verify")
class RunVerifyTool(Tool):
    name = "run_verify"
    description = DESCRIPTION
    args_model = RunVerifyArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        del args
        command = "/opt/triton-agent-tools/verify_once.sh"
        result = await self.sandbox.exec_shell(command, timeout=timeout or 900)
        text = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        if result.exit_code != 0:
            return ToolResult(text=text.strip() or f"{command} exited {result.exit_code}", status="error")
        return ToolResult(text=text.strip() or "verify finished")
