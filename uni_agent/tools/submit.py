"""``submit``: signal that the task is complete.

A control tool with no arguments and no sandbox side effect: the solution lives in
the sandbox (e.g. an applied patch) which the task scores after the loop ends. The
ReAct loop stops the episode when the policy calls it (see ``_FINISH_TOOLS``). For
text-answer tasks use ``finish`` instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .base import Tool, ToolResult, register_tool

DESCRIPTION = """
A simple submit tool to finish tasks.
This tool signals completion of a task or submission of results.
""".strip()


class SubmitArguments(BaseModel):
    # No fields: submitting is a bare completion signal. (A docstring here would
    # leak onto the parameters schema as a description, so keep it a comment.)
    pass


@register_tool("submit")
class SubmitTool(Tool):
    name = "submit"
    description = DESCRIPTION
    args_model = SubmitArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        return ToolResult(text="Submitted.")
