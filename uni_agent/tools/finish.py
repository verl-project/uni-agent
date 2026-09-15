"""``finish``: end the interaction and hand back a final answer.

A control tool with no sandbox side effect: the ReAct loop stops the episode when
the policy calls it (see ``_FINISH_TOOLS``). For tasks where the reply is the
deliverable (QA); code tasks use ``submit`` (the solution lives in the sandbox).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import Tool, ToolResult, register_tool

DESCRIPTION = """
Finish the task and output the final answer.
Always call this tool when you are ready to end the interaction.
""".strip()


class FinishArguments(BaseModel):
    answer: str = Field(description="Final answer to return to the user.")


@register_tool("finish")
class FinishTool(Tool):
    name = "finish"
    description = DESCRIPTION
    args_model = FinishArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        # Echo the answer so it lands in the transcript; the loop ends the episode.
        return ToolResult(text=str(args.get("answer", "")))
