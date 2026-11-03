"""``list_skills`` and ``read_skill`` over the sandbox workspace."""

from __future__ import annotations

import shlex
from typing import Any

from pydantic import BaseModel, Field

from .base import Tool, ToolResult, register_tool


class ListSkillsArguments(BaseModel):
    pass


@register_tool("list_skills")
class ListSkillsTool(Tool):
    name = "list_skills"
    description = "List available skills with name, description, and path."
    args_model = ListSkillsArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        del args
        script = (
            "for root in /workspace/.skills /opt/triton-agent-template/.skills "
            "/opt/triton-agent-template/.claude/skills; do "
            "if [ -d \"$root\" ]; then find \"$root\" -name SKILL.md -print; fi; done"
        )
        result = await self.sandbox.exec_shell(script, timeout=timeout or 30)
        text = (result.stdout or "").strip()
        if result.exit_code != 0:
            return ToolResult(text=result.stderr.strip() or "list_skills failed", status="error")
        return ToolResult(text=text or "(no skills found)")


class ReadSkillArguments(BaseModel):
    path: str = Field(description="Path of a SKILL.md or reference file returned by list_skills.")


@register_tool("read_skill")
class ReadSkillTool(Tool):
    name = "read_skill"
    description = "Read a skill SKILL.md or one of its reference files."
    args_model = ReadSkillArguments

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        path = str(args.get("path", ""))
        if not path.startswith(("/workspace/", "/opt/triton-agent-template/")):
            return ToolResult(text="path must stay inside the workspace or the skill template", status="error")
        result = await self.sandbox.exec_shell(f"sed -n '1,400p' {shlex.quote(path)}", timeout=timeout or 30)
        if result.exit_code != 0:
            return ToolResult(text=result.stderr.strip() or f"cannot read {path}", status="error")
        return ToolResult(text=result.stdout or "")
