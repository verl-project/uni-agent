"""Claude Code: a black-box agent launched *inside* the sandbox."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from ...sandbox import Sandbox


class ClaudeCodeConfig(AgentConfig):
    """Black-box launch params for Claude Code (endpoint is supplied at run time)."""

    name: str = "claude_code"
    command: list[str] = Field(default_factory=lambda: ["claude", "-p"], description="Launch argv inside the sandbox.")
    model: str | None = Field(default="claude-sonnet-4", description="Model the agent should use.")
    max_turns: int | None = Field(default=50, description="Turn budget passed to the agent.")
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Bash", "Edit", "Read", "Write"], description="Tool allow-list."
    )


@register_agent("claude_code")
class ClaudeCodeAgent(Agent):
    """Black-box solver: launch Claude Code in the sandbox, pointed at ``base_url``."""

    config_model = ClaudeCodeConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        base_url: str,
        api_key: str,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        # Claude Code owns its loop + tools, so we can only seed it: a required user
        # turn (the problem statement) and at most one optional system turn.
        if len(messages) > 2:
            raise ValueError(f"claude_code accepts at most 2 messages (system?, user), got {len(messages)}")
        problem_statement = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem_statement:
            raise ValueError("claude_code requires a 'user' message (the problem statement)")
        system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), None)
        argv = [
            *cfg.command,
            *(["--model", cfg.model] if cfg.model else []),
            *(["--max-turns", str(cfg.max_turns)] if cfg.max_turns is not None else []),
            *(["--allowedTools", ",".join(cfg.allowed_tools)] if cfg.allowed_tools else []),
            # Append (not replace) so Claude Code keeps its built-in tool/safety prompt.
            *(["--append-system-prompt", system_prompt] if system_prompt else []),
            problem_statement,
        ]
        # Point the in-sandbox CLI at our endpoint via the env vars it reads.
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": api_key,
        }
        # No client-side timeout: the sandbox's runtime_timeout bounds the run.
        proc = await sandbox.exec(argv, env=env)
        patch = (await sandbox.exec_shell("git diff")).stdout
        return AgentResult(output={"patch": patch, "agent_stdout": proc.stdout, "exit_code": proc.exit_code})
