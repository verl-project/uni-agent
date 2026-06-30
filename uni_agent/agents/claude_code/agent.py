"""Claude Code: a black-box agent launched *inside* the sandbox.

The solver is an opaque Claude Code process that runs its own loop + tools in the
container, but it is **not** a different model: we point it at the task-created
gateway session (its per-session ``base_url``) so its model calls go through our
policy and become trainable trajectories. This agent therefore only configures the
*launch* -- command, model, turn budget, tool allow-list, how to inject the gateway
endpoint, extra env -- and reads back what it produced (the working-tree diff). The
task is expected to have provisioned the instance (clone repo @ base commit, ...)
and created the session before handing over the live sandbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from ...gateway.session.types import SessionHandle
    from ...sandbox import Sandbox


class ClaudeCodeConfig(AgentConfig):
    """Black-box launch params for Claude Code, pointed at the gateway session."""

    name: str = "claude_code"
    command: list[str] = Field(
        default_factory=lambda: ["claude", "-p"], description="Launch argv inside the sandbox."
    )
    model: str | None = Field(default="claude-sonnet-4", description="Model the agent should use.")
    max_turns: int | None = Field(default=50, description="Turn budget passed to the agent.")
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Bash", "Edit", "Read", "Write"], description="Tool allow-list."
    )
    # How to point the in-sandbox CLI at our per-session gateway endpoint instead
    # of a real provider. The gateway is OpenAI-compatible and self-hosted, so the
    # key value can be any non-empty string.
    base_url_env: str = Field(
        default="ANTHROPIC_BASE_URL",
        description="Env var the CLI reads for its API base; set to session.base_url.",
    )
    api_key_env: str = Field(
        default="ANTHROPIC_API_KEY", description="Env var the CLI reads for its API key."
    )
    api_key: str = Field(
        default="EMPTY", description="API key value (the self-hosted gateway accepts any non-empty)."
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Extra env vars injected into the launched process."
    )


@register_agent("claude_code")
class ClaudeCodeAgent(Agent):
    """Black-box solver: launch Claude Code in the sandbox, pointed at the gateway URL."""

    config_model = ClaudeCodeConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        sample: dict[str, Any],
        session: SessionHandle,
    ) -> AgentResult:
        # The black-box loop runs in the sandbox, but its model is *our* policy:
        # it needs the per-session gateway URL to reach it.
        if session.base_url is None:
            raise ValueError("claude_code needs session.base_url to route the model through the gateway")
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        argv = [
            *cfg.command,
            *(["--model", cfg.model] if cfg.model else []),
            *(["--max-turns", str(cfg.max_turns)] if cfg.max_turns is not None else []),
            *(["--allowedTools", ",".join(cfg.allowed_tools)] if cfg.allowed_tools else []),
            sample.get("problem_statement", ""),
        ]
        # Route the CLI's model calls through our per-session gateway URL (so they
        # become trainable trajectories) -- it never talks to a real provider.
        env = {
            **cfg.env,
            cfg.base_url_env: session.base_url,
            cfg.api_key_env: cfg.api_key,
        }
        # No client-side timeout: the sandbox's runtime_timeout bounds the run.
        proc = await sandbox.exec(argv, env=env)
        patch = (await sandbox.exec_shell("git diff")).stdout
        return AgentResult(
            output={"patch": patch, "agent_stdout": proc.stdout, "exit_code": proc.exit_code}
        )
