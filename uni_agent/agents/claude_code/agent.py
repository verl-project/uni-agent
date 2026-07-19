"""Claude Code: a black-box agent that runs the real ``claude`` CLI inside the sandbox.

Claude Code speaks the *Anthropic Messages* protocol (``POST /v1/messages``), served
natively by both modern vLLM (direct) and the uni-agent gateway session (training
path). So we point ``ANTHROPIC_BASE_URL`` at ``config.model.base_url`` (trailing
``/v1`` stripped, since claude re-appends ``/v1/messages``) and let the server parse
the tool calls -- vLLM's parser directly, or the gateway codec on the training path.
No proxy process.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_CC_QUIET_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_BUG_COMMAND": "1",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
}

_CLAUDE_INSTALL_COMMAND = "npm install -g @anthropic-ai/claude-code --no-audit --no-fund"
_CLAUDE_INSTALL_TIMEOUT = 600


def _strip_v1(base_url: str) -> str:
    """Drop a trailing ``/v1`` from an OpenAI-style base URL to get the Anthropic root.

    The Anthropic endpoint lives at ``<root>/v1/messages`` and Claude Code appends
    ``/v1/messages`` to ``ANTHROPIC_BASE_URL`` itself, so an OpenAI base (ending in
    ``/v1``) must be reduced to its root -- for both transports: direct vLLM
    ``http://h:8000/v1`` -> ``http://h:8000``, and a gateway session
    ``http://h:8000/sessions/<id>/v1`` -> ``http://h:8000/sessions/<id>``. (A bare host
    is returned unchanged; skipping the strip yields a broken ``/v1/v1/messages``.)
    """
    b = base_url.rstrip("/")
    return b[:-3].rstrip("/") if b.endswith("/v1") else b


class ClaudeCodeConfig(AgentConfig):
    """Black-box launch params for Claude Code (policy endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "claude_code"
    max_turns: int | None = Field(default=80, description="--max-turns budget; None to omit.")
    disallowed_tools: list[str] = Field(
        default_factory=lambda: ["WebFetch", "WebSearch", "AskUserQuestion"],
        description=(
            "--disallowedTools deny-list. Under --dangerously-skip-permissions an *allow*-list is a "
            "no-op (bypass approves every tool), so we DENY the tools that can't work in a headless, "
            "offline sandbox: web tools (no egress) and AskUserQuestion (would hang on input). A bare "
            "tool name drops it from Claude's context entirely, and deny wins even under bypass."
        ),
    )
    verbose: bool = Field(default=False, description="Pass --verbose (streams per-turn detail; noisy at scale).")
    run_timeout: float = Field(default=1800.0, description="Wallclock cap (s) on the claude process.")
    extra_args: list[str] = Field(default_factory=list, description="Extra flags appended to the claude argv.")
    extra_env: dict[str, str] = Field(default_factory=dict, description="Extra env for the claude process.")


@register_agent("claude_code")
class ClaudeCodeAgent(Agent):
    """Black-box solver: launch the real Claude Code CLI in the sandbox against ``config.model``."""

    config_model = ClaudeCodeConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]]) -> AgentResult:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        base_url = cfg.model.base_url
        if not base_url:
            raise ValueError("claude_code: config.model.base_url is not set (the gateway/vLLM policy endpoint)")
        system_prompt, problem = self._split_messages(messages)

        await self._ensure_claude(sandbox)
        # Let the agent's git commands trust the repo even if it's owned by another uid.
        await sandbox.exec_shell("git config --system safe.directory '*' || true")

        # Point claude at the Anthropic endpoint (gateway session or vLLM) and run it.
        endpoint = _strip_v1(base_url)
        argv = self._claude_argv(problem, system_prompt)
        env = self._claude_env(endpoint)
        logger.info("claude_code: launch (endpoint=%s)", endpoint)
        proc = await sandbox.exec(argv, env=env, timeout=cfg.run_timeout)

        out_tail = (proc.stdout or "").strip()[-2000:]
        err_tail = (proc.stderr or "").strip()[-2000:]
        if proc.exit_code != 0:
            logger.warning(
                "claude_code: claude exited %s\n--- stdout (tail) ---\n%s\n--- stderr (tail) ---\n%s",
                proc.exit_code,
                out_tail,
                err_tail,
            )
        else:
            logger.info("claude_code: claude finished (exit 0)\n--- stdout (tail) ---\n%s", out_tail)

        return AgentResult(info={"exit_code": proc.exit_code, "stdout_tail": out_tail, "stderr_tail": err_tail})

    # ----- helpers -----
    async def _ensure_claude(self, sandbox: Sandbox) -> None:
        if (await sandbox.exec_shell("command -v claude >/dev/null 2>&1")).exit_code == 0:
            return

        logger.info("claude_code: claude not found; installing it inside the sandbox")
        result = await sandbox.exec_shell(_CLAUDE_INSTALL_COMMAND, timeout=_CLAUDE_INSTALL_TIMEOUT)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()[-2000:]
            raise RuntimeError(f"claude_code: failed to install Claude Code: {detail}")

        if (await sandbox.exec_shell("command -v claude >/dev/null 2>&1")).exit_code != 0:
            raise RuntimeError("claude_code: installation finished but claude is not available on PATH")
        logger.info("claude_code: installation completed")

    def _split_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, str]:
        if len(messages) > 2:
            raise ValueError(f"claude_code accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem:
            raise ValueError("claude_code requires a 'user' message (the problem statement)")
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        return system, problem

    def _claude_argv(self, problem: str, system_prompt: str | None) -> list[str]:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        argv = ["claude", "-p", problem]
        if cfg.disallowed_tools:
            argv += ["--disallowedTools", ",".join(cfg.disallowed_tools)]
        if cfg.max_turns is not None:
            argv += ["--max-turns", str(cfg.max_turns)]
        if system_prompt:
            # Append (don't replace) so Claude Code keeps its built-in tool/safety prompt.
            argv += ["--append-system-prompt", system_prompt]
        # Headless runs must not block on permission prompts.
        argv += ["--dangerously-skip-permissions"]
        if cfg.verbose:
            argv += ["--verbose"]
        return argv + list(cfg.extra_args)

    def _claude_env(self, endpoint: str) -> dict[str, str]:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        model = cfg.model.model_name
        if not model:
            raise ValueError("claude_code: set config.model.model_name (the model claude sends)")
        return {
            "ANTHROPIC_BASE_URL": endpoint,
            # We always run inside a sandbox: lets `--dangerously-skip-permissions` run as
            # root (else the CLI refuses) and skips its 529-overload guard path.
            "IS_SANDBOX": "1",
            # The server ignores the key/token values, but the CLI requires both to be set.
            "ANTHROPIC_API_KEY": "sk-ant-uni-agent-placeholder",
            "ANTHROPIC_AUTH_TOKEN": str(uuid.uuid4()),
            # Route every model slot to our single served model. Besides the main tiers,
            # Claude Code fires *background*/subagent calls (summaries, sub-tasks) on the
            # haiku + subagent slots. On direct vLLM leaving those unset 404s on a name it
            # doesn't serve; the gateway ignores the model name, but pinning is harmless
            # and keeps both paths identical, so pin them all to `model`.
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            # claude only needs to reach ANTHROPIC_BASE_URL (the gateway node, or a direct
            # vLLM host); it's reachable directly, so strip the sandbox's injected egress proxy.
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HTTP_PROXY": "",
            "http_proxy": "",
            "HTTPS_PROXY": "",
            "https_proxy": "",
            **_CC_QUIET_ENV,
            **cfg.extra_env,
        }
