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


_CLAUDE_NPM_INSTALL_COMMAND = "npm install -g @anthropic-ai/claude-code --no-audit --no-fund"
_CLAUDE_NATIVE_INSTALL_COMMAND = r"""
set -euo pipefail
if command -v curl >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh
elif command -v wget >/dev/null 2>&1; then
  wget -qO- https://claude.ai/install.sh
else
  echo "Claude Code native install requires curl or wget" >&2
  exit 1
fi | bash -s stable

claude_bin="${HOME:-/root}/.local/bin/claude"
if [ ! -x "${claude_bin}" ]; then
  echo "native installer did not create ${claude_bin}" >&2
  exit 1
fi
ln -sf "${claude_bin}" /usr/local/bin/claude
""".strip()
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
    enable_web_tools: bool = Field(
        default=False,
        description="Allow Claude Code to use WebFetch and WebSearch during a rollout.",
    )
    enable_subagents: bool = Field(
        default=False,
        description="Allow Claude Code to dispatch subagents through the Agent/Task tool.",
    )
    disable_slash_commands: bool = Field(
        default=True,
        description="Disable Claude Code skills and slash commands for deterministic rollouts.",
    )
    verbose: bool = Field(default=False, description="Pass --verbose (streams per-turn detail; noisy at scale).")
    run_timeout: float = Field(
        default=1800.0,
        description="Maximum wall-clock time (s) for Claude Code execution.",
    )
    extra_args: list[str] = Field(default_factory=list, description="Extra flags appended to the claude argv.")
    extra_env: dict[str, str] = Field(default_factory=dict, description="Extra env for the claude process.")


@register_agent("claude_code")
class ClaudeCodeAgent(Agent):
    """Black-box solver: launch the real Claude Code CLI in the sandbox against ``config.model``."""

    config_model = ClaudeCodeConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        base_url = cfg.model.base_url
        if not base_url:
            raise ValueError("claude_code: config.model.base_url is not set (the gateway/vLLM policy endpoint)")
        user_messages = [message.get("content") for message in messages if message.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError("claude_code requires exactly one 'user' message")
        user_prompt = user_messages[0]
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("claude_code requires a non-empty user prompt")

        await self._ensure_claude(sandbox)
        # Let the agent's git commands trust the repo even if it's owned by another uid.
        await sandbox.exec_shell("git config --system safe.directory '*' || true")

        # Point claude at the Anthropic endpoint (gateway session or vLLM) and run it.
        endpoint = _strip_v1(base_url)
        argv = self._claude_argv(user_prompt)
        env = self._claude_env(endpoint)
        logger.info("claude_code: launch with user_prompt:\n%s", user_prompt)
        proc = await sandbox.exec(argv, env=env, timeout=cfg.run_timeout, workdir=workdir)

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

        return AgentResult(
            info={"exit_code": proc.exit_code, "stdout_tail": out_tail, "stderr_tail": err_tail},
            finished=proc.exit_code == 0,
        )

    # ----- helpers -----
    async def _ensure_claude(self, sandbox: Sandbox) -> None:
        if (await sandbox.exec_shell("command -v claude >/dev/null 2>&1")).exit_code == 0:
            return

        has_npm = (await sandbox.exec_shell("command -v npm >/dev/null 2>&1")).exit_code == 0
        install_method = "npm" if has_npm else "native installer"
        install_command = _CLAUDE_NPM_INSTALL_COMMAND if has_npm else _CLAUDE_NATIVE_INSTALL_COMMAND
        logger.info("claude_code: claude not found; installing it with %s", install_method)
        result = await sandbox.exec_shell(install_command, timeout=_CLAUDE_INSTALL_TIMEOUT)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()[-2000:]
            raise RuntimeError(f"claude_code: failed to install Claude Code with {install_method}: {detail}")

        if (await sandbox.exec_shell("command -v claude >/dev/null 2>&1")).exit_code != 0:
            raise RuntimeError("claude_code: installation finished but claude is not available on PATH")
        logger.info("claude_code: installation completed")

    def _claude_argv(self, user_prompt: str) -> list[str]:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        model = cfg.model.model_name
        if not model:
            raise ValueError("claude_code: set config.model.model_name (the model claude sends)")
        argv = [
            "claude",
            "-p",
            user_prompt,
            "--model",
            model,
            "--permission-mode",
            "bypassPermissions",
        ]
        if cfg.disable_slash_commands:
            argv.append("--disable-slash-commands")
        disallowed_tools = ["AskUserQuestion"]
        if not cfg.enable_subagents:
            disallowed_tools.extend(["Agent", "Task"])
        if not cfg.enable_web_tools:
            disallowed_tools.extend(["WebFetch", "WebSearch"])
        argv += ["--disallowedTools", ",".join(disallowed_tools)]
        if cfg.max_turns is not None:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.verbose:
            argv += ["--verbose"]
        return argv + list(cfg.extra_args)

    def _claude_env(self, endpoint: str) -> dict[str, str]:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        model = cfg.model.model_name
        if not model:
            raise ValueError("claude_code: set config.model.model_name (the model claude sends)")
        auth_token = cfg.model.api_key
        if not auth_token or auth_token == "EMPTY":
            auth_token = str(uuid.uuid4())
        env = {
            "IS_SANDBOX": "1",
            "ANTHROPIC_BASE_URL": endpoint,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "ANTHROPIC_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
            "API_TIMEOUT_MS": "86400000",  # 24 hours
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HTTP_PROXY": "",
            "http_proxy": "",
            "HTTPS_PROXY": "",
            "https_proxy": "",
        }
        if cfg.enable_subagents:
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = model
        env.update(cfg.extra_env)
        return env
