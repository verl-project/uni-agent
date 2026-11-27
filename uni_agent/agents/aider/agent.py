"""Aider black-box agent adapter.

The Aider CLI is supplied by a prebuilt sidecar mounted at ``/opt/aider``.
This host-side adapter maps the model endpoint to Aider's OpenAI-compatible
flags, passes the prompt by file, launches one headless Aider run inside the
sandbox, and converts the process result into ``AgentResult``. The Task scores
the uncommitted working-tree diff left by ``--no-auto-commits``.
"""

from __future__ import annotations

import logging
import re
import shlex
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_MAX_DIAGNOSTIC_CHARS = 4000


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    """Require the single text message passed to Aider's message file."""
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("aider requires exactly one 'user' message")
    prompt = messages[0].get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("aider requires non-empty user text")
    return prompt


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        if secret and secret != "EMPTY":
            value = value.replace(secret, "<redacted>")
        return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1<redacted>", value)
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if str(key).lower() in {"api_key", "apikey", "authorization", "token", "password"}
            else _redact(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _diagnostic(value: str | None, secret: str = "") -> str:
    return _redact(value or "", secret).strip()[-_MAX_DIAGNOSTIC_CHARS:]


class AiderConfig(AgentConfig):
    """Black-box launch parameters bound to the Aider sidecar layout."""

    name: str = "aider"
    run_timeout: float = Field(
        default=1800, gt=0, description="Sandbox exec timeout in seconds for the headless Aider run."
    )
    edit_format: str | None = Field(
        default=None,
        description="Optional aider --edit-format (e.g. whole, diff, udiff); leave unset for aider defaults.",
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Additional CLI args appended after the managed flags."
    )
    state_root: str = Field(default="/tmp/uni-agent-aider")


@register_agent("aider")
class AiderAgent(Agent):
    """Run the pinned Aider CLI in a sandbox through an OpenAI-compatible endpoint."""

    config_model = AiderConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]], workdir: str | None = None) -> AgentResult:
        cfg: AiderConfig = self.config  # type: ignore[assignment]
        model = cfg.model
        if not model.base_url or not model.model_name:
            raise ValueError("aider requires config.model.base_url and config.model.model_name")
        prompt = _extract_user_prompt(messages)

        workspace = workdir or "/testbed"
        episode_id = uuid.uuid4().hex
        root = str(PurePosixPath(cfg.state_root) / episode_id)
        prompt_path = f"{root}/task.txt"
        history_path = f"{root}/input-history"
        prepared = await sandbox.exec_shell(f"umask 077; mkdir -p {shlex.quote(root)}", timeout=30)
        if prepared.exit_code != 0:
            return AgentResult(
                finished=False,
                info={"error_kind": "startup_failure", "exit_code": prepared.exit_code, "session_id": episode_id},
            )
        failure = None
        try:
            await sandbox.write_file(prompt_path, prompt)
            argv = [
                "aider",
                "--model",
                f"openai/{model.model_name}",
                "--openai-api-base",
                model.base_url,
                "--message-file",
                prompt_path,
                "--exit",
                "--yes-always",
                "--no-auto-commits",
                "--no-analytics",
                "--no-check-update",
                "--no-restore-chat-history",
                "--no-browser",
                "--no-suggest-shell-commands",
                "--no-show-model-warnings",
                "--input-history-file",
                history_path,
            ]
            if cfg.edit_format:
                argv += ["--edit-format", cfg.edit_format]
            argv += list(cfg.extra_args)
            # The API key travels through the private exec environment, never argv.
            env = {"HOME": root, "OPENAI_API_KEY": model.api_key}
            proc = await sandbox.exec(argv, env=env, timeout=cfg.run_timeout, workdir=workspace)
            if proc.exit_code == -1:
                failure = {"error_kind": "timeout", "error_type": "sandbox_timeout", "exit_code": -1}
        except (TimeoutError, OSError) as exc:
            failure = {
                "error_kind": "timeout" if isinstance(exc, TimeoutError) else "startup_failure",
                "error_type": type(exc).__name__,
                "exit_code": -1 if isinstance(exc, TimeoutError) else None,
            }

        info: dict[str, Any] = {
            "session_id": episode_id,
            "state_dir": root,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        if failure is not None:
            info.update(failure)
            return AgentResult(finished=False, info=_redact(info, model.api_key))
        info.update(
            exit_code=proc.exit_code,
            stdout_tail=_diagnostic(proc.stdout, model.api_key),
            stderr_tail=_diagnostic(proc.stderr, model.api_key),
        )
        if proc.exit_code != 0:
            info["error_kind"] = "agent_failure"
        return AgentResult(
            output={"session_id": episode_id},
            transcript=list(messages),
            info=_redact(info, model.api_key),
            finished=proc.exit_code == 0,
        )
