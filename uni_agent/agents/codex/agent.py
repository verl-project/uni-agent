"""Codex CLI black-box agent for the updated uni-agent architecture.

The host-side agent only owns command construction and result normalization.
The Codex executable and its native resources live in a sidecar mounted inside
the sandbox at ``/opt/codex``. The sandbox is the outer security boundary, so
Codex is invoked with its own approvals/sandbox bypass enabled.
"""

from __future__ import annotations

import base64
import json
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


def _diagnostic(value: str | None, secret: str = "") -> str:
    text = value or ""
    if secret and secret != "EMPTY":
        text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1<redacted>", text)
    return text.strip()[-_MAX_DIAGNOSTIC_CHARS:]


def build_agent_command(
    *,
    task_b64: str,
    tool_script: str,
    gateway_url: str,
    model_name: str,
    api_key: str,
    conda_env_path: str | None = None,
    path: str | None = None,
    project_dir: str = "/testbed",
) -> str:
    """Build the shell command that pipes a task into the Codex sidecar.

    The prompt is base64-encoded to avoid argv length limits and shell quoting
    hazards. Gateway and process settings are passed as environment variables
    consumed by the sidecar entrypoint.
    """
    conda_env_vars = ""
    if conda_env_path:
        env_dir = PurePosixPath(conda_env_path)
        conda_env_vars = (
            f"CONDA_DEFAULT_ENV={shlex.quote(env_dir.name)} "
            f"CONDA_PREFIX={shlex.quote(str(env_dir))} "
        )
    path_value = path
    if path_value is None and conda_env_path:
        env_dir = PurePosixPath(conda_env_path)
        path_value = f"{env_dir / 'bin'}:{env_dir.parent.parent / 'bin'}"
    path_setup = f"PATH={shlex.quote(path_value)}:\"$PATH\" " if path_value else ""
    env = (
        f"{conda_env_vars}{path_setup}"
        f"CODEX_API_BASE={shlex.quote(gateway_url)} "
        f"CODEX_MODEL={shlex.quote(model_name)} "
        f"CODEX_API_KEY={shlex.quote(api_key)} "
        f"CODEX_PROJECT_DIR={shlex.quote(project_dir)}"
    )
    return (
        f"printf %s {shlex.quote(task_b64)} | base64 -d | "
        f"env {env} bash {shlex.quote(tool_script)}"
    )


def parse_agent_result(stdout: str, exit_code: int) -> dict[str, Any]:
    """Normalize Codex ``--json`` JSONL events into a compact result object."""
    if exit_code == -1:
        return {
            "exit_status": "timeout",
            "ok": False,
            "content": "",
            "error": "agent process timed out",
        }

    event_count = 0
    final_content = ""
    errors: list[str] = []
    process_exit_code = exit_code
    saw_turn_completed = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_count += 1
        event_type = event.get("type")
        if event_type == "process.completed":
            value = event.get("exit_code")
            if isinstance(value, int):
                process_exit_code = value
        elif event_type in {"error", "turn.failed"}:
            error = event.get("error")
            errors.append(str(error.get("message") if isinstance(error, dict) else error or event))
        item = event.get("item")
        if isinstance(item, dict):
            if item.get("type") in {"agent_message", "message"}:
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    final_content = text
        if event_type in {"turn.completed", "response.completed"}:
            saw_turn_completed = True
            response = event.get("response")
            if isinstance(response, dict) and isinstance(response.get("output_text"), str):
                final_content = response["output_text"]

    ok = process_exit_code == 0 and not errors and saw_turn_completed
    result: dict[str, Any] = {
        "exit_status": "ok" if ok else "error",
        "ok": ok,
        "content": final_content,
        "event_count": event_count,
    }
    if errors:
        result["error"] = errors[-1]
    elif process_exit_code != 0:
        result["error"] = f"codex exited with code {process_exit_code}"
    elif not saw_turn_completed:
        result["error"] = "codex output did not contain a turn.completed event"
    return result


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Extract the user task using the mini-swe-agent message contract.

    The Codex sidecar accepts one task string.  Like mini-swe-agent, an optional
    leading system message is accepted for framework compatibility but is not
    sent to the sidecar; callers must put task instructions in the user message.
    """
    if not isinstance(messages, list) or not messages or len(messages) > 2:
        raise ValueError("codex accepts at most 2 messages (system?, user)")
    user_parts: list[str] = []
    saw_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"codex message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"}:
            raise ValueError(f"codex only supports initial system/user messages; message {index} has role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"codex message {index} must have non-empty string content")
        if role == "system":
            if saw_user:
                raise ValueError("codex system messages must precede the user message")
        else:
            user_parts.append(content)
            saw_user = True
    if len(user_parts) != 1:
        raise ValueError("codex requires exactly one 'user' message")
    return user_parts[0]


class CodexConfig(AgentConfig):
    """Launch parameters for Codex inside the configured sandbox."""

    name: str = "codex"
    run_timeout: float = Field(default=7200.0, gt=0, description="Maximum wall-clock time for one Codex episode.")
    conda_env_path: str | None = Field(
        default=None,
        description="Task-image path of the Conda environment; unset leaves the launch unactivated.",
    )
    path: str | None = Field(
        default=None,
        description="Optional colon-separated PATH entries; unset derives the task Conda bin paths.",
    )
    tool_script: str = Field(default="/opt/codex/bin/run_agent.sh", description="Sidecar entrypoint.")


@register_agent("codex")
class CodexAgent(Agent):
    """Run Codex ``exec`` non-interactively against the per-session Gateway."""

    config_model = CodexConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: CodexConfig = self.config  # type: ignore[assignment]
        base_url = cfg.model.base_url
        if not base_url:
            raise ValueError("codex: config.model.base_url is not set (the gateway/vLLM policy endpoint)")
        user_prompt = _extract_prompt(messages)
        model_name = cfg.model.model_name
        if not model_name:
            raise ValueError("codex: set config.model.model_name (the model Codex sends)")

        api_key = cfg.model.api_key
        if not api_key or api_key == "EMPTY":
            api_key = uuid.uuid4().hex
        project_dir = workdir or "/testbed"
        task_b64 = base64.b64encode(user_prompt.encode()).decode()
        command = build_agent_command(
            task_b64=task_b64,
            tool_script=cfg.tool_script,
            gateway_url=base_url,
            model_name=model_name,
            api_key=api_key,
            project_dir=project_dir,
            conda_env_path=cfg.conda_env_path,
            path=cfg.path,
        )

        logger.info("codex: launch in %s", project_dir)
        proc = await sandbox.exec_shell(command, timeout=cfg.run_timeout, workdir=project_dir)
        parsed = parse_agent_result(proc.stdout or "", proc.exit_code)
        out_tail = _diagnostic(proc.stdout, api_key)
        err_tail = _diagnostic(proc.stderr, api_key)
        if proc.exit_code != 0 or parsed.get("error") or not parsed.get("content"):
            logger.warning(
                "codex: status=%s stdout_tail=%s stderr_tail=%s",
                parsed.get("exit_status"),
                out_tail,
                err_tail,
            )
        return AgentResult(
            output=parsed,
            transcript=list(messages),
            info={
                **parsed,
                "exit_code": proc.exit_code,
                "error_kind": (
                    None
                    if parsed.get("ok") is True
                    else "timeout"
                    if parsed.get("exit_status") == "timeout"
                    else "agent_failure"
                ),
                "stdout_tail": out_tail,
                "stderr_tail": err_tail,
            },
            finished=parsed.get("ok") is True,
        )
