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
import shlex
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


def build_agent_command(
    *,
    task_b64: str,
    tool_script: str,
    gateway_url: str,
    model_name: str,
    api_key: str,
    project_dir: str = "/testbed",
    conda_env: str = "testbed",
) -> str:
    """Build the shell command that pipes a task into the Codex sidecar.

    The prompt is base64-encoded to avoid argv length limits and shell quoting
    hazards. Gateway and process settings are passed as environment variables
    consumed by the sidecar entrypoint.
    """
    conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
    env = (
        f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)} "
        f"CONDA_PREFIX={shlex.quote(conda_prefix)} "
        f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH "
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
            response = event.get("response")
            if isinstance(response, dict) and isinstance(response.get("output_text"), str):
                final_content = response["output_text"]

    ok = process_exit_code == 0 and not errors
    result: dict[str, Any] = {
        "exit_status": "ok" if ok else "error",
        "ok": ok,
        "content": final_content,
        "event_count": event_count,
    }
    if errors:
        result["error"] = errors[-1]
    elif not ok:
        result["error"] = f"codex exited with code {process_exit_code}"
    return result


class CodexConfig(AgentConfig):
    """Launch parameters for Codex inside the configured sandbox."""

    name: str = "codex"
    run_timeout: float = Field(default=7200.0, description="Maximum wall-clock time for one Codex episode.")
    conda_env: str = Field(default="testbed", description="Conda environment used by repository tools.")
    tool_script: str = Field(description="Sidecar entrypoint, normally /opt/codex/bin/run_agent.sh.")


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
        user_messages = [message.get("content") for message in messages if message.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError("codex requires exactly one 'user' message")
        user_prompt = user_messages[0]
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("codex requires a non-empty user prompt")
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
            conda_env=cfg.conda_env,
        )

        logger.info("codex: launch in %s", project_dir)
        proc = await sandbox.exec_shell(command, timeout=cfg.run_timeout, workdir=project_dir)
        parsed = parse_agent_result(proc.stdout or "", proc.exit_code)
        out_tail = (proc.stdout or "").strip()[-4000:]
        err_tail = (proc.stderr or "").strip()[-2000:]
        if proc.exit_code != 0 or parsed.get("error") or not parsed.get("content"):
            logger.warning("codex: result=%s stdout_tail=%s stderr_tail=%s", parsed, out_tail, err_tail)
        return AgentResult(
            output=parsed,
            transcript=list(messages),
            info={
                **parsed,
                "exit_code": proc.exit_code,
                "stdout_tail": out_tail,
                "stderr_tail": err_tail,
            },
            finished=parsed.get("ok") is True,
        )
