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
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

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
    codex_bin: str,
    gateway_url: str,
    model_name: str,
    api_key: str,
    project_dir: str = "/testbed",
    model_catalog_path: str | None = None,
    codex_home: str = "/tmp/codex-home",
) -> str:
    """Pipe a task into the native Codex CLI with per-episode configuration.

    The prompt is base64-encoded to avoid argv length limits and shell quoting
    hazards. Provider settings use Codex's native configuration overrides.
    """
    env = (
        f"CODEX_HOME={shlex.quote(codex_home)} "
        f"OPENAI_API_KEY={shlex.quote(api_key)} "
        'CODEX_MANAGED_PACKAGE_ROOT="${CODEX_MANAGED_PACKAGE_ROOT:-/opt/codex}" '
    )
    gateway_host = urlsplit(gateway_url).hostname
    if gateway_host:
        for key in ("NO_PROXY", "no_proxy"):
            env += f'{key}="${{{key}:+${key},}}"{shlex.quote(gateway_host)} '

    argv = [
        codex_bin,
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        project_dir,
        "--model",
        model_name,
    ]
    overrides = {
        "model_provider": "gateway",
        "model_providers.gateway.name": "Uni-Agent Gateway",
        "model_providers.gateway.base_url": gateway_url,
        "model_providers.gateway.wire_api": "responses",
        "model_providers.gateway.requires_openai_auth": True,
        "model_providers.gateway.env_key": "OPENAI_API_KEY",
        "disable_response_storage": True,
        "check_for_update_on_startup": False,
    }
    if model_catalog_path is not None:
        overrides["model_catalog_json"] = model_catalog_path
    for key, value in overrides.items():
        argv.extend(["-c", f"{key}={json.dumps(value)}"])
    return (
        f"mkdir -p {shlex.quote(codex_home)} && "
        f"printf %s {shlex.quote(task_b64)} | base64 -d | env {env}{shlex.join(argv)}"
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
    model_catalog_path: str | None = Field(
        default=None,
        description="Host-side Codex model catalog JSON; copied into the sandbox when set.",
    )
    codex_bin: str = Field(
        default="/opt/codex/vendor/x86_64-unknown-linux-musl/bin/codex",
        description="Full in-sandbox path of the native Codex executable.",
    )


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
        catalog_path = None
        if cfg.model_catalog_path is not None:
            catalog_path = f"/tmp/codex-model-catalog-{uuid.uuid4().hex}.json"
            await sandbox.write_file(catalog_path, Path(cfg.model_catalog_path).read_text(encoding="utf-8"))
        command = build_agent_command(
            task_b64=task_b64,
            codex_bin=cfg.codex_bin,
            gateway_url=base_url,
            model_name=model_name,
            api_key=api_key,
            project_dir=project_dir,
            model_catalog_path=catalog_path,
            codex_home=f"/tmp/codex-home-{uuid.uuid4().hex}",
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
