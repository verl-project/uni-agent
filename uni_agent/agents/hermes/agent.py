"""Hermes Agent black-box adapter.

The adapter deliberately keeps Hermes out of the Uni-Agent driver environment.  A
small, JSON-only request is written into the task sandbox and the real Hermes
``AIAgent`` is launched from the mounted ``/opt/hermes`` tool image.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RESULT_MARKER = "HERMES_RECIPE_RESULT "
MAX_RESULT_BYTES = 8 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class HermesConfig(AgentConfig):
    """Launch and budget settings for the sidecar Hermes runtime."""

    name: str = "hermes"
    max_iterations: int = Field(default=100, ge=1)
    run_budget_seconds: float = Field(default=6600.0, gt=0)
    run_timeout: float = Field(default=7200.0, gt=0)
    terminal_timeout: int = Field(default=600, ge=1)
    environment_prefix: str | None = Field(default=None, description="Absolute task environment prefix, if needed.")
    tool_python: str = Field(default="/opt/hermes/bin/python", min_length=1)
    runner_script: str = Field(default="/opt/hermes/bin/run_hermes.py", min_length=1)
    approval_mode: Literal["off", "default"] = Field(
        default="off",
        description="Sandbox-scoped approval policy. off is the explicit unattended recipe policy.",
    )


def _safe_run_id(value: str) -> str:
    cleaned = _SAFE_RUN_ID.sub("-", value).strip("-.")
    return cleaned[:120] or f"run-{uuid.uuid4().hex}"


def _message_text(message: object, index: int) -> str:
    if not isinstance(message, dict):
        raise ValueError(f"hermes message {index} must be an object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"hermes message {index} must have non-empty string content")
    return content


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate the system/user prefix accepted by the Hermes runner."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("hermes requires a non-empty messages list")
    saw_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"hermes message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user"}:
            raise ValueError(f"hermes only supports initial system/user messages; message {index} has role {role!r}")
        if role == "system" and saw_user:
            raise ValueError("hermes system messages must precede user messages")
        _message_text(message, index)
        saw_user = saw_user or role == "user"
    if not saw_user:
        raise ValueError("hermes requires at least one user message")


def build_runner_command(
    *,
    input_path: str,
    result_path: str,
    messages_path: str,
    log_path: str,
    hermes_home: str,
    environment_prefix: str | None,
    tool_python: str,
    runner_script: str,
    terminal_timeout: int,
    approval_mode: str,
) -> str:
    """Build the in-sandbox launch command.

    Only fixed paths and scalar controls appear in the command.  Prompt text and
    credentials are transferred through ``input_path`` instead of argv.
    """

    q = shlex.quote
    if environment_prefix is not None and not environment_prefix.startswith("/"):
        raise ValueError("environment_prefix must be an absolute sandbox path")
    path_setup = f'export PATH={q(environment_prefix + "/bin")}:"${{PATH:-}}"; ' if environment_prefix else ""
    yolo = "1" if approval_mode == "off" else "0"
    assignments = " ".join(
        [
            f"HERMES_HOME={q(hermes_home)}",
            "HERMES_INTERACTIVE=0",
            f"HERMES_YOLO_MODE={yolo}",
            "TERMINAL_ENV=local",
            f"TERMINAL_TIMEOUT={q(str(terminal_timeout))}",
            "PYTHONUNBUFFERED=1",
        ]
    )
    return (
        f"umask 077; mkdir -p {q(hermes_home)}; "
        f"{path_setup}"
        f"env {assignments} {q(tool_python)} {q(runner_script)} "
        f"--result-path {q(result_path)} --messages-path {q(messages_path)} --log-path {q(log_path)} "
        f"< {q(input_path)}"
    )


def _parse_json_bytes(data: bytes) -> dict[str, Any] | None:
    if not data or len(data) > MAX_RESULT_BYTES:
        return None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_result_stdout(stdout: str) -> dict[str, Any] | None:
    """Read the runner's marked result without trusting arbitrary stdout JSON."""

    if not stdout or len(stdout.encode("utf-8", errors="replace")) > MAX_RESULT_BYTES:
        return None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                value = json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def validate_result(value: object, *, run_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Hermes runner result must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Hermes runner schema_version must be {SCHEMA_VERSION}")
    if value.get("run_id") != run_id:
        raise ValueError(f"Hermes runner run_id mismatch: expected {run_id!r}")
    if value.get("status") not in {"completed", "budget_exhausted", "error", "timeout", "cancelled"}:
        raise ValueError("Hermes runner result status must be a string")
    if type(value.get("finished")) is not bool:
        raise ValueError("Hermes runner result finished must be a bool")
    if value["finished"] != (value["status"] == "completed"):
        raise ValueError("Hermes runner finished must agree with status")
    return value


def _model_payload(cfg: HermesConfig) -> dict[str, Any]:
    if not cfg.model.base_url:
        raise ValueError("hermes: config.model.base_url is not set (the gateway session endpoint)")
    if not cfg.model.model_name:
        raise ValueError("hermes: config.model.model_name is not set (the served policy model)")
    if not cfg.model.api_key:
        raise ValueError("hermes: config.model.api_key must be non-empty")
    return {
        "endpoint": cfg.model.base_url,
        "name": cfg.model.model_name,
        "api_key": cfg.model.api_key,
    }


@register_agent("hermes")
class HermesAgent(Agent):
    """Launch the real Hermes loop inside the already-running task sandbox."""

    config_model = HermesConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: HermesConfig = self.config  # type: ignore[assignment]
        validate_messages(messages)
        model = _model_payload(cfg)
        run_id = _safe_run_id(f"{uuid.uuid4().hex}-{getattr(self.config, 'name', 'hermes')}")
        remote_root = f"/tmp/hermes-{run_id}"
        input_path = f"{remote_root}/input.json"
        result_path = f"{remote_root}/result.json"
        messages_path = f"{remote_root}/messages.json"
        log_path = f"{remote_root}/runner.log"
        hermes_home = f"{remote_root}/home"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "messages": messages,
            "workdir": workdir or "/testbed",
            "model": model,
            "sampling": cfg.model.sampling_params(),
            "limits": {
                "max_iterations": cfg.max_iterations,
                "max_tokens": cfg.model.max_tokens_per_turn,
                "run_budget_seconds": cfg.run_budget_seconds,
                "terminal_timeout": cfg.terminal_timeout,
            },
            "approval_mode": cfg.approval_mode,
        }
        await sandbox.write_file(input_path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        command = build_runner_command(
            input_path=input_path,
            result_path=result_path,
            messages_path=messages_path,
            log_path=log_path,
            hermes_home=hermes_home,
            environment_prefix=cfg.environment_prefix,
            tool_python=cfg.tool_python,
            runner_script=cfg.runner_script,
            terminal_timeout=cfg.terminal_timeout,
            approval_mode=cfg.approval_mode,
        )
        process = await sandbox.exec_shell(command, timeout=cfg.run_timeout, workdir=workdir or "/testbed")

        envelope: dict[str, Any] | None = None
        if process.exit_code != -1:
            try:
                envelope = _parse_json_bytes(await sandbox.read_file(result_path))
            except Exception:
                logger.debug("hermes: result file unavailable", exc_info=True)
            if envelope is None:
                envelope = parse_result_stdout(process.stdout or "")

        if process.exit_code == -1:
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "status": "timeout",
                "finished": False,
                "final_response": None,
                "error": (process.stderr or "sandbox command timed out")[-4000:],
            }
        elif envelope is None:
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "status": "error",
                "finished": False,
                "final_response": None,
                "error": "missing or malformed Hermes runner result",
            }
        elif process.exit_code != 0:
            envelope = {
                **envelope,
                "status": "error",
                "finished": False,
                "error": f"Hermes runner exited with code {process.exit_code}",
            }
        try:
            envelope = validate_result(envelope, run_id=run_id)
        except ValueError as exc:
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "status": "error",
                "finished": False,
                "final_response": None,
                "error": str(exc),
            }

        info = {
            "run_id": run_id,
            "status": envelope.get("status"),
            "process_exit_code": process.exit_code,
            "stop_reason": envelope.get("stop_reason"),
        }
        return AgentResult(
            output=envelope,
            transcript=list(messages),
            info=info,
            finished=envelope["finished"],
        )
