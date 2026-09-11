"""OpenClaw black-box agent adapter.

The OpenClaw CLI is supplied by a prebuilt sidecar mounted at ``/opt/openclaw``.
This host-side adapter creates a private per-episode config/state directory,
passes the prompt by file (OpenClaw's ``agent --local`` does not read it from
stdin), then converts the CLI JSON envelope into ``AgentResult``.
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import shlex
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_MAX_DIAGNOSTIC_CHARS = 4000


def _shell_quote_path(path: str) -> str:
    return shlex.quote(path)


def parse_openclaw_result(stdout: str) -> dict[str, Any] | None:
    """Return the last JSON object emitted by the OpenClaw CLI, if present."""
    text = stdout.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    end = 0
    for offset, char in enumerate(text):
        if offset < end:
            continue
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[offset:])
            end = offset + consumed
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates[-1] if candidates else None


def build_openclaw_config(*, base_url: str, api_key: str, model_name: str, workspace: str, timeout_seconds: int) -> dict[str, Any]:
    """Build an isolated OpenClaw config for one episode.

    The recipe intentionally allows only ``exec``. Network policy remains an
    outer sandbox responsibility; OpenClaw's exec permission is not a network
    sandbox.
    """
    return {
        "models": {
            "mode": "replace",
            "providers": {
                "vllm": {
                    "baseUrl": base_url,
                    "apiKey": api_key,
                    "api": "openai-completions",
                    "agentRuntime": {"id": "openclaw"},
                    "models": [
                        {
                            "id": model_name,
                            "name": model_name,
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 32768,
                            "maxTokens": 8192,
                        }
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": f"vllm/{model_name}", "fallbacks": []},
                "models": {f"vllm/{model_name}": {"agentRuntime": {"id": "openclaw"}}},
                "workspace": workspace,
                "cwd": workspace,
                "skipBootstrap": True,
                "skills": [],
                "embeddedAgent": {"projectSettingsPolicy": "ignore", "executionContract": "default"},
                "contextPruning": {"mode": "off"},
                "compaction": {
                    "enabled": False,
                    "midTurnPrecheck": {"enabled": False},
                    "memoryFlush": {"enabled": False},
                },
                "sandbox": {"mode": "off"},
                "timeoutSeconds": timeout_seconds,
                "maxConcurrent": 1,
            }
        },
        "tools": {
            "allow": ["exec"],
            "codeMode": False,
            "toolSearch": False,
            "swarm": False,
            "agentToAgent": {"enabled": False},
            "fs": {"workspaceOnly": True},
            "exec": {
                "host": "gateway",
                "security": "full",
                "ask": "off",
                "applyPatch": {"enabled": False},
                "timeoutSeconds": min(timeout_seconds, 600),
            },
        },
        "plugins": {"allow": ["vllm"]},
    }


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    if len(messages) > 2:
        raise ValueError(f"openclaw accepts at most 2 messages (system?, user), got {len(messages)}")
    prompt = next((message.get("content") for message in messages if message.get("role") == "user"), None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("openclaw requires a non-empty user prompt")
    return prompt


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        if secret and secret != "EMPTY":
            value = value.replace(secret, "<redacted>")
        return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1<redacted>", value)
    if isinstance(value, dict):
        return {
            key: "<redacted>" if str(key).lower() in {"api_key", "apikey", "authorization", "token", "password"}
            else _redact(item, secret) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _diagnostic(value: str | None, secret: str = "") -> str:
    return _redact(value or "", secret).strip()[-_MAX_DIAGNOSTIC_CHARS:]


class OpenClawConfig(AgentConfig):
    """Black-box launch parameters bound to the OpenClaw tool-image layout."""

    name: str = "openclaw"
    run_timeout: float = Field(default=1800.0, gt=0, description="Hard wall-clock cap for the OpenClaw CLI.")
    cli_timeout_seconds: int = Field(default=1700, gt=0, description="OpenClaw --timeout deadline; must be below run_timeout.")
    agent_id: str = Field(default="main", min_length=1)
    tool_command: str = Field(default="/opt/openclaw/bin/openclaw")
    artifact_dir: str | None = Field(default=None, description="Host artifact directory outside the checkout.")
    state_root: str = Field(default="/tmp/uni-agent-openclaw")
    max_prompt_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    strict_single_trajectory: bool = Field(
        default=True,
        description="Reject an output showing a model fallback. Compaction/recovery is additionally audited by task logs.",
    )
    agent_max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Reserved rollout policy limit. OpenClaw 2026.9.2 has no public agent --local max-turns control.",
    )


@register_agent("openclaw")
class OpenClawAgent(Agent):
    """Run the pinned OpenClaw CLI in a sandbox through an OpenAI-compatible endpoint."""

    config_model = OpenClawConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]], workdir: str | None = None) -> AgentResult:
        cfg: OpenClawConfig = self.config  # type: ignore[assignment]
        model = cfg.model
        if not model.base_url or not model.model_name:
            raise ValueError("openclaw requires config.model.base_url and config.model.model_name")
        if cfg.cli_timeout_seconds >= cfg.run_timeout:
            raise ValueError("openclaw cli_timeout_seconds must be lower than run_timeout")
        if cfg.agent_max_turns is not None:
            raise ValueError("OpenClaw 2026.9.2 cannot enforce agent_max_turns; refusing an unsupported limit")
        prompt = _extract_user_prompt(messages)
        if len(prompt.encode("utf-8")) > cfg.max_prompt_bytes:
            raise ValueError(f"openclaw prompt exceeds max_prompt_bytes={cfg.max_prompt_bytes}")

        workspace = workdir or "/workspace"
        episode_id = uuid.uuid4().hex
        root = str(PurePosixPath(cfg.state_root) / episode_id)
        config_path = f"{root}/openclaw.json"
        prompt_path = f"{root}/task.txt"
        state_dir = f"{root}/state"
        home_dir = f"{root}/home"
        config = build_openclaw_config(
            base_url=model.base_url,
            api_key=model.api_key,
            model_name=model.model_name,
            workspace=workspace,
            timeout_seconds=cfg.cli_timeout_seconds,
        )
        prepared = await sandbox.exec_shell(f"umask 077; mkdir -p {_shell_quote_path(root)}", timeout=30)
        if prepared.exit_code != 0:
            return AgentResult(finished=False, info={"error_kind": "startup_failure", "session_id": episode_id})
        failure = None
        cleanup_error = None
        artifact_info = {}
        try:
            await sandbox.write_file(config_path, json.dumps(config, separators=(",", ":")))
            launch_prompt = (
                prompt + "\n\nRecipe completion protocol: Use only tools listed in the provided tool schema "
                "(exec). NO_REPLY is a text marker, not a tool; never call it. "
                "After finishing and checking the task, return a brief plain-text final answer "
                "without a tool call. Do not send messages or delegate to another agent."
            )
            await sandbox.write_file(prompt_path, launch_prompt)
            argv = [
                cfg.tool_command,
                "agent",
                "--local",
                "--agent",
                cfg.agent_id,
                "--session-id",
                episode_id,
                "--message-file",
                prompt_path,
                "--thinking",
                "off",
                "--timeout",
                str(cfg.cli_timeout_seconds),
                "--json",
            ]
            env = {
                "HOME": home_dir,
                "PATH": str(PurePosixPath(cfg.tool_command).parent) + ":/usr/local/bin:/usr/bin:/bin",
                "OPENCLAW_CONFIG_PATH": config_path,
                "OPENCLAW_STATE_DIR": state_dir,
                "OPENCLAW_NO_RESPAWN": "1",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "http_proxy": "",
                "https_proxy": "",
            }
            proc = await sandbox.exec(argv, env=env, timeout=cfg.run_timeout, workdir=workspace)
            audit_path = f"{root}/audit_trajectory.py"
            await sandbox.write_file(audit_path, Path(__file__).with_name("trajectory.py").read_text(encoding="utf-8"))
            audit_argv = ["python3", audit_path, state_dir, episode_id, model.model_name]
            if cfg.artifact_dir:
                audit_argv += ["--export", f"{root}/trajectory.json"]
            audit_proc = await sandbox.exec(audit_argv, timeout=60, workdir=workspace)
            audit_result = parse_openclaw_result(audit_proc.stdout or "")
            if audit_proc.exit_code != 0 or not isinstance(audit_result, dict):
                audit_result = {"verified": False, "errors": ["audit_process_failed"]}
        except (TimeoutError, OSError) as exc:
            failure = {
                "error_kind": "timeout" if isinstance(exc, TimeoutError) else "startup_failure",
                "error_type": type(exc).__name__, "session_id": episode_id, "state_dir": state_dir,
            }
        finally:
            # Remove credentials only; preserve SQLite state for task-level auditing.
            try:
                cleanup = await sandbox.exec_shell(f"rm -f {_shell_quote_path(config_path)}", timeout=30)
                if cleanup.exit_code != 0:
                    cleanup_error = "nonzero_exit"
            except Exception as exc:
                cleanup_error = type(exc).__name__

        if cfg.artifact_dir and failure is None:
            try:
                raw = await sandbox.read_file(f"{root}/trajectory.json")
                exported = _redact(json.loads(raw), model.api_key)
                directory = Path(cfg.artifact_dir).resolve() / episode_id
                directory.mkdir(mode=0o700, parents=True, exist_ok=False)
                saved = json.dumps(exported, ensure_ascii=False, sort_keys=True).encode()
                path = directory / "trajectory.json"
                with path.open("xb") as stream:
                    stream.write(saved)
                path.chmod(0o600)
                artifact_info = {"trajectory_path": str(path), "trajectory_sha256": hashlib.sha256(saved).hexdigest()}
            except Exception as exc:
                failure = {"error_kind": "artifact_failure", "error_type": type(exc).__name__,
                           "session_id": episode_id, "state_dir": state_dir}
        if failure is not None:
            failure["cleanup_error"] = cleanup_error
            return AgentResult(finished=False, info=failure)

        payload = parse_openclaw_result(proc.stdout or "")
        info: dict[str, Any] = {
            "exit_code": proc.exit_code,
            "stdout_tail": _diagnostic(proc.stdout, model.api_key),
            "stderr_tail": _diagnostic(proc.stderr, model.api_key),
            "cleanup_error": cleanup_error,
            **artifact_info,
            "session_id": episode_id,
            "state_dir": state_dir,
            "agent_max_turns": cfg.agent_max_turns,
        }
        if payload is not None:
            info["openclaw"] = payload
        meta = payload.get("meta") if isinstance(payload, dict) else None
        trace = meta.get("executionTrace") if isinstance(meta, dict) else None
        fallback_used = trace.get("fallbackUsed") if isinstance(trace, dict) else None
        if cfg.strict_single_trajectory and fallback_used is True:
            info["trajectory_rejected"] = "model_fallback"
        info["trajectory_audit"] = audit_result
        if cfg.strict_single_trajectory and audit_result.get("verified") is not True:
            info.setdefault("trajectory_rejected", "sqlite_audit_failed")
        finished = (
            proc.exit_code == 0
            and isinstance(meta, dict)
            and meta.get("aborted") is False
            and isinstance(payload.get("payloads"), list)
            and info.get("trajectory_rejected") is None
            and cleanup_error is None
        )
        if isinstance(meta, dict) and meta.get("timeoutPhase"):
            info["error_kind"] = "timeout"
            info["timeout_phase"] = meta["timeoutPhase"]
            finished = False
        elif proc.exit_code != 0:
            info["error_kind"] = "agent_failure"
        elif not isinstance(meta, dict) or not isinstance(payload.get("payloads"), list):
            info["error_kind"] = "protocol_error"
        elif meta.get("aborted") is not False:
            info["error_kind"] = "agent_failure"
        elif info.get("trajectory_rejected"):
            info["error_kind"] = "trajectory_rejected"
        if cleanup_error is not None:
            info.setdefault("error_kind", "cleanup_failure")
        return AgentResult(output=_redact(payload or {}, model.api_key), transcript=list(messages),
                           info=_redact(info, model.api_key), finished=finished)
