"""Launch pinned OpenCode in a Task-owned Sandbox through Chat Completions.

The runner supplies the reachable, session-scoped model URL. OpenCode owns its
tools and loop; the Gateway owns protocol translation and trajectory capture.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

OPENCODE_VERSION = "1.19.3"
_PROVIDER = "uni-agent"
_KEY_ENV = "UNI_AGENT_OPENCODE_API_KEY"


class OpenCodeConfig(AgentConfig):
    """Launch settings for the tested OpenCode release, with no runtime install."""

    name: str = "opencode"
    opencode_bin: str = Field(default="opencode", min_length=1)
    run_timeout: float = Field(default=1800, gt=0, allow_inf_nan=False)
    state_root: str = "/tmp/uni-agent-opencode"
    context_limit: int = Field(default=32768, gt=0)
    output_limit: int = Field(default=8192, gt=0)

    @field_validator("state_root")
    @classmethod
    def _absolute_root(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError("opencode state_root must be an absolute non-root path without '..'")
        return str(path)


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "<redacted>")
            value = value.replace(json.dumps(secret)[1:-1], "<redacted>")
        return re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1<redacted>", value)
    if isinstance(value, dict):
        return {
            _redact(str(key), secret): "<redacted>"
            if str(key).lower() in {"api_key", "apikey", "authorization", "token", "password"}
            else _redact(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _diagnostic(text: str, secret: str) -> str:
    # CLI diagnostics sometimes contain JSON with structured credential fields.
    lines = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            value = line
        redacted = _redact(value, secret)
        lines.append(redacted if isinstance(redacted, str) else json.dumps(redacted))
    return "\n".join(lines)[-4000:]


def _prompt(messages: list[dict[str, Any]]) -> str:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("opencode requires exactly one user message")
    content = messages[0].get("content")
    if not isinstance(content, str) or not content.strip() or "\x00" in content:
        raise ValueError("opencode requires non-empty user text without NUL")
    return content


def _runtime_config(cfg: OpenCodeConfig) -> dict[str, Any]:
    model = cfg.model
    endpoint = urlsplit(model.base_url or "")
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or any(char in (model.base_url or "") for char in "{}")
    ):
        raise ValueError("opencode requires an HTTP(S) model.base_url without credentials, query or fragment")
    if not model.model_name or not model.model_name.strip() or any(char in model.model_name for char in "{}\x00"):
        raise ValueError("opencode requires a model.model_name without config substitutions or NUL")
    if not model.api_key:
        raise ValueError("opencode requires a non-empty model.api_key (EMPTY is sufficient for the Gateway)")
    if cfg.output_limit >= cfg.context_limit:
        raise ValueError("opencode output_limit must be smaller than context_limit")
    return {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": [_PROVIDER],
        "provider": {
            _PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Uni-Agent Gateway",
                "options": {
                    "baseURL": model.base_url.rstrip("/"),
                    "apiKey": "{env:" + _KEY_ENV + "}",
                    "timeout": max(1, int(cfg.run_timeout * 1000)),
                },
                "models": {
                    model.model_name: {
                        "name": model.model_name,
                        "limit": {"context": cfg.context_limit, "output": cfg.output_limit},
                    }
                },
            }
        },
        "model": f"{_PROVIDER}/{model.model_name}",
        "small_model": f"{_PROVIDER}/{model.model_name}",
        "agent": {"title": {"disable": True}, "summary": {"disable": True}},
        "compaction": {"auto": False, "prune": False},
        "permission": {
            "*": "allow",
            "task": "deny",
            "question": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "external_directory": "deny",
        },
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
    }


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    session_id = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("type") not in {
            "step_start",
            "step_finish",
            "text",
            "tool_use",
            "reasoning",
            "error",
        }:
            raise ValueError("invalid OpenCode event")
        current_id = event.get("sessionID")
        if not isinstance(current_id, str) or not current_id or (session_id and current_id != session_id):
            raise ValueError("invalid OpenCode session")
        session_id = current_id
        if event["type"] != "error" and not isinstance(event.get("part"), dict):
            raise ValueError("missing OpenCode event part")
        events.append(event)
    if not events:
        raise ValueError("empty OpenCode event stream")
    return events


@register_agent("opencode")
class OpenCodeAgent(Agent):
    """Black-box prototype for OpenCode 1.19.3; never starts or stops the Sandbox."""

    config_model = OpenCodeConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]], workdir: str | None = None) -> AgentResult:
        cfg: OpenCodeConfig = self.config  # type: ignore[assignment]
        prompt = _prompt(messages)
        runtime_config = _runtime_config(cfg)
        workspace = workdir or "/testbed"
        if not PurePosixPath(workspace).is_absolute():
            raise ValueError("opencode requires an absolute workdir")
        root = str(PurePosixPath(cfg.state_root) / uuid.uuid4().hex)
        env = {
            "HOME": f"{root}/home",
            "TMPDIR": root,
            "XDG_CONFIG_HOME": f"{root}/config",
            "XDG_DATA_HOME": f"{root}/data",
            "XDG_STATE_HOME": f"{root}/state",
            "XDG_CACHE_HOME": f"{root}/cache",
            "OPENCODE_CONFIG": "",
            "OPENCODE_CONFIG_DIR": f"{root}/config/opencode",
            "OPENCODE_CONFIG_CONTENT": json.dumps(runtime_config),
            "OPENCODE_DB": f"{root}/opencode.db",
            "OPENCODE_TEST_HOME": f"{root}/home",
            "OPENCODE_PERMISSION": json.dumps(runtime_config["permission"]),
            "OPENCODE_WORKSPACE_ID": "",
            "OPENCODE_EXPERIMENTAL_WORKSPACES": "0",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_AUTOCOMPACT": "1",
            "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
            _KEY_ENV: cfg.model.api_key,
        }
        info: dict[str, Any] = {"exit_code": None, "error_kind": None, "cli_version": OPENCODE_VERSION}
        output: dict[str, Any] = {}
        finished = False
        try:
            prepared = await sandbox.exec(["mkdir", "-p", "-m", "700", root], timeout=30)
            if prepared.exit_code != 0:
                info["error_kind"] = "startup_failure"
            else:
                # Preflight without the credential: wrong binaries fail before the prompt is submitted.
                check_env = {key: value for key, value in env.items() if key != _KEY_ENV}
                version = await sandbox.exec([cfg.opencode_bin, "--version"], env=check_env, timeout=30)
                if version.exit_code != 0:
                    info.update(
                        exit_code=version.exit_code,
                        error_kind="timeout" if version.exit_code == -1 else "cli_unavailable",
                    )
                elif version.stdout.strip() != OPENCODE_VERSION:
                    info["error_kind"] = "unsupported_version"
                else:
                    proc = await sandbox.exec(
                        [
                            cfg.opencode_bin,
                            "run",
                            "--pure",
                            "--format",
                            "json",
                            "--model",
                            f"{_PROVIDER}/{cfg.model.model_name}",
                            "--agent",
                            "build",
                            # OpenCode prefers inherited PWD over cwd without --dir.
                            "--dir",
                            workspace,
                            "--title",
                            "Uni-Agent rollout",
                            "--",
                            prompt,
                        ],
                        env=env,
                        timeout=cfg.run_timeout,
                        workdir=workspace,
                    )
                    info.update(
                        exit_code=proc.exit_code,
                        stdout_tail=_diagnostic(proc.stdout or "", cfg.model.api_key),
                        stderr_tail=_diagnostic(proc.stderr or "", cfg.model.api_key),
                    )
                    try:
                        events = _parse_events(proc.stdout or "")
                    except (ValueError, TypeError):
                        events = []
                        info["error_kind"] = "protocol_error"
                    output = {
                        "events": events,
                        "text": "".join(
                            event["part"].get("text", "")
                            for event in events
                            if event["type"] == "text" and isinstance(event["part"].get("text"), str)
                        ),
                    }
                    steps = [event["part"] for event in events if event["type"] == "step_finish"]
                    if proc.exit_code == -1:
                        info["error_kind"] = "timeout"
                    elif proc.exit_code != 0 or any(event["type"] == "error" for event in events):
                        info["error_kind"] = "agent_failure"
                    elif events and (
                        not steps or steps[-1].get("reason") != "stop" or events[-1]["type"] != "step_finish"
                    ):
                        info["error_kind"] = "incomplete"
                    finished = bool(events) and info["error_kind"] is None
        except (TimeoutError, OSError) as exc:
            info["error_kind"] = "timeout" if isinstance(exc, TimeoutError) else "startup_failure"
            # Provider exception messages may contain credentials or entire launch environments.
            info["error_type"] = type(exc).__name__
        finally:
            # This UUID directory belongs only to this invocation. CancelledError propagates
            # after cleanup; process termination remains the Sandbox provider's contract.
            try:
                cleanup = await sandbox.exec(["rm", "-rf", "--", root], timeout=30)
                if cleanup.exit_code != 0:
                    info["cleanup_error"] = "nonzero_exit"
            except Exception as exc:
                info["cleanup_error"] = type(exc).__name__
            if info.get("cleanup_error"):
                finished = False
                info["error_kind"] = info["error_kind"] or "cleanup_failure"
        transcript = list(messages)
        if output.get("text"):
            transcript.append({"role": "assistant", "content": output["text"]})
        return AgentResult(
            output=_redact(output, cfg.model.api_key),
            transcript=_redact(transcript, cfg.model.api_key),
            info=_redact(info, cfg.model.api_key),
            finished=finished,
        )
