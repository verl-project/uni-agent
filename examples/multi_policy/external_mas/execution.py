"""Generic command-driven external MAS runner.

Each invocation receives a rollout-scoped Gateway URL, injects it into a copy
of the MAS configuration, and launches the configured command inside the
rollout's sandbox.

The sandbox owns execution: the Task builds it from its ``sandbox`` block, so
``provider: local`` runs the command on the host while ``docker``/``modal``/...
run it in an isolated or remote environment. The rendered config is written
into the sandbox filesystem.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_COMMAND_PLACEHOLDERS = ("config_path", "prompt", "session_group_id")
_UNSUPPORTED_PLACEHOLDERS = ("gateway_url",)
_COMMAND_PLACEHOLDER_RE = re.compile(
    r"(?<![$\{])\{(?P<key>" + "|".join(re.escape(name) for name in _COMMAND_PLACEHOLDERS) + r")\}"
)
_UNSUPPORTED_PLACEHOLDER_RE = re.compile(
    r"(?<![$\{])\{(?P<key>" + "|".join(re.escape(name) for name in _UNSUPPORTED_PLACEHOLDERS) + r")\}"
)

_MISSING = object()

#: Directory inside the sandbox that receives the rendered per-rollout config.
_DEFAULT_SANDBOX_CONFIG_DIR = "/tmp"

#: Exit code the base sandbox contract returns when a command times out.
_SANDBOX_TIMEOUT_EXIT_CODE = -1


class ExternalMASError(RuntimeError):
    """The external MAS command failed (non-zero exit)."""

    def __init__(self, message: str, *, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class ExternalMASTimeout(TimeoutError):
    """The external MAS command exceeded its configured timeout."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class ExternalMASRequest:
    """Framework-independent input for one external MAS application run."""

    prompt: str
    role_urls: Mapping[str, str]
    session_group_id: str


def extract_user_task(raw_prompt: Any) -> str:
    """Extract the user task from the prompt shapes accepted by the framework."""
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, Mapping):
        content = raw_prompt.get("content")
        return str(content if content is not None else raw_prompt)
    if isinstance(raw_prompt, Sequence) and not isinstance(raw_prompt, str | bytes):
        for message in reversed(raw_prompt):
            if isinstance(message, Mapping) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(raw_prompt)


def _set_path(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = [key for key in dotted_path.split(".") if key]
    if not keys:
        raise ValueError("injection paths must not be empty")
    current = root
    for key in keys[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot inject {dotted_path!r}: {key!r} is not a mapping")
        current = child
    current[keys[-1]] = value


def _apply_overrides(
    template: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a deep copy of ``template`` with dotted-path overrides applied."""
    resolved = copy.deepcopy(dict(template))
    if not overrides:
        return resolved
    if not isinstance(overrides, Mapping):
        raise ValueError("config.overrides must be a mapping")
    for dotted_path, value in overrides.items():
        _set_path(resolved, str(dotted_path), value)
    return resolved


def inject_gateway_config(
    template: Mapping[str, Any],
    *,
    role_urls: Mapping[str, str],
    injection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copied MAS config with one Gateway URL and model name per role.

    Each role owns its own Gateway session URL, so ``role_urls[role]`` is
    written into that role's config. The global base URL falls back to the
    first role's URL for MAS configs that share one endpoint at the root.
    """
    if not role_urls:
        raise ValueError("role_urls must not be empty")
    resolved = copy.deepcopy(dict(template))
    injection = dict(injection or {})
    default_gateway_url = next(iter(role_urls.values()))
    _set_path(
        resolved,
        str(injection.get("global_base_url_path", "llm.base_url")),
        default_gateway_url,
    )

    agents_path = str(injection.get("agents_path", "agents"))
    agents: dict[str, Any] = {}
    _set_path(resolved, agents_path, agents)
    original_agents = template
    for component in agents_path.split("."):
        if not isinstance(original_agents, Mapping):
            original_agents = {}
            break
        original_agents = original_agents.get(component, {})
    if isinstance(original_agents, Mapping):
        agents.update(copy.deepcopy(dict(original_agents)))

    inject_agent_base_url = bool(injection.get("inject_agent_base_url", True))
    agent_base_url_path = str(injection.get("agent_base_url_path", "llm.base_url"))
    agent_model_path = str(injection.get("agent_model_path", "model"))
    for role, gateway_url in role_urls.items():
        role_config = agents.setdefault(str(role), {})
        if not isinstance(role_config, dict):
            raise ValueError(f"MAS config for role {role!r} must be a mapping")
        if inject_agent_base_url:
            _set_path(role_config, agent_base_url_path, gateway_url)
        _set_path(role_config, agent_model_path, str(role))
    return resolved


def _format_command_value(
    value: str,
    context: Mapping[str, str],
    *,
    shell_quote: bool = False,
) -> str:
    """Substitute known placeholders and leave all other braces untouched."""
    unsupported = _UNSUPPORTED_PLACEHOLDER_RE.search(value)
    if unsupported is not None:
        placeholder = unsupported.group("key")
        raise ValueError(
            f"Unsupported command placeholder {{{placeholder}}}; use the role Gateway URLs in the MAS config."
        )

    def replace(match: re.Match[str]) -> str:
        replacement = str(context[match.group("key")])
        return shlex.quote(replacement) if shell_quote else replacement

    return _COMMAND_PLACEHOLDER_RE.sub(replace, value)


def _format_values(values: Mapping[str, Any], context: Mapping[str, str]) -> dict[str, str]:
    return {str(key): _format_command_value(str(value), context) for key, value in values.items()}


def _build_command(command_config: Mapping[str, Any], context: Mapping[str, str]) -> tuple[str | list[str], bool]:
    shell = bool(command_config.get("shell", False))
    argv = command_config.get("argv")
    template = command_config.get("template")
    if argv is not None:
        if shell or not isinstance(argv, Sequence) or isinstance(argv, str | bytes):
            raise ValueError("command.argv must be a sequence and cannot be combined with shell=true")
        return [_format_command_value(str(part), context) for part in argv], False
    if not isinstance(template, str) or not template:
        raise ValueError("command.argv or command.template is required")
    if shell:
        return _format_command_value(template, context, shell_quote=True), True
    parts = shlex.split(template, posix=True)
    return [_format_command_value(part, context) for part in parts], False


def _parse_stdout_json(stdout: str) -> dict[str, Any]:
    """Extract the last JSON object from stdout, including pretty-printed output."""
    text = stdout.strip()
    if not text:
        raise ValueError("External MAS stdout is empty")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    consumed_until = 0
    for offset, char in enumerate(text):
        # Skip objects nested inside an already decoded result. This keeps the
        # parser linear for large trace payloads and JSON-like agent logs.
        if offset < consumed_until or char != "{":
            continue
        try:
            value, consumed_until = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)

    if candidates:
        return candidates[-1]
    raise ValueError("External MAS stdout did not contain a JSON object")


def _get_nested_value(value: Any, path: str) -> Any:
    """Read a dotted path from a nested mapping."""
    current = value
    for component in str(path).split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _first_present(
    value: Mapping[str, Any],
    paths: Sequence[str],
    *,
    default: Any = _MISSING,
) -> Any:
    for path in paths:
        resolved = _get_nested_value(value, path)
        if resolved is not None:
            return resolved
    if default is not _MISSING:
        return default
    raise ValueError(f"None of the configured result fields exist: {list(paths)}")


def _result_field_paths(config: Any) -> list[str]:
    if isinstance(config, Mapping):
        fields = config.get("fields", [])
    else:
        fields = config
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, Sequence) or isinstance(fields, str | bytes):
        raise ValueError("result.field_mapping.final_result.fields must be a sequence")
    paths = [str(field) for field in fields if str(field)]
    if not paths:
        raise ValueError("result.field_mapping.final_result.fields is required")
    return paths


def _normalize_result(
    raw_result: Mapping[str, Any],
    result_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map an external result into the runner's unified result contract."""
    mapping = result_config.get("field_mapping")
    if not mapping:
        return dict(raw_result)
    if not isinstance(mapping, Mapping):
        raise ValueError("result.field_mapping must be a mapping")

    try:
        final_result = _first_present(
            raw_result,
            _result_field_paths(mapping.get("final_result")),
        )
    except ValueError as exc:
        raise ValueError(f"Configured final result field is missing: {exc}") from exc
    if isinstance(final_result, dict | list | tuple):
        raise ValueError("Configured final result field must contain a scalar value")

    reward_info: dict[str, Any] = {"final_result": final_result}
    reward_mapping = mapping.get("reward_info", {})
    if not isinstance(reward_mapping, Mapping):
        raise ValueError("result.field_mapping.reward_info must be a mapping")
    for output_key, input_path in reward_mapping.items():
        resolved = _get_nested_value(raw_result, str(input_path))
        if resolved is not None:
            reward_info[str(output_key)] = resolved

    return {
        "final_result": str(final_result),
        "reward_info": reward_info,
    }


def _parse_result(process_result: Any, result_config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(result_config.get("mode", "stdout_json"))
    if mode == "none":
        return {}
    stdout = str(getattr(process_result, "stdout", "") or "")
    if mode == "stdout_text":
        text = stdout.strip()
        raw_result = {"final_result": text, "reward_info": {"final_result": text}}
        return _normalize_result(raw_result, result_config)
    if mode == "stdout_json":
        raw_result = _parse_stdout_json(stdout)
        return _normalize_result(raw_result, result_config)
    else:
        raise ValueError(f"Unsupported external MAS result mode: {mode!r}")


def _safe_filename_component(value: Any, *, default: str = "rollout") -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or default


def _write_external_mas_logs(
    *,
    logs: Mapping[str, Any] | None,
    session_group_id: str,
    stdout: str,
    stderr: str,
) -> None:
    """Persist one rollout's process output when a log directory is configured."""
    log_dir = (logs or {}).get("directory")
    if not log_dir:
        return

    output_dir = Path(str(log_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_name = _safe_filename_component(session_group_id)
    (output_dir / f"{rollout_name}.stdout.log").write_text(
        stdout,
        encoding="utf-8",
        newline="",
    )
    (output_dir / f"{rollout_name}.stderr.log").write_text(
        stderr,
        encoding="utf-8",
        newline="",
    )


def _write_external_mas_logs_best_effort(
    *,
    logs: Mapping[str, Any] | None,
    session_group_id: str,
    stdout: str,
    stderr: str,
) -> None:
    """Write diagnostics without replacing the original rollout outcome."""
    try:
        _write_external_mas_logs(
            logs=logs,
            session_group_id=session_group_id,
            stdout=stdout,
            stderr=stderr,
        )
    except Exception:
        logger.exception(
            "failed to write external MAS logs for session %s",
            session_group_id,
        )


def _stdout_of(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _stderr_of(result: Any) -> str:
    return str(getattr(result, "stderr", "") or "")


def _exit_code_of(result: Any) -> int:
    return int(getattr(result, "exit_code", 0) or 0)


async def _execute_mas_command(
    *,
    sandbox: Sandbox,
    command: Mapping[str, Any],
    config: Mapping[str, Any],
    request: ExternalMASRequest,
    injected: Mapping[str, Any],
    work_dir: str | None,
    timeout_seconds: Mapping[str, Any] | float | int | None,
) -> Any:
    """Write the rendered config into the sandbox and run the MAS command there.

    The config path is generated inside the sandbox filesystem (not on the host)
    because a remote provider owns the filesystem the MAS reads; ``{config_path}``
    therefore resolves to a path the command can actually open.
    ``config.config_dir`` overrides the directory, which is also where the config
    is cleaned up after the run.
    """
    config_dir = str(config.get("config_dir") or _DEFAULT_SANDBOX_CONFIG_DIR).rstrip("/") or "/"
    safe_id = _safe_filename_component(request.session_group_id, default="rollout")
    config_path = f"{config_dir}/uni_mas_{safe_id}_{uuid.uuid4().hex[:12]}.yaml"

    context = {
        "config_path": config_path,
        "prompt": request.prompt,
        "session_group_id": request.session_group_id,
    }
    process_command, shell = _build_command(command, context)
    process_env = _format_values(command.get("env", {}), context)
    timeout = float(timeout_seconds) if timeout_seconds is not None else None

    await sandbox.write_file(
        config_path,
        yaml.safe_dump(dict(injected), sort_keys=False, allow_unicode=True),
    )
    try:
        if shell:
            exec_result = await sandbox.exec_shell(
                str(process_command),
                timeout=timeout,
                workdir=work_dir,
                env=process_env or None,
            )
        else:
            exec_result = await sandbox.exec(
                list(process_command),
                timeout=timeout,
                workdir=work_dir,
                env=process_env or None,
            )
    finally:
        # The rollout config is written per execution, so remove it even when the
        # command fails. Cleanup failures must not mask the original outcome.
        try:
            await sandbox.exec(["rm", "-f", config_path])
        except Exception:
            logger.debug("failed to remove sandbox config %s", config_path, exc_info=True)

    return exec_result


async def run_external_mas(
    *,
    request: ExternalMASRequest,
    sandbox: Sandbox,
    command: Mapping[str, Any],
    config: Mapping[str, Any],
    timeout: Mapping[str, Any] | float | int | None = None,
    result: Mapping[str, Any] | None = None,
    logs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Launch one external MAS command in the rollout's sandbox.

    The sandbox is required: ``ExternalMASTask`` builds it through
    ``build_sandbox()``, so the provider comes from the Task Config exactly like
    the other recipes. ``LocalSandbox`` runs the command on the host; container
    and remote providers run it in their own environment.
    """
    if not request.role_urls:
        raise ValueError("external MAS request requires at least one role URL")
    if sandbox is None:
        raise ValueError("external MAS execution requires a started sandbox")

    template_path = config.get("template_path")
    if not template_path:
        raise ValueError("config.template_path is required")
    with Path(str(template_path)).expanduser().open(encoding="utf-8") as file:
        template = yaml.safe_load(file) or {}
    if not isinstance(template, Mapping):
        raise ValueError("External MAS config template must contain a mapping")

    template = _apply_overrides(template, config.get("overrides"))
    injected = inject_gateway_config(
        template,
        role_urls=request.role_urls,
        injection=config.get("injection"),
    )
    timeout_seconds = timeout.get("seconds") if isinstance(timeout, Mapping) else timeout
    work_dir = str(command["work_dir"]) if command.get("work_dir") else None

    exec_result = await _execute_mas_command(
        sandbox=sandbox,
        command=command,
        config=config,
        request=request,
        injected=injected,
        work_dir=work_dir,
        timeout_seconds=timeout_seconds,
    )

    exit_code = _exit_code_of(exec_result)
    stdout = _stdout_of(exec_result)
    stderr = _stderr_of(exec_result)
    _write_external_mas_logs_best_effort(
        logs=logs,
        session_group_id=request.session_group_id,
        stdout=stdout,
        stderr=stderr,
    )
    # The sandbox contract reports a timeout as exit code -1, so the timeout path
    # is a result rather than an exception.
    if exit_code == _SANDBOX_TIMEOUT_EXIT_CODE:
        raise ExternalMASTimeout(
            f"External MAS command timed out after {timeout_seconds}s",
            stdout=stdout,
            stderr=stderr,
        )
    if exit_code != 0:
        raise ExternalMASError(
            f"External MAS command exited with code {exit_code}: {stderr.strip()}",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )
    return _parse_result(exec_result, result or {})
