"""Generic command-driven external MAS runner.

Each invocation receives a rollout-scoped Gateway URL, injects it into a copy
of the MAS configuration, and launches the configured command in the Ray
worker that owns this rollout task.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .process_backend import LocalProcessResult, run_local_process


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


def _safe_filename_component(value: Any, *, default: str = "rollout") -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or default


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


_COMMAND_PLACEHOLDERS = ("config_path", "prompt", "session_group_id")
_UNSUPPORTED_PLACEHOLDERS = ("gateway_url",)
_COMMAND_PLACEHOLDER_RE = re.compile(
    r"(?<![$\{])\{(?P<key>" + "|".join(re.escape(name) for name in _COMMAND_PLACEHOLDERS) + r")\}"
)
_UNSUPPORTED_PLACEHOLDER_RE = re.compile(
    r"(?<![$\{])\{(?P<key>" + "|".join(re.escape(name) for name in _UNSUPPORTED_PLACEHOLDERS) + r")\}"
)


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


_MISSING = object()


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
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index + end, index, value))

    if candidates:
        _, _, result = max(candidates)
        return result
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


def _parse_result(process_result: LocalProcessResult, result_config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(result_config.get("mode", "stdout_json"))
    if mode == "none":
        return {}
    if mode == "stdout_text":
        text = process_result.stdout.strip()
        raw_result = {"final_result": text, "reward_info": {"final_result": text}}
        return _normalize_result(raw_result, result_config)
    if mode == "stdout_json":
        raw_result = _parse_stdout_json(process_result.stdout)
        return _normalize_result(raw_result, result_config)
    else:
        raise ValueError(f"Unsupported external MAS result mode: {mode!r}")


async def run_external_mas(
    *,
    request: ExternalMASRequest,
    execution: Mapping[str, Any],
    command: Mapping[str, Any],
    config: Mapping[str, Any],
    timeout: Mapping[str, Any] | float | int | None = None,
    result: Mapping[str, Any] | None = None,
    logs: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Launch one external MAS command against the rollout Gateway."""
    if not request.role_urls:
        raise ValueError("external MAS request requires at least one role URL")

    backend = str(execution.get("backend", "local_process"))
    if backend != "local_process":
        raise NotImplementedError(f"External MAS execution backend {backend!r} is not implemented")

    template_path = config.get("template_path")
    if not template_path:
        raise ValueError("config.template_path is required")
    with Path(str(template_path)).expanduser().open(encoding="utf-8") as file:
        template = yaml.safe_load(file) or {}
    if not isinstance(template, Mapping):
        raise ValueError("External MAS config template must contain a mapping")

    injected = inject_gateway_config(
        template,
        role_urls=request.role_urls,
        injection=config.get("injection"),
    )
    temp_dir = config.get("temp_dir")
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="uni_mas_",
        dir=str(temp_dir) if temp_dir else None,
        delete=False,
        encoding="utf-8",
    )
    config_path = Path(temporary.name)
    try:
        yaml.safe_dump(injected, temporary, sort_keys=False, allow_unicode=True)
        temporary.close()

        context = {
            "config_path": str(config_path),
            "prompt": request.prompt,
            "session_group_id": request.session_group_id,
        }
        process_command, shell = _build_command(command, context)
        process_env = _format_values(command.get("env", {}), context)
        timeout_seconds = timeout.get("seconds") if isinstance(timeout, Mapping) else timeout

        process_result = await run_local_process(
            process_command,
            work_dir=str(command["work_dir"]) if command.get("work_dir") else None,
            env=process_env,
            timeout_seconds=(float(timeout_seconds) if timeout_seconds is not None else None),
            shell=shell,
        )
        log_dir = (logs or {}).get("directory")
        if log_dir:
            output_dir = Path(str(log_dir)).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            rollout_name = _safe_filename_component(request.session_group_id)
            (output_dir / f"{rollout_name}.stdout.log").write_text(
                process_result.stdout,
                encoding="utf-8",
                newline="",
            )
            (output_dir / f"{rollout_name}.stderr.log").write_text(
                process_result.stderr,
                encoding="utf-8",
                newline="",
            )
        return _parse_result(process_result, result or {})
    finally:
        temporary.close()
        config_path.unlink(missing_ok=True)
