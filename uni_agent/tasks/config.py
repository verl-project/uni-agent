"""Task Config composition shared by standalone and framework-managed execution."""

from __future__ import annotations

import functools
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge ``overrides`` onto ``base`` without mutating either mapping.

    Nested dictionaries merge recursively. Lists and scalar values are replaced.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def render_prompt_template(
    source_messages: object,
    prompt_template: object,
) -> list[dict[str, Any]]:
    """Render one source user message into a validated chat-message template."""
    if not isinstance(source_messages, list) or len(source_messages) != 1:
        raise ValueError("prompt_template requires exactly one source user message")
    source_message = source_messages[0]
    if not isinstance(source_message, Mapping) or source_message.get("role") != "user":
        raise ValueError("prompt_template source user message must be a mapping with role='user'")
    if "content" not in source_message:
        raise ValueError("prompt_template source user message must contain 'content'")
    source_content = source_message["content"]

    if not isinstance(prompt_template, list):
        raise ValueError("prompt_template must be a list of template messages")

    formatter = string.Formatter()
    parsed_messages: list[tuple[dict[str, Any], list[tuple[str, str | None, str | None, str | None]]]] = []
    prompt_fields = 0
    for index, message in enumerate(prompt_template):
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt_template message {index} must be a mapping")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"prompt_template message {index} must contain a non-empty string 'role'")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"prompt_template message {index} must contain string 'content'")
        try:
            parsed = list(formatter.parse(content))
        except ValueError as exc:
            raise ValueError(f"prompt_template message {index} has invalid content: {exc}") from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name != "prompt":
                raise ValueError(f"prompt_template contains unknown field {field_name!r}; only 'prompt' is allowed")
            if conversion is not None:
                raise ValueError("prompt_template field 'prompt' does not support conversion")
            if format_spec:
                raise ValueError("prompt_template field 'prompt' does not support a format spec")
            prompt_fields += 1
        parsed_messages.append((dict(message), parsed))

    if prompt_fields != 1:
        raise ValueError(f"prompt_template must contain exactly one 'prompt' field; found {prompt_fields}")

    rendered: list[dict[str, Any]] = []
    for message, _ in parsed_messages:
        content = message["content"]
        if content == "{prompt}":
            message["content"] = source_content
        else:
            if "{prompt}" in content and not isinstance(source_content, str):
                raise ValueError("embedded prompt substitution requires source content to be a string")
            message["content"] = formatter.vformat(content, (), {"prompt": source_content})
        rendered.append(message)
    return rendered


@functools.lru_cache(maxsize=8)
def _load_task_config_file(path: str) -> dict[str, dict[str, Any]]:
    """Load a Task Config YAML file into a ``{name: config}`` index."""
    import yaml

    raw = yaml.safe_load(Path(path).expanduser().read_text())
    entries = raw if isinstance(raw, list) else [raw]
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"task_config_path {path!r}: each entry must be a mapping with a 'name' (got {entry!r})")
        name = str(entry["name"])
        if name in index:
            raise ValueError(f"task_config_path {path!r} contains duplicate task name {name!r}")
        index[name] = entry
    return index


@dataclass(frozen=True)
class TaskConfigResolver:
    """Route and compose Task Config defaults, sample values, and runtime bindings."""

    defaults_by_name: Mapping[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> TaskConfigResolver:
        """Build a resolver from a YAML mapping or list keyed by Task ``name``."""
        return cls(defaults_by_name=_load_task_config_file(path))

    def resolve(
        self,
        sample_config: Mapping[str, Any],
        *,
        runtime_model: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one sample using Task Config → Sample Config → runtime model."""
        task_name = sample_config.get("name")
        if not task_name:
            raise ValueError("sample Task Config requires a 'name'")

        if self.defaults_by_name and task_name not in self.defaults_by_name:
            raise ValueError(
                f"no Task Config for sample task {task_name!r}; available configs: {sorted(self.defaults_by_name)}"
            )

        file_defaults = self.defaults_by_name.get(str(task_name), {})
        resolved = _deep_merge(dict(file_defaults), dict(sample_config))
        if "prompt_template" in file_defaults:
            resolved["prompt_template"] = file_defaults["prompt_template"]

        model_binding = {key: value for key, value in (runtime_model or {}).items() if value is not None}
        if model_binding:
            resolved = _deep_merge(resolved, {"agent": {"model": model_binding}})
        return resolved
