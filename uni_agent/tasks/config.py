"""Task Config composition shared by standalone and framework-managed execution."""

from __future__ import annotations

import functools
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PerTaskSamplingConfig(BaseModel):
    """Explicit overrides of VERL rollout sampling defaults for one task."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=-1)


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
    metadata: object,
    prompt_template: object,
) -> list[dict[str, Any]]:
    """Render text-only chat messages from direct Task metadata fields."""
    if not isinstance(prompt_template, list):
        raise ValueError("prompt_template must be a list of template messages")
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt_template metadata must be a mapping")

    formatter = string.Formatter()
    rendered: list[dict[str, Any]] = []
    for index, message in enumerate(prompt_template):
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt_template message {index} must be a template message mapping")
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
        rendered_parts: list[str] = []
        for literal_text, field_name, format_spec, conversion in parsed:
            rendered_parts.append(literal_text)
            if field_name is None:
                continue
            if not field_name.isidentifier():
                raise ValueError(
                    f"prompt_template message {index} must reference a direct metadata field; got {field_name!r}"
                )
            if conversion is not None:
                raise ValueError(f"prompt_template message {index} field {field_name!r} does not support conversion")
            if format_spec:
                raise ValueError(f"prompt_template message {index} field {field_name!r} does not support a format spec")
            try:
                value = metadata[field_name]
            except KeyError as exc:
                raise ValueError(
                    f"prompt_template message {index} references missing metadata field {field_name!r}"
                ) from exc
            if not isinstance(value, str):
                raise ValueError(
                    f"prompt_template message {index} requires text metadata field {field_name!r}; "
                    f"got {type(value).__name__}"
                )
            rendered_parts.append(value)
        rendered_content = "".join(rendered_parts)
        rendered.append({**message, "content": rendered_content})
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
        sample_values = dict(sample_config)
        # Typed TaskConfig dumps include None for unset fields. They must not
        # erase task YAML sampling defaults when passed through a dataset.
        sample_sampling = sample_values.get("per_task_sampling")
        if sample_sampling is None:
            sample_values.pop("per_task_sampling", None)
        elif isinstance(sample_sampling, dict):
            sample_values["per_task_sampling"] = {
                key: value for key, value in sample_sampling.items() if value is not None
            }
        resolved = _deep_merge(dict(file_defaults), sample_values)
        if "prompt_template" in file_defaults:
            resolved["prompt_template"] = file_defaults["prompt_template"]

        model_binding = {key: value for key, value in (runtime_model or {}).items() if value is not None}
        if model_binding:
            resolved = _deep_merge(resolved, {"agent": {"model": model_binding}})
        return resolved
