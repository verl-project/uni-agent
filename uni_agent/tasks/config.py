"""Task Config composition shared by standalone and framework-managed execution."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any


def deep_merge(base: dict, overrides: dict) -> dict:
    """Merge ``overrides`` onto ``base`` without mutating either mapping.

    Nested dictionaries merge recursively. Lists and scalar values are replaced.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_task_config(
    tools_kwargs: dict[str, Any] | None,
    *,
    session_base_url: str | None,
    task_defaults: dict[str, Any] | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Resolve Task defaults, sample overrides, and the live model endpoint."""
    if not tools_kwargs or "task" not in tools_kwargs:
        raise ValueError("run_task requires tools_kwargs['task'] (the serialized task config)")

    task = deep_merge(task_defaults or {}, tools_kwargs["task"])

    model_config: dict[str, Any] = {"base_url": session_base_url, "api_key": api_key}
    if model_name is not None:
        model_config["model_name"] = model_name
    return deep_merge(task, {"agent": {"model": model_config}})


@functools.lru_cache(maxsize=8)
def load_task_config_file(path: str) -> dict[str, dict[str, Any]]:
    """Load a Task Config YAML file into a ``{name: config}`` index."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    entries = raw if isinstance(raw, list) else [raw]
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"task_config_path {path!r}: each entry must be a mapping with a 'name' (got {entry!r})")
        index[str(entry["name"])] = entry
    return index


def route_task_config(path: str, task_name: str | None) -> dict[str, Any]:
    """Return the Task Config entry matching ``task_name``."""
    index = load_task_config_file(path)
    if task_name is None or task_name not in index:
        raise ValueError(f"task_config_path {path!r} has no config for task name {task_name!r} (have {sorted(index)})")
    return index[task_name]
