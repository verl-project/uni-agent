"""Task registry: register a task family by name and build it by name.

Mirrors the agent / sandbox registries. Concrete tasks live in
``tasks/<name>/task.py`` and register themselves with :func:`register_task`;
:func:`get_task` builds one from either a :class:`TaskConfig` (dispatched on
``config.name``, like ``build_agent``) or a flat ``{"name", ...fields}`` mapping
(the form training carries in ``extra_info.tools_kwargs.task``), importing the
task's module on first use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

from .base import Task, TaskConfig

TASK_REGISTRY: dict[str, type[Task]] = {}

#: name -> module that defines (and registers) the task, for lazy loading.
TASK_MODULES: dict[str, str] = {
    "swe_bench": "uni_agent.tasks.swe_bench.task",
    "swe_rebench": "uni_agent.tasks.swe_rebench.task",
}


def register_task(name: str) -> Callable[[type[Task]], type[Task]]:
    """Class decorator: register a :class:`Task` under ``name`` (and stamp ``cls.name``)."""

    def decorator(cls: type[Task]) -> type[Task]:
        if name in TASK_REGISTRY and TASK_REGISTRY[name] is not cls:
            raise ValueError(f"Task {name!r} already registered: {TASK_REGISTRY[name]!r} vs {cls!r}")
        cls.name = name
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task_cls(name: str) -> type[Task]:
    """Return a registered task class by name, importing its module on first use."""
    if name not in TASK_REGISTRY and name in TASK_MODULES:
        import_module(TASK_MODULES[name])
    if name not in TASK_REGISTRY:
        available = sorted(set(TASK_REGISTRY) | set(TASK_MODULES))
        raise ValueError(f"Unknown task: {name!r}. Available: {available}")
    return TASK_REGISTRY[name]


def get_task(config: TaskConfig | Mapping[str, Any]) -> Task:
    """Build a task from a :class:`TaskConfig` or a flat config mapping.

    Both dispatch on ``name`` (the registry key): a :class:`TaskConfig` reads
    :attr:`~TaskConfig.name` (like ``build_agent``); a mapping -- the serialized
    form training carries in ``extra_info.tools_kwargs.task`` -- reads ``["name"]``
    and is unpacked whole into the task's :attr:`Task.config_model`.
    """
    if isinstance(config, TaskConfig):
        return get_task_cls(config.name)(config)
    if isinstance(config, Mapping):
        name = config.get("name")
        if not name:
            raise KeyError("task config has no 'name'")
        cls = get_task_cls(name)
        return cls(cls.config_model(**config))
    raise TypeError(f"get_task config must be a TaskConfig or mapping, got {type(config).__name__}")
