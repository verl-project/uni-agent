"""Task registry: register a task family by name and load it by name.

Mirrors the reward / tools registries. Concrete tasks live in
``tasks/<name>/run.py`` and register themselves with :func:`register_task`;
:func:`get_task` builds one by name from a caller-supplied config, importing its
module on first use.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from .base import Task, TaskConfig

TASK_REGISTRY: dict[str, type[Task]] = {}

#: name -> module that defines (and registers) the task, for lazy loading.
TASK_MODULES: dict[str, str] = {
    "swe_bench": "uni_agent.tasks.swe_bench.run",
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


def get_task(name: str, config: TaskConfig) -> Task:
    """Instantiate a registered task by name with the given ``config``.

    There is no default config -- the caller must supply a :class:`TaskConfig`
    (a task-specific subclass, e.g. ``SWEBenchTaskConfig``). The gateway is not passed
    here: a task fetches the process-global one via
    :func:`~uni_agent.gateway.get_gateway_manager` when it needs a model.
    """
    if name not in TASK_REGISTRY and name in TASK_MODULES:
        import_module(TASK_MODULES[name])
    if name not in TASK_REGISTRY:
        available = sorted(set(TASK_REGISTRY) | set(TASK_MODULES))
        raise ValueError(f"Unknown task: {name!r}. Available: {available}")
    return TASK_REGISTRY[name](config)
