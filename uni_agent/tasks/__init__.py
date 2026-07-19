"""Task layer: compose sandbox + agent into one runnable family (gateway wired at run time).

See :mod:`uni_agent.tasks.base` for the abstraction. The base config holds only
the shared fields; concrete tasks under ``tasks/<name>/task.py`` subclass
:class:`TaskConfig`, set ``agent`` to a concrete
:class:`~uni_agent.agents.AgentConfig` (from :mod:`uni_agent.agents`), register
themselves, and are built with :func:`get_task` -- from either a config instance
(dispatched on its ``name``) or a flat ``{"name", ...fields}`` mapping::

    from uni_agent.tasks import get_task
    from uni_agent.tasks.swe_bench.task import SWEBenchTaskConfig

    task = get_task(SWEBenchTaskConfig(metadata=sample))        # a config instance
    task = get_task({"name": "swe_bench", "metadata": sample})  # or a flat mapping
    result = await task.run()      # runs one episode (prompt + endpoint live on the config)
"""

from __future__ import annotations

from .base import Task, TaskConfig, TaskResult
from .config import TaskConfigResolver
from .registry import get_task

__all__ = [
    "Task",
    "TaskConfig",
    "TaskConfigResolver",
    "TaskResult",
    "get_task",
]
