"""Task layer: compose gateway + sandbox + agent into one family.

See :mod:`uni_agent.tasks.base` for the abstraction. The base config holds only
the shared fields; concrete tasks under ``tasks/<name>/run.py`` subclass
:class:`TaskConfig`, set ``agent`` to a concrete
:class:`~uni_agent.agents.AgentConfig` (from :mod:`uni_agent.agents`), register
themselves, and are built by name with :func:`get_task` from a config you supply
(there is no default)::

    from uni_agent.gateway import set_gateway_manager
    from uni_agent.tasks import get_task
    from uni_agent.tasks.swe_bench.run import SWEBenchTaskConfig

    set_gateway_manager(gateway)   # runner installs the process-global gateway once
    task = get_task("swe_bench", SWEBenchTaskConfig(metadata=sample))
    result = await task.run()      # reads the global gateway; no args
"""

from __future__ import annotations

from .base import Task, TaskConfig, TaskResult
from .registry import TASK_REGISTRY, get_task, register_task

__all__ = [
    "Task",
    "TaskConfig",
    "TaskResult",
    "register_task",
    "get_task",
    "TASK_REGISTRY",
]
