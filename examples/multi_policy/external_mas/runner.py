"""External-MAS runner entry point.

Importing this module registers ``ExternalMASTask`` and exposes the generic
multi-role runner configured under ``agent_runners``.
"""

from examples.multi_policy.external_mas.task import ExternalMASTask, ExternalMASTaskConfig
from examples.multi_policy.framework.multi_agent_runner import run_multi_agent_task

__all__ = ["ExternalMASTask", "ExternalMASTaskConfig", "run_multi_agent_task"]
