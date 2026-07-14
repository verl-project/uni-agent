"""mini-swe-agent: black-box agent launched *inside* the sandbox. NOT YET IMPLEMENTED.

Planned: install mini-swe-agent into an isolated venv (off the task image's env), point
it at ``config.model`` via LiteLLM, launch it against ``/testbed``, and let the task's
reward step score the resulting ``git diff``.

Reference: https://github.com/SWE-agent/mini-swe-agent
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


class MiniSweAgentConfig(AgentConfig):
    """Black-box launch params for mini-swe-agent (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "mini_swe_agent"


@register_agent("mini_swe_agent")
class MiniSweAgentAgent(Agent):
    """Black-box solver stub: will launch mini-swe-agent in the sandbox (``run`` not yet implemented)."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        pass
