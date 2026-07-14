"""mini-swe-agent: a black-box agent launched *inside* the sandbox.

Like :mod:`~uni_agent.agents.claude_code`, mini-swe-agent owns its own loop and
tools (a single "bash" tool), so the host side only needs to: install it into
an isolated venv (kept off the task image's own Python/conda env so it can
never shadow the repo's dependencies), point it at ``config.model`` via
LiteLLM, launch it against ``/testbed``, and let the task's own reward step
(a plain ``git diff`` + eval harness run) score whatever it left on disk.

Reference: https://github.com/SWE-agent/mini-swe-agent
(``minisweagent.agents.default.DefaultAgent`` + ``minisweagent.models.litellm_model.LitellmModel``
+ ``minisweagent.environments.local.LocalEnvironment``).
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
    """Black-box solver: launch mini-swe-agent in the sandbox, pointed at ``config.model``."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        pass
