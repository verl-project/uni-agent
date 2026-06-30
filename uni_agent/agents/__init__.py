"""Agent layer: the solver that runs *outside* the task image over a sandbox.

See :mod:`uni_agent.agents.base` for the abstraction. An agent's launch params
live in an :class:`AgentConfig` subclass (replacing the old ``AgentSpec``); a
task picks an agent by setting ``TaskConfig.agent`` to one of these and the
runner builds it with :func:`build_agent`::

    from uni_agent.agents import build_agent
    from uni_agent.agents.code_act import CodeActConfig

    agent = build_agent(CodeActConfig())     # white-box: native framework loop
    # ... task starts + provisions the sandbox, then:
    # result = await agent.run(sandbox=sandbox, sample=sample, gateway=gateway)

Concrete agents under ``agents/<name>/`` register themselves and are imported
*lazily* by :func:`build_agent` (see ``AGENT_MODULES``), so importing this
package never forces an agent's optional deps to be installed.
"""

from __future__ import annotations

from .base import Agent, AgentConfig, AgentResult
from .registry import (
    AGENT_MODULES,
    AGENT_REGISTRY,
    build_agent,
    get_agent_cls,
    register_agent,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "build_agent",
    "get_agent_cls",
    "register_agent",
    "AGENT_REGISTRY",
    "AGENT_MODULES",
]
