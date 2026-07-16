"""Agent layer: pluggable solvers over a sandbox (white-box on the host, black-box inside the sandbox).

See :mod:`uni_agent.agents.base` for the abstraction. An agent's launch params
live in an :class:`AgentConfig` subclass; a task picks one by setting
``TaskConfig.agent`` and the runner builds it with :func:`build_agent`::

    from uni_agent.agents import build_agent
    from uni_agent.agents.react import ReActConfig

    agent = build_agent(ReActConfig())     # white-box: native framework loop
    # ... task starts + provisions the sandbox (endpoint lives on the config), then:
    # result = await agent.run(sandbox=sandbox, messages=messages)

Concrete agents under ``agents/<name>/`` register themselves and are imported
*lazily* by :func:`build_agent` (see ``AGENT_MODULES``), so importing this
package never forces an agent's optional deps to be installed.
"""

from __future__ import annotations

from .base import Agent, AgentConfig, AgentResult, ModelConfig
from .registry import build_agent, get_agent_cls

__all__ = [
    "Agent",
    "AgentConfig",
    "ModelConfig",
    "AgentResult",
    "build_agent",
    "get_agent_cls",
]
