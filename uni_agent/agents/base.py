"""Agent layer: *who* solves a task and *how it is launched*.

An :class:`Agent` turns an :class:`AgentConfig` into a runnable solver over a live
sandbox, talking to the model at its own :attr:`AgentConfig.model` endpoint (the
runner fills it in; in RL it points at the current policy server). Agents differ
in where the loop runs and whether we control it:

* **white-box** (e.g. ``code_act``) -- our framework loop runs host-side, drives
  host-side tools, and calls the policy itself.
* **black-box** (e.g. ``claude_code``) -- an opaque solver launched *inside* the
  sandbox with its own loop + tools, pointed at the *same* endpoint so its model
  calls still become trainable trajectories.

Each agent lives under ``agents/<name>/`` and registers itself
(:func:`~uni_agent.agents.registry.register_agent`); a task builds one by name
(:func:`~uni_agent.agents.registry.build_agent`). The agent owns neither the
sandbox nor its lifecycle: the task hands it a *live* sandbox plus ``messages``,
then stops the sandbox and scores whatever :meth:`Agent.run` returns.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..sandbox import Sandbox


class ModelConfig(BaseModel):
    """The OpenAI-compatible LLM endpoint the agent's policy talks to, plus sampling knobs."""

    base_url: str | None = Field(
        default=None, description="Endpoint URL; the runner fills this in (in RL, the current policy server)."
    )
    api_key: str = Field(default="EMPTY", description="Bearer key (the gateway accepts any non-empty value).")
    sampling_params: dict[str, Any] = Field(
        default_factory=dict, description="Sampling knobs (temperature, top_p, max_tokens, ...)."
    )

    model_config = ConfigDict(extra="forbid")


class AgentConfig(BaseModel):
    """Base config for a registered agent.

    Nearly empty on purpose: agents take very different launch params, so each
    defines its own subclass under ``agents/<name>/``. The two shared fields are
    :attr:`name` (the registry key :func:`~uni_agent.agents.registry.build_agent`
    dispatches on; subclasses default it to their own key) and :attr:`model` (the
    endpoint the agent's policy talks to).
    """

    name: str = Field(default="", description="Registered agent name (key in AGENT_REGISTRY).")
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="LLM endpoint + sampling params for the policy."
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


@dataclasses.dataclass
class AgentResult:
    """Artifacts one agent produced for an episode -- the task scores these.

    * :attr:`output` -- the solution payload the task's reward consumes (e.g. a
      ``patch`` for SWE-bench).
    * :attr:`transcript` -- the step-by-step trace; white-box loops fill it, a
      black box may leave it empty.
    * :attr:`info` -- free-form diagnostics (exit codes, token usage, ...).
    """

    output: dict[str, Any] = dataclasses.field(default_factory=dict)
    transcript: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class Agent(ABC):
    """A solver bound to an :class:`AgentConfig`, runnable over a live sandbox.

    Concrete agents live under ``agents/<name>/`` (set :attr:`config_model`,
    register with ``@register_agent("<name>")`` which stamps :attr:`name`) and
    implement :meth:`run`, talking to the model at their own
    :attr:`~AgentConfig.model` endpoint.
    """

    #: Registry key, stamped by ``@register_agent``.
    name: ClassVar[str] = ""
    #: Pydantic config subclass this agent is built from (carries :attr:`AgentConfig.model`).
    config_model: ClassVar[type[AgentConfig]] = AgentConfig

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or self.config_model()

    @classmethod
    def from_config(cls, config: AgentConfig) -> Agent:
        """Build an instance from its :class:`AgentConfig` (override to remap fields)."""
        return cls(config)

    @abstractmethod
    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        """Solve the task described by ``messages`` inside ``sandbox``.

        * ``sandbox`` is already *live* -- the task started it and did any
          per-instance provisioning (e.g. cloning the repo at the base commit), and
          stops it after this returns.
        * ``messages`` is the task prompt in OpenAI chat form (a ``user`` turn,
          optionally preceded by a ``system`` turn).

        The model endpoint is the agent's own :attr:`config.model`; returns the
        artifacts the task scores (see :class:`AgentResult`).
        """
        ...
