"""Agent layer: *who* solves a task and *how it is launched*.

An :class:`Agent` turns an :class:`AgentConfig` into a runnable solver over a
live sandbox. Every agent talks to the model through an OpenAI-compatible
``base_url`` + ``api_key`` the *task* supplies (it derives them from the gateway
session it owns); agents never see the session itself. They differ in *where the
agent loop runs* and *whether we control it*:

* **white-box** (e.g. ``code_act``) -- our framework loop runs host-side, drives
  host-side tools, and calls the policy at ``base_url``.
* **black-box** (e.g. ``claude_code``) -- an opaque solver launched *inside* the
  sandbox with its own loop + tools, pointed at the *same* ``base_url`` so its
  model calls still become trainable trajectories. It's "black-box" because we
  don't drive its loop -- not because it uses a different model.

A concrete agent lives under ``agents/<name>/`` and registers itself with
:func:`~uni_agent.agents.registry.register_agent`; a task loads one by name with
:func:`~uni_agent.agents.registry.build_agent`. The agent owns **neither** the
sandbox nor the gateway-session lifecycle: the task starts the sandbox, provisions
the instance, creates the gateway session, then hands the *live* sandbox plus a
``base_url`` / ``api_key`` / ``messages`` to :meth:`Agent.run`, finalizes the
session, stops the sandbox, and scores whatever artifacts the agent returns.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..sandbox import Sandbox


class AgentConfig(BaseModel):
    """Base config for a registered agent (this replaces the old ``AgentSpec``).

    Near-empty on purpose: white-box and black-box agents take very different
    launch params, so each agent under ``agents/<name>/`` defines its own
    subclass with the fields it needs. The one shared field is :attr:`name` --
    the registry key that tells :func:`~uni_agent.agents.registry.build_agent`
    which agent to construct (mirrors ``SandboxConfig.provider``). Subclasses
    default :attr:`name` to their own registry key.
    """

    name: str = Field(default="", description="Registered agent name (key in AGENT_REGISTRY).")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


@dataclasses.dataclass
class AgentResult:
    """Artifacts one agent produced for an episode -- the task scores these.

    * :attr:`output` -- the solution payload the task's reward consumes (e.g. a
      ``patch`` for SWE-bench, plus any extras the scorer keys on). A task
      typically merges this with the ``sample`` and passes it to
      :func:`~uni_agent.reward.load_reward_spec`'s ``compute_reward``.
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
    implement :meth:`run`. Every agent talks to the model at the ``base_url`` /
    ``api_key`` the task passes in -- white-box loops call it from our framework,
    black-box solvers are launched in the sandbox pointed at the same endpoint.
    """

    #: Registry key, stamped by ``@register_agent``.
    name: ClassVar[str] = ""
    #: Pydantic config subclass this agent is built from.
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
        base_url: str,
        api_key: str,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        """Solve the task described by ``messages`` inside ``sandbox``.

        Everything is owned by the *task*, not the agent:

        * ``sandbox`` is already *live* -- the task started it and did any
          per-instance provisioning (e.g. cloning the repo at the base commit).
        * ``base_url`` / ``api_key`` point at an OpenAI-compatible endpoint (the
          task derives them from the gateway session it created). Agents never see
          the session: white-box loops call ``base_url`` from our framework,
          black-box solvers are launched in the sandbox pointed at the same URL.
        * ``messages`` is the task prompt in OpenAI chat form (a ``user`` turn,
          optionally preceded by a ``system`` turn). A white-box loop seeds its
          conversation with it; a black-box solver maps it onto its own launch.

        The task finalizes the session and stops the sandbox after this returns.
        """
        ...
