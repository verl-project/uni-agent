"""Task layer: one runnable problem family = sandbox + agent (+ gateway at run time).

A *task* is the top-level unit a trainer / evaluator instantiates. The base
:class:`TaskConfig` holds only what *every* task shares:

* **sandbox** -- where execution happens (:class:`~uni_agent.sandbox.SandboxConfig`).
* **agent**   -- *who* solves it and *how it is launched*, picked from the agent
  layer (an :class:`~uni_agent.agents.AgentConfig`; see :mod:`uni_agent.agents`).

The **gateway** (the LLM the agent talks to) is a live runtime object, not config.
The runner installs one process-global
:class:`~uni_agent.gateway.manager.GatewayManager` (via
:func:`~uni_agent.gateway.set_gateway_manager`) and the task reads it with
:func:`~uni_agent.gateway.get_gateway_manager` inside :meth:`run`, so ``run`` takes
no arguments. White-box agents drive the policy through it; black-box agents point
their own process at its session URL. A task that needs no model (e.g. an oracle
gold-patch run) never fetches it.

Reward is **not** a base concern either: each task declares its scorer
(``reward.py``) and calls :func:`~uni_agent.reward.load_reward_spec` itself inside
:meth:`run`.

The solving strategy is **not** task-specific: agents live in their own layer and
are reused across tasks. A concrete task only *selects* one and wires the world:

* set ``agent`` to a concrete :class:`~uni_agent.agents.AgentConfig` subclass
  (e.g. ``CodeActConfig`` for the white-box framework loop, or ``ClaudeCodeConfig``
  for a black box launched in the sandbox); the base only types it as the shared
  :class:`~uni_agent.agents.AgentConfig`.
* subclass :class:`TaskConfig` to narrow ``agent`` to that config and add typed
  knobs (dataset, split, per-instance setup, ...).

Each concrete task lives in ``tasks/<name>/run.py`` and is constructed with an
explicit :class:`TaskConfig` (there is no default -- the caller always passes one).
The base turns the shared pieces into runtime objects (:meth:`build_sandbox`,
:meth:`build_agent`) so a runner stays task-agnostic; the task owns the sandbox
lifecycle + per-instance provisioning and hands the live sandbox to the agent.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..agents import AgentConfig
from ..sandbox import SandboxConfig

if TYPE_CHECKING:
    from ..agents import Agent
    from ..sandbox import Sandbox


class TaskConfig(BaseModel):
    """Base task config: only the fields every task shares.

    Concrete tasks subclass this to narrow :attr:`agent` to a concrete
    :class:`~uni_agent.agents.AgentConfig` subclass and to add their own typed knobs.
    """

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig, description="Execution sandbox.")
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent that solves the task; a concrete AgentConfig subclass.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass
class TaskResult:
    """Outcome of one task episode: the reward plus auxiliary info."""

    reward: Any
    info: dict[str, Any] | None = None


class Task(ABC):
    """A task family: turns a :class:`TaskConfig` into the runnable lower layers.

    Concrete tasks live in ``tasks/<name>/run.py``: set :attr:`name`, subclass
    :class:`TaskConfig`, and implement :meth:`run`. A config is always passed in
    explicitly (there is no default). The base provides the config -> runtime glue
    (:meth:`build_sandbox`, :meth:`build_agent`) so runners stay generic; reward
    scoring is each task's own concern, done in :meth:`run`.
    """

    name: ClassVar[str] = ""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    async def run(self) -> TaskResult:
        """Run one episode and return its score.

        Takes no arguments: the sample is :attr:`TaskConfig.metadata` and, when a
        model is needed, the gateway is the process-global
        :func:`~uni_agent.gateway.get_gateway_manager` (installed by the runner).
        White-box tasks serve the policy through it; black-box tasks point their own
        process at its session URL.
        """
        ...

    def build_sandbox(self) -> Sandbox:
        """Instantiate the execution sandbox from :attr:`TaskConfig.sandbox`."""
        from ..sandbox import build_sandbox

        return build_sandbox(self.config.sandbox)

    def build_agent(self) -> Agent:
        """Instantiate the solving agent from :attr:`TaskConfig.agent`."""
        from ..agents import build_agent

        return build_agent(self.config.agent)
