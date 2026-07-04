"""Task layer: one runnable problem family = sandbox + agent (+ gateway at run time).

A *task* is the top-level unit a trainer / evaluator instantiates. The base
:class:`TaskConfig` holds only what *every* task shares:

* **sandbox** -- where execution happens (:class:`~uni_agent.sandbox.SandboxConfig`).
* **agent**   -- *who* solves it and *how it is launched*, picked from the agent
  layer (an :class:`~uni_agent.agents.AgentConfig`; see :mod:`uni_agent.agents`).

The **model** the agent talks to is *not* a task-level concern: it lives on the
agent (:attr:`~uni_agent.agents.AgentConfig.model` -- an OpenAI-compatible
``base_url`` / ``api_key`` / ``sampling_params``), which the runner fills in (in RL
it points at the current policy server). A task that needs no model (e.g. an oracle
gold-patch run) simply never builds an agent.

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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator

from ..agents import AgentConfig
from ..sandbox import SandboxConfig

if TYPE_CHECKING:
    from ..agents import Agent
    from ..sandbox import Sandbox


class TaskConfig(BaseModel):
    """Base task config: only the fields every task shares.

    The model the agent talks to is *not* here -- it lives on :attr:`agent`'s
    :attr:`~uni_agent.agents.AgentConfig.model`.

    :attr:`agent` is polymorphic: it is typed as the base
    :class:`~uni_agent.agents.AgentConfig` but also accepts a ``{"name", ...}``
    mapping, which :meth:`_resolve_agent` parses into the concrete subclass
    registered under that name (so subclass fields like ``max_steps`` are kept
    instead of rejected by the base's ``extra="forbid"``). ``SerializeAsAny`` keeps
    those subclass fields on ``model_dump`` too, so a dict round-trip is lossless.
    """

    name: str = Field(default="", description="Registered task name (key in TASK_REGISTRY).")
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig, description="Execution sandbox.")
    agent: SerializeAsAny[AgentConfig] = Field(
        default_factory=AgentConfig,
        description="A concrete AgentConfig subclass, or a {name, ...} mapping resolved via the agent registry.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, v: Any) -> Any:
        """Parse an agent mapping into its concrete AgentConfig subclass.

        The field is typed as the base AgentConfig (``extra="forbid"``), so a plain
        dict would validate against the base and reject subclass fields. Dispatch on
        ``name`` via the agent registry instead (mirrors get_task / get_tool).
        """
        if isinstance(v, Mapping):
            from ..agents import get_agent_cls

            name = v.get("name")
            if not name:
                raise ValueError("task 'agent' config needs a 'name'")
            return get_agent_cls(name).config_model(**v)
        return v


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
    config_model: ClassVar[type[TaskConfig]] = TaskConfig

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    async def run(self) -> TaskResult:
        """Run one episode and return its score.

        Takes no arguments: the sample is :attr:`TaskConfig.metadata` and the model
        the agent talks to lives on :attr:`TaskConfig.agent`'s
        :attr:`~uni_agent.agents.AgentConfig.model` (filled in by the runner).
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
