"""Task layer: one runnable problem family = sandbox + agent (+ gateway at run time).

A *task* is the top-level unit a trainer / evaluator instantiates. The base
:class:`TaskConfig` holds only what every task shares:

* **sandbox** -- where execution happens (:class:`~uni_agent.sandbox.SandboxConfig`).
* **agent**   -- who solves it and how it is launched (an
  :class:`~uni_agent.agents.AgentConfig`; the model it talks to lives on the agent,
  not here, and is filled in by the runner).
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator

from uni_agent.agents import AgentConfig
from uni_agent.logging import current_run_id, sample_logging
from uni_agent.sandbox import SandboxConfig

if TYPE_CHECKING:
    from uni_agent.agents import Agent
    from uni_agent.sandbox import Sandbox


class TaskConfig(BaseModel):
    """Base task config: only the fields every task shares (the model lives on
    :attr:`agent`, not here).

    :attr:`agent` is polymorphic -- typed as the base AgentConfig but also accepting
    a ``{"name", ...}`` mapping that :meth:`_resolve_agent` parses into the concrete
    subclass (keeping subclass fields like ``max_steps``). ``SerializeAsAny`` keeps
    those fields on ``model_dump`` too, so a dict round-trip is lossless.
    """

    name: str = Field(default="", description="Registered task name (key in TASK_REGISTRY).")
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig, description="Execution sandbox.")
    agent: SerializeAsAny[AgentConfig] = Field(
        default_factory=AgentConfig,
        description="A concrete AgentConfig subclass, or a {name, ...} mapping resolved via the agent registry.",
    )
    prompt: list[dict[str, Any]] = Field(default_factory=list, description="The task prompt.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    log_dir: str = Field(
        default="",
        description="Directory for this episode's per-sample log (<log_dir>/<run_id>/run.log); "
        "empty falls back to /tmp/uni_agent_logs/<task_name>.",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, v: Any) -> Any:
        """Parse a ``{"name", ...}`` mapping into its concrete AgentConfig subclass.

        The field is typed as the base (``extra="forbid"``), so a plain dict would
        reject subclass fields; dispatch on ``name`` via the agent registry instead.
        """
        if isinstance(v, Mapping):
            from uni_agent.agents import get_agent_cls

            name = v.get("name")
            if not name:
                raise ValueError("task 'agent' config needs a 'name'")
            return get_agent_cls(name).config_model(**v)
        return v


@dataclasses.dataclass
class TaskResult:
    """Outcome of one task episode."""

    reward: Any
    accuracy: float | None = None
    info: dict[str, Any] | None = None


class Task(ABC):
    """A task family: turns a :class:`TaskConfig` into the runnable lower layers.

    Concrete tasks live in ``tasks/<name>/task.py``: set :attr:`name`, subclass
    :class:`TaskConfig`, and implement :meth:`run` (which also does reward scoring).
    The base provides the config -> runtime glue (:meth:`build_sandbox`,
    :meth:`build_agent`) so runners stay generic.
    """

    name: ClassVar[str] = ""
    config_model: ClassVar[type[TaskConfig]] = TaskConfig

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    async def run(self) -> TaskResult:
        """Run one episode and return its score.

        Takes no arguments -- the sample is :attr:`TaskConfig.metadata` and the model
        lives on :attr:`TaskConfig.agent`'s ``model`` (filled in by the runner).
        """
        ...

    def build_sandbox(self) -> Sandbox:
        """Instantiate the execution sandbox from :attr:`TaskConfig.sandbox`."""
        from uni_agent.sandbox import build_sandbox

        return build_sandbox(self.config.sandbox)

    def build_agent(self) -> Agent:
        """Instantiate the solving agent from :attr:`TaskConfig.agent`."""
        from uni_agent.agents import build_agent

        return build_agent(self.config.agent)

    @asynccontextmanager
    async def episode_logging(self) -> AsyncIterator[None]:
        """Scope this episode's logs to a run_id via :func:`sample_logging`.

        Reuses the caller's ambient run_id when set (the framework binds
        run_id = session_id, so agent/tool/sandbox logs join that session's file);
        otherwise mints one and opens ``<log_dir>/<run_id>/run.log`` (standalone eval).
        Yields nothing -- use as ``async with self.episode_logging():``.
        """
        ambient_run_id = current_run_id()
        if ambient_run_id is not None:
            run_id, log_path = ambient_run_id, None
        else:
            run_id = str(uuid.uuid4())
            log_dir = os.path.expanduser(self.config.log_dir or f"/tmp/uni_agent_logs/{self.name}")
            log_path = Path(log_dir) / run_id / "run.log"
        async with sample_logging(run_id, log_path):
            yield
