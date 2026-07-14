"""Agent layer: *who* solves a task and *how it is launched*."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox


class ModelConfig(BaseModel):
    """The OpenAI-compatible LLM endpoint the agent's policy talks to, plus sampling knobs."""

    base_url: str | None = Field(
        default=None, description="Endpoint URL; the runner fills this in (in RL, the current policy server)."
    )
    api_key: str = Field(default="EMPTY", description="Bearer key (the gateway accepts any non-empty value).")
    model_name: str | None = Field(
        default=None, description="Model name sent to the endpoint (the served model / policy)."
    )

    # Sampling knobs -- keep aligned with the RL rollout config so inference == training.
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    top_p: float = Field(default=1.0, description="Nucleus-sampling probability mass.")
    top_k: int = Field(default=-1, description="Top-k sampling; -1 disables it.")

    # Generation budget: one turn's generation vs the whole episode's generation.
    max_total_tokens: int | None = Field(
        default=None,
        description="Whole-episode generation budget (sum of completion tokens over all turns)",
    )
    max_tokens_per_turn: int | None = Field(
        default=None,
        description="Per-turn generation cap, sent as `max_tokens` on each chat-completions call.",
    )

    model_config = ConfigDict(extra="forbid")


class AgentConfig(BaseModel):
    """Base config for a registered agent."""

    name: str = Field(default="", description="Registered agent name (key in AGENT_REGISTRY).")
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="LLM endpoint + sampling params for the policy."
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


@dataclasses.dataclass
class AgentResult:
    """Artifacts one agent produced for an episode -- the task scores these."""

    output: dict[str, Any] = dataclasses.field(default_factory=dict)
    transcript: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class Agent(ABC):
    """A solver bound to an :class:`AgentConfig`, runnable over a live sandbox."""

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
        """Solve the task described by ``messages`` inside the live ``sandbox``.

        ``sandbox`` is already started/provisioned (the task stops it afterwards);
        ``messages`` is the prompt in OpenAI chat form. Talks to the agent's own
        :attr:`config.model` and returns the artifacts the task scores.
        """
        ...
