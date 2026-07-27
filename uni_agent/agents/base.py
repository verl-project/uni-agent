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

    # During RL, the Gateway session gets these defaults from the rollout config
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature; None inherits the endpoint/session default.",
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus-sampling probability mass; None inherits the endpoint/session default.",
    )
    top_k: int | None = Field(
        default=None,
        description="Top-k sampling; -1 disables it and None inherits the endpoint/session default.",
    )

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

    def sampling_params(self) -> dict[str, float | int]:
        """Return only sampling knobs explicitly configured by this Agent."""
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        return {key: value for key, value in params.items() if value is not None}


class AgentConfig(BaseModel):
    """Base config for a registered agent."""

    name: str = Field(default="", description="Registered agent name (key in AGENT_REGISTRY).")
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="LLM endpoint + sampling params for the policy."
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


@dataclasses.dataclass
class AgentResult:
    """Artifacts one Agent produced for an episode.

    ``finished`` is tri-state: ``True``/``False`` report a known completion
    state, and ``None`` means the Agent does not track one. Agents that never
    set it stay indistinguishable from Agents that finished normally, so
    opting out never silently marks an episode untrainable.
    """

    output: dict[str, Any] = dataclasses.field(default_factory=dict)
    transcript: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    finished: bool | None = None


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
