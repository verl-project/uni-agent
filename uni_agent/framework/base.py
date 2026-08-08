from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tensordict import TensorDict


@dataclass
class EpisodeResult:
    """Typed annotations returned by a managed Agent Runner."""

    reward: float | None = None
    metrics: dict[str, int | float | bool] = field(default_factory=dict)
    episode_finished: bool | None = None
    reward_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward is not None and (isinstance(self.reward, bool) or not isinstance(self.reward, int | float)):
            raise ValueError("EpisodeResult.reward must be a number or None")
        if self.episode_finished is not None and type(self.episode_finished) is not bool:
            raise ValueError("EpisodeResult.episode_finished must be a bool or None")
        if not isinstance(self.metrics, dict):
            raise ValueError("EpisodeResult.metrics must be a dict")
        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise ValueError("EpisodeResult.metrics keys must be strings")
            if key == "reward":
                raise ValueError("EpisodeResult.metrics key 'reward' is reserved for the outcome reward")
            if not isinstance(value, int | float | bool):
                raise ValueError(f"EpisodeResult.metrics[{key!r}] must be scalar (int/float/bool)")
        if not isinstance(self.reward_context, dict):
            raise ValueError("EpisodeResult.reward_context must be a dict")


class AgentFramework(ABC):
    """Abstract base for trainer-driven agent frameworks."""

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        *,
        config,
        **kwargs,
    ) -> AgentFramework: ...

    @abstractmethod
    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run agent sessions and write finalized trajectories to TransferQueue."""
        ...
