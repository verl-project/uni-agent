from __future__ import annotations

from abc import ABC, abstractmethod

from tensordict import TensorDict


class AgentFramework(ABC):
    """Abstract base for trainer-driven agent frameworks."""

    @classmethod
    @abstractmethod
    async def from_config(
        cls,
        *,
        config,
        replay_buffer,
        **kwargs,
    ) -> AgentFramework: ...

    @abstractmethod
    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run agent sessions and write finalized trajectories to TransferQueue."""
        ...
