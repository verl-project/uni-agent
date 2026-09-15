from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from tensordict import TensorDict

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle
    from uni_agent.tasks import TaskResult


class AgentRunner(Protocol):
    """Callable that executes one agent episode against a Framework-owned Gateway session."""

    async def __call__(
        self,
        *,
        session: SessionHandle,
        raw_prompt: object,
        sample_index: int,
        **sample_runner_kwargs: object,
    ) -> TaskResult | None: ...


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
