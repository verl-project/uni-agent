"""Abstract base for reward specs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class AbstractRewardSpec(ABC):
    """Reward spec: computes reward from interaction result and optional env eval."""

    @abstractmethod
    def compute_reward(self) -> AgentLoopOutput:
        """
        Compute reward (and optionally run eval in env) and return agent loop output.

        Returns:
            AgentLoopOutput with reward_score and token ids from interaction_result.
        """
        ...
