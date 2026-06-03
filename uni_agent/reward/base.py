"""Abstract base for reward specs."""

from abc import ABC, abstractmethod


class AbstractRewardSpec(ABC):
    """Reward spec: computes reward from interaction result and optional env eval."""

    @abstractmethod
    def compute_reward(self):
        """
        Compute reward (and optionally run eval in env) and return agent loop output.

        Returns:
            AgentLoopOutput with reward_score and token ids from interaction_result.
        """
        ...
