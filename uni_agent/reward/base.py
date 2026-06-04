"""Abstract base for reward specs."""

from abc import ABC, abstractmethod


class AbstractRewardSpec(ABC):
    """Reward spec: computes reward from interaction result and optional env eval."""

    @abstractmethod
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple:
        """
        Compute reward (and optionally run eval in env) from the interaction result.

        Returns:
            A 2-tuple whose first element is the reward score (or eval report) and
            whose second element is auxiliary info; the concrete element types
            depend on the reward spec.
        """
        ...
