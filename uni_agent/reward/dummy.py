from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec


@register_reward_spec("dummy")
class DummyRewardSpec(AbstractRewardSpec):
    """Dummy reward for the quickstart: 1.0 if the agent called submit, else 0.0."""

    def __init__(self, **kwargs):
        # Absorbs run_id / env injected by the agent loop; unused here.
        pass

    async def compute_reward(self, interaction_result: dict, **kwargs):
        trajectory = interaction_result.get("trajectory", [])
        submitted = any(step.exit_reason == "finished" for step in trajectory)
        return (1.0 if submitted else 0.0), {"submitted": submitted}
