"""HotpotQA task for long-context question answering."""

from __future__ import annotations

from pydantic import Field

from uni_agent.agents.mem_agent import ContextManagerResult

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_score


class HotpotQATaskConfig(TaskConfig):
    name: str = "hotpotqa"
    ground_truth: list[str] = Field(
        default_factory=list,
        description="Accepted answers; falls back to metadata.reward_model.ground_truth.",
    )


@register_task("hotpotqa")
class HotpotQATask(Task):
    """Score a HotpotQA answer and broadcast its reward to every context chain."""

    config_model = HotpotQATaskConfig

    async def run(self) -> TaskResult:
        cfg: HotpotQATaskConfig = self.config  # type: ignore[assignment]
        raw_data = dict(cfg.metadata)

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir=None,
                raw_data=raw_data,
            )

        response = str(agent_result.output.get("response", ""))
        ground_truths = list(cfg.ground_truth)
        if not ground_truths:
            reward_model = cfg.metadata.get("reward_model", {})
            raw_ground_truth = reward_model.get("ground_truth", []) if isinstance(reward_model, dict) else []
            if isinstance(raw_ground_truth, str):
                ground_truths = [raw_ground_truth]
            elif isinstance(raw_ground_truth, list | tuple):
                ground_truths = [str(answer) for answer in raw_ground_truth]
            elif raw_ground_truth is not None:
                ground_truths = [str(raw_ground_truth)]

        reward = compute_score(response, ground_truths)
        context_manager_result = agent_result.output.get("context_manager_result")
        thinking_turns = 0
        if isinstance(context_manager_result, ContextManagerResult):
            context_manager_result.set_reward(reward)
            thinking_turns = sum(
                "<think" in turn.response.lower() or "</think>" in turn.response.lower()
                for context_step in context_manager_result.trajectory
                for turn in context_step.steps
            )

        return TaskResult(
            reward=reward,
            accuracy=reward,
            extra_info={
                "response": response,
                "num_contexts": agent_result.info.get("num_contexts", 0),
                "total_steps": agent_result.info.get("total_steps", 0),
                "thinking_detected": thinking_turns > 0,
                "thinking_turns": thinking_turns,
            },
        )
