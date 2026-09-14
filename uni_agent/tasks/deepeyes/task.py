"""DeepEyes visual question-answering task."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import DeepEyesRewardConfig, compute_score

logger = logging.getLogger(__name__)


class DeepEyesTaskConfig(TaskConfig):
    """One DeepEyes sample plus the configured image-solving Agent."""

    name: str = "deepeyes"
    question: str = Field(min_length=1)
    ground_truth: str
    reward: DeepEyesRewardConfig = Field(default_factory=DeepEyesRewardConfig)


@register_task("deepeyes")
class DeepEyesTask(Task):
    """Run one image question and score the final answer with the DeepEyes Judge."""

    config_model = DeepEyesTaskConfig

    async def run(self) -> TaskResult:
        cfg: DeepEyesTaskConfig = self.config  # type: ignore[assignment]
        logger.info("DeepEyes task start: question=%s", cfg.question.strip())

        async with self.build_sandbox() as sandbox:
            agent_result = await self.build_agent().run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir=None,
            )

        final_answer_value = agent_result.output.get("final_answer")
        final_answer = final_answer_value if isinstance(final_answer_value, str) else ""
        telemetry = {
            "steps": int(agent_result.info.get("steps", 0) or 0),
            "tool_calls": int(agent_result.info.get("tool_calls", 0) or 0),
            "tool_successes": int(agent_result.info.get("tool_successes", 0) or 0),
            "tool_errors": int(agent_result.info.get("tool_errors", 0) or 0),
            "prompt_tokens": int(agent_result.info.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(agent_result.info.get("completion_tokens", 0) or 0),
            "total_tokens": int(agent_result.info.get("total_tokens", 0) or 0),
        }
        finished = agent_result.finished is True
        score = await asyncio.to_thread(
            compute_score,
            final_answer,
            cfg.ground_truth,
            question=cfg.question,
            finished=finished,
            tool_successes=telemetry["tool_successes"],
            reward_config=cfg.reward,
        )

        # ``reward``, ``accuracy``, and ``finished`` have canonical TaskResult
        # fields. Keep only scorer context and episode telemetry in extra_info so
        # the framework can forward it as runner reward context unchanged.
        extra_info: dict[str, Any] = {
            **{key: value for key, value in score.items() if key not in {"score", "acc"}},
            "termination_reason": agent_result.info.get("termination_reason", "unknown"),
            **telemetry,
        }
        logger.info(
            "DeepEyes task done: reward=%s acc=%s finished=%s calls=%s successes=%s errors=%s reason=%s",
            score["score"],
            score["acc"],
            finished,
            telemetry["tool_calls"],
            telemetry["tool_successes"],
            telemetry["tool_errors"],
            extra_info["termination_reason"],
        )
        return TaskResult(
            reward=float(score["score"]),
            accuracy=float(score["acc"]),
            finished=finished,
            extra_info=extra_info,
        )
