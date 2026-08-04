# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uni_agent.framework.base import EpisodeResult
from uni_agent.tasks import TaskConfigResolver, get_task

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_config_path: str | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
) -> EpisodeResult:
    """Resolve the sample's task, run it against ``session``, and return its result.

    Satisfies the framework's ``AgentRunner`` contract (``session`` / ``raw_prompt``
    / ``sample_index`` / ``tools_kwargs``). ``raw_prompt`` is accepted for protocol
    parity but unused: a uni_agent task carries its own prompt on the task config.

    Run-level defaults come from the per-task-name YAML file selected by
    ``task_config_path``. ``TaskConfigResolver`` applies that Task Config, the
    sample values, and the live endpoint in order. The Task result is normalized
    to the Agent Framework's typed episode-result contract.
    """
    sample_config = tools_kwargs.get("task") if tools_kwargs else None
    if not isinstance(sample_config, dict):
        raise ValueError("run_task requires tools_kwargs['task'] (the serialized Task Config)")

    resolver = TaskConfigResolver.from_file(task_config_path) if task_config_path else TaskConfigResolver()
    task = resolver.resolve(
        sample_config,
        runtime_model={
            "base_url": session.base_url,
            "api_key": api_key,
            "model_name": model_name,
        },
    )

    task_name = task.get("name")
    logger.info("run_task start: task=%s sample_index=%s", task_name, sample_index)

    result = await get_task(task).run()

    logger.info(
        "run_task done: task=%s reward=%s acc=%s episode_finished=%s",
        task_name,
        result.reward,
        result.accuracy,
        result.episode_finished,
    )
    metrics: dict[str, int | float | bool] = {}
    if result.accuracy is not None:
        metrics["acc"] = result.accuracy
    return EpisodeResult(
        reward=None if result.reward is None else float(result.reward),
        metrics=metrics,
        episode_finished=result.episode_finished,
        reward_context=dict(result.extra_info or {}),
    )
