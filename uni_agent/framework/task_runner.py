# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uni_agent.tasks import TaskConfigResolver, TaskResult, get_task

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
    report_reward: bool = False,
    **_: Any,
) -> TaskResult:
    """Resolve the sample's task, run it against ``session``, and return its result.

    Satisfies the framework's ``AgentRunner`` contract (``session`` / ``raw_prompt``
    / ``sample_index`` / ``tools_kwargs``). ``raw_prompt`` is accepted for protocol
    parity but unused: a uni_agent task carries its own prompt on the task config.

    Run-level defaults come from the per-task-name YAML file selected by
    ``task_config_path``. ``TaskConfigResolver`` applies that Task Config, the
    sample values, and the live endpoint in order. When ``report_reward`` is set,
    the task's reward + info are POSTed back to the session's reward-info endpoint;
    the standalone evaluator reads the returned :class:`TaskResult` directly.
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

    reward_posted = False
    if report_reward and session.reward_info_url:
        await _post_reward_info(session.reward_info_url, result)
        reward_posted = True
    logger.info(
        "run_task done: task=%s reward=%s acc=%s reward_posted=%s",
        task_name,
        result.reward,
        result.accuracy,
        reward_posted,
    )
    return result


async def _post_reward_info(reward_info_url: str, result: TaskResult) -> None:
    """Best-effort POST of the task reward + accuracy to the gateway session."""
    import aiohttp

    reward_info: dict[str, Any] = {"reward": result.reward}
    if result.accuracy is not None:
        reward_info["acc"] = result.accuracy
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(reward_info_url, json={"reward_info": reward_info}) as response:
                response.raise_for_status()
        logger.debug("posted reward_info to %s: %s", reward_info_url, reward_info)
    except Exception as exc:  # noqa: BLE001 - reward-info is best-effort telemetry
        logger.warning("failed to post reward_info to %s: %s: %s", reward_info_url, type(exc).__name__, exc)
