# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uni_agent.tasks import TaskResult, get_task, resolve_task_config
from uni_agent.tasks.config import deep_merge, route_task_config

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_defaults: dict[str, Any] | None = None,
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

    The run-wide task base may be supplied inline (``task_defaults``) and/or from a
    per-task-name YAML file (``task_config_path``): the row's task name selects the
    matching entry (:func:`route_task_config`), inline defaults are merged onto that
    entry, and the row's sample config wins on top. When ``report_reward`` is set, the
    task's reward + info are POSTed back to the session's reward-info endpoint so a
    training reward manager can pick them up; the standalone evaluator reads the
    returned :class:`TaskResult` directly and leaves this off.
    """
    if task_config_path:
        row_task = tools_kwargs.get("task") if tools_kwargs else None
        task_name = row_task.get("name") if isinstance(row_task, dict) else None
        routed = route_task_config(str(task_config_path), task_name)
        task_defaults = deep_merge(routed, task_defaults or {})

    task = resolve_task_config(
        tools_kwargs,
        session_base_url=session.base_url,
        task_defaults=task_defaults,
        api_key=api_key,
        model_name=model_name,
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
