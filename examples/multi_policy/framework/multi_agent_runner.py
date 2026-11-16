"""Generic runner for multi-role session-group tasks."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from examples.multi_policy.framework.multi_agent_types import SessionGroupHandle
from uni_agent.framework.task_runner import run_task
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskResult
from uni_agent.tasks.config import _deep_merge


async def run_multi_agent_task(
    *,
    session: SessionGroupHandle,
    raw_prompt: object,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    task_config_path: str | None = None,
    **runner_kwargs: Any,
) -> TaskResult:
    """Bind a multi-role session group and delegate to the native Task runner.

    The sample must carry a serialized Task Config under
    ``tools_kwargs["task"]``, including its registry ``name``. Runner kwargs are
    merged under ``task.launcher_kwargs``; the live session supplies
    ``session_group_id`` and ``role_urls`` so serialized values cannot point at
    a stale group. The Framework owns group finalization and abort, so the
    group ID is never passed to single-session lifecycle methods.
    """
    if tools_kwargs is not None and not isinstance(tools_kwargs, dict):
        raise ValueError("tools_kwargs must be a dictionary or None")
    if not isinstance(session, SessionGroupHandle) or not session.role_base_urls:
        raise ValueError("multi-agent runner requires a session handle with role_base_urls")

    copied = copy.deepcopy(tools_kwargs) if tools_kwargs is not None else {}
    sample_task = copied.get("task")
    if not isinstance(sample_task, dict):
        raise ValueError("multi-agent runner requires tools_kwargs['task'] (the serialized Task Config)")

    if runner_kwargs:
        existing = sample_task.get("launcher_kwargs", {})
        if not isinstance(existing, dict):
            raise ValueError("tools_kwargs['task']['launcher_kwargs'] must be a mapping")
        sample_task["launcher_kwargs"] = _deep_merge(runner_kwargs, existing)
    sample_task.update(
        session_group_id=session.session_group_id,
        role_urls=dict(session.role_base_urls),
    )
    copied["task"] = sample_task

    if isinstance(raw_prompt, str):
        raw_prompt = [{"role": "user", "content": raw_prompt}]
    elif isinstance(raw_prompt, Mapping):
        raw_prompt = [dict(raw_prompt)]

    return await run_task(
        session=SessionHandle(
            session_id=session.session_group_id,
            base_url=None,
        ),
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        tools_kwargs=copied,
        task_config_path=task_config_path,
    )
