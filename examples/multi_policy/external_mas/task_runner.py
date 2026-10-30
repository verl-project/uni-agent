"""Bridge external MAS execution to Uni-Agent's generic Task runner."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator

from examples.multi_policy.framework import SessionGroupHandle
from uni_agent.agents import AgentConfig
from uni_agent.framework.task_runner import run_task
from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import Task, TaskConfig, TaskResult
from uni_agent.tasks.config import _deep_merge
from uni_agent.tasks.registry import register_task

from .external_mas_runner import (
    ExternalMASRequest,
    extract_user_task,
    run_external_mas,
)

TASK_NAME = "multi_policy_external_mas"


class ExternalMASTaskConfig(TaskConfig):
    """Command settings plus authoritative rollout bindings supplied by the runner."""

    name: str = TASK_NAME
    # The external command owns execution; no Uni-Agent sandbox is constructed.
    sandbox: SandboxConfig | None = None
    launcher_kwargs: dict[str, Any] = Field(default_factory=dict)
    session_group_id: str
    role_urls: dict[str, str]

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and not value.get("name"):
            return AgentConfig(**value)
        return super()._resolve_agent(value)


@register_task(TASK_NAME)
class ExternalMASTask(Task):
    config_model = ExternalMASTaskConfig

    def __init__(self, config: ExternalMASTaskConfig) -> None:
        super().__init__(config)
        self.config: ExternalMASTaskConfig = config

    async def run(self) -> TaskResult:
        config = self.config
        request = ExternalMASRequest(
            prompt=extract_user_task(config.prompt),
            role_urls=dict(config.role_urls),
            session_group_id=config.session_group_id,
        )
        raw_result = await run_external_mas(request=request, **config.launcher_kwargs)
        info = dict(raw_result.get("reward_info") or {})
        if raw_result.get("final_result") is not None:
            info.setdefault("final_result", raw_result["final_result"])
        reward = next(
            (info[key] for key in ("reward_score", "score", "reward") if info.get(key) is not None),
            None,
        )
        return TaskResult(
            reward=reward,
            accuracy=info.get("accuracy"),
            finished=info.get("finished", raw_result.get("finished")),
            extra_info=info,
        )


async def run_multi_agent_task(
    *,
    session: SessionGroupHandle,
    raw_prompt: object,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    task_config_path: str | None = None,
    **launcher_kwargs: Any,
) -> TaskResult:
    """Bind a multi-role rollout handle and delegate Task composition to native run_task.

    The ``SessionGroupHandle`` exposes the group ID and role endpoints. The
    Framework owns group finalization and abort; the group ID must never be
    passed to single-session lifecycle methods.
    """
    if tools_kwargs is not None and not isinstance(tools_kwargs, dict):
        raise ValueError("tools_kwargs must be a dictionary or None")
    if not isinstance(session, SessionGroupHandle) or not session.role_base_urls:
        raise ValueError("multi-agent runner requires a session handle with role_base_urls")
    copied = copy.deepcopy(tools_kwargs) if tools_kwargs is not None else {}
    sample_task = copied.get("task", {})
    if not isinstance(sample_task, dict):
        raise ValueError("tools_kwargs['task'] must be a Task configuration mapping")
    sample_task.setdefault("name", TASK_NAME)
    if sample_task["name"] != TASK_NAME:
        raise ValueError(f"External MAS runner requires task name {TASK_NAME!r}")
    if launcher_kwargs:
        sample_task["launcher_kwargs"] = _deep_merge(launcher_kwargs, sample_task.get("launcher_kwargs", {}))
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
        session=session,
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        tools_kwargs=copied,
        task_config_path=task_config_path,
    )
