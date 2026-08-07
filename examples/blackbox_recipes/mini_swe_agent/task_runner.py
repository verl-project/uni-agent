"""Recipe adapter from legacy OpenYuanRong rows to the unified Task Runner."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from examples.blackbox_recipes.sandbox_client import extract_upstream, rewrite_gateway_url
from uni_agent.framework.task_runner import run_task as run_unified_task
from uni_agent.gateway.session import SessionHandle

from . import akernel_sandbox as _akernel_sandbox  # noqa: F401 - registers the provider


def build_task_config(raw_prompt: Any, tools_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Return a unified Task Config from either a new or legacy dataset row."""
    tools_kwargs = tools_kwargs or {}
    task_config = tools_kwargs.get("task")
    if isinstance(task_config, dict):
        return copy.deepcopy(task_config)

    env_config = tools_kwargs.get("env") or {}
    reward_config = tools_kwargs.get("reward") or {}
    deployment = env_config.get("deployment") or {}
    image = env_config.get("image") or deployment.get("image")
    if not image:
        raise ValueError("mini-swe-agent task requires a sandbox image")
    return {
        "name": reward_config.get("name", "swe_bench"),
        "sandbox": {
            "image": image,
            "sandbox_kwargs": {"post_setup_cmd": env_config.get("post_setup_cmd", "")},
        },
        "prompt": raw_prompt,
        "metadata": reward_config.get("metadata") or {},
    }


async def run_task(
    *,
    session: SessionHandle,
    raw_prompt: Any,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    task_config_path: str | None = None,
    model_name: str = "default",
    report_reward: bool = True,
    tool_image: str | None = None,
    run_timeout: float = 7200,
    max_turns: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    **kwargs: Any,
):
    """Bind AKernel tunnel values, then delegate lifecycle ownership to ``run_task``."""
    if not session.base_url:
        raise ValueError("mini-swe-agent task runner requires session.base_url")

    task_config = build_task_config(raw_prompt, tools_kwargs)
    sandbox_config = task_config.setdefault("sandbox", {})
    sandbox_config["runtime_timeout"] = run_timeout
    sandbox_kwargs = sandbox_config.setdefault("sandbox_kwargs", {})
    sandbox_kwargs["upstream"] = extract_upstream(session.base_url)
    if tool_image:
        sandbox_kwargs["sidecar_image"] = tool_image
        sandbox_kwargs["sidecar_target"] = "/opt/mini-swe-agent-venv"

    agent_config = task_config.setdefault("agent", {})
    agent_config["step_limit"] = max_turns
    agent_config["run_timeout"] = run_timeout
    if tool_image:
        agent_config["install_command"] = "test -x /opt/mini-swe-agent-venv/bin/python"
    model_config = agent_config.setdefault("model", {})
    model_config["temperature"] = temperature
    model_config["top_p"] = top_p

    tunneled_session = replace(session, base_url=rewrite_gateway_url(session.base_url))
    return await run_unified_task(
        session=tunneled_session,
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        tools_kwargs={"task": task_config},
        task_config_path=task_config_path,
        model_name=model_name,
        report_reward=report_reward,
        **kwargs,
    )
