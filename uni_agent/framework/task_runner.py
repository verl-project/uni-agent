# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks.

The framework layer runs *agent runners* against per-session OpenAI-compatible
gateway endpoints (see :mod:`uni_agent.framework.framework`); it has no notion of
a uni_agent :class:`~uni_agent.tasks.Task`. This runner closes that gap -- it is
the "framework meets task" glue:

1. resolve the task config carried on the sample's ``tools_kwargs["task"]``
   (the same serialized form the dataset stores under
   ``extra_info.tools_kwargs.task``),
2. deep-merge any run-wide ``task_overrides`` (agent / sandbox / sampling ...),
3. point the agent's policy endpoint (``agent.model.base_url``) at the gateway
   session so every ``/v1/chat/completions`` call is captured as a trajectory,
4. run the task and return its :class:`~uni_agent.tasks.TaskResult`.

Wire it into a framework recipe via ``runner_fqn`` so training rollouts run
uni_agent tasks::

    actor_rollout_ref.rollout.custom.agent_framework.agent_runners:
      swe_bench:
        runner_fqn: uni_agent.framework.task_runner.run_task
        dispatch_mode: inline_async

It is also driven directly by the standalone evaluator
``examples/agent_interaction/parallel_infer_verl.py``, which creates a session
per sample and calls this runner to reproduce ``parallel_infer_api.py`` semantics
over a verl-launched inference engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from uni_agent.tasks import TaskResult, get_task

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` on top of ``base``, returning a new dict.

    Nested dicts merge key-wise (``overrides`` wins); lists and scalars replace
    wholesale. ``base`` is never mutated. (Same semantics as the agent-loop's
    and ``parallel_infer.py``'s.)
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_task_config(
    tools_kwargs: dict[str, Any] | None,
    *,
    session_base_url: str | None,
    task_overrides: dict[str, Any] | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build the concrete task-config dict for one sample.

    Layers, in order: the sample's own task (``tools_kwargs["task"]``), the
    run-wide ``task_overrides`` (agent / sandbox / sampling ...), and finally the
    endpoint -- ``agent.model.{base_url,api_key,model_name}`` -- which is runtime
    state bound to the live gateway session rather than dataset content.
    """
    if not tools_kwargs or "task" not in tools_kwargs:
        raise ValueError("run_task requires tools_kwargs['task'] (the serialized task config)")

    task = deep_merge(tools_kwargs["task"], task_overrides or {})

    model_cfg: dict[str, Any] = {"base_url": session_base_url, "api_key": api_key}
    if model_name is not None:
        model_cfg["model_name"] = model_name
    return deep_merge(task, {"agent": {"model": model_cfg}})


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_overrides: dict[str, Any] | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
    report_reward: bool = False,
    **_: Any,
) -> TaskResult:
    """Resolve the sample's task, run it against ``session``, and return its result.

    Satisfies the framework's ``AgentRunner`` contract (``session`` / ``raw_prompt``
    / ``sample_index`` / ``tools_kwargs``). ``raw_prompt`` is accepted for protocol
    parity but unused: a uni_agent task carries its own prompt on the task config.

    When ``report_reward`` is set, the task's reward + info are POSTed back to the
    session's reward-info endpoint so a training reward manager can pick them up;
    the standalone evaluator reads the returned :class:`TaskResult` directly and
    leaves this off.
    """
    task = resolve_task_config(
        tools_kwargs,
        session_base_url=session.base_url,
        task_overrides=task_overrides,
        api_key=api_key,
        model_name=model_name,
    )

    result = await get_task(task).run()

    if report_reward and session.reward_info_url:
        await _post_reward_info(session.reward_info_url, result)

    return result


async def _post_reward_info(reward_info_url: str, result: TaskResult) -> None:
    """Best-effort POST of the task reward + info to the gateway session.

    Failures are logged rather than raised: reward-info is auxiliary metadata for
    the training reward path and must not fail an otherwise-successful episode.
    """
    import aiohttp

    reward_info: dict[str, Any] = {"reward": result.reward, **(result.info or {})}
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(reward_info_url, json={"reward_info": reward_info}) as response:
                response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - reward-info is best-effort telemetry
        logger.warning("failed to post reward_info to %s: %s: %s", reward_info_url, type(exc).__name__, exc)
