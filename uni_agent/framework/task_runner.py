# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=8)
def load_task_config_file(path: str) -> dict[str, dict[str, Any]]:
    """Load a per-task-name config file into a ``{name: task_config}`` index.

    Backs the training recipe's ``runner_kwargs.task_config_path`` -- the same
    file-path idea as the legacy ``agent_loop_config_path``, so the run-wide task
    bases live in one YAML (the inference ``task_config.yaml`` shape) instead of
    dozens of Hydra overrides. The file is a list of task configs each keyed by
    ``name``; :func:`route_task_config` picks the one whose name matches a row's task,
    so a mixed train(swe_rebench)/test(swe_bench) run routes each row to its own base.
    Cached per path (loaded once in the rollout worker).
    """
    import yaml

    raw = yaml.safe_load(open(path))
    entries = raw if isinstance(raw, list) else [raw]
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"task_config_path {path!r}: each entry must be a mapping with a 'name' (got {entry!r})")
        index[str(entry["name"])] = entry
    return index


def route_task_config(path: str, task_name: str | None) -> dict[str, Any]:
    """Return the task config whose ``name`` matches ``task_name`` (route by task name)."""
    index = load_task_config_file(path)
    if task_name is None or task_name not in index:
        raise ValueError(f"task_config_path {path!r} has no config for task name {task_name!r} (have {sorted(index)})")
    return index[task_name]


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_overrides: dict[str, Any] | None = None,
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

    The run-wide task base may be supplied inline (``task_overrides``) and/or from a
    per-task-name YAML file (``task_config_path``): the row's task name selects the
    matching entry (:func:`route_task_config`), then inline ``task_overrides`` win on
    top. When ``report_reward`` is set, the task's reward + info are POSTed back to the
    session's reward-info endpoint so a training reward manager can pick them up; the
    standalone evaluator reads the returned :class:`TaskResult` directly and leaves
    this off.
    """
    if task_config_path:
        row_task = tools_kwargs.get("task") if tools_kwargs else None
        task_name = row_task.get("name") if isinstance(row_task, dict) else None
        routed = route_task_config(str(task_config_path), task_name)
        task_overrides = deep_merge(routed, task_overrides or {})

    task = resolve_task_config(
        tools_kwargs,
        session_base_url=session.base_url,
        task_overrides=task_overrides,
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
