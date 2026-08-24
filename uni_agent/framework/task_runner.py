# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from uni_agent.tasks import TaskConfigResolver, TaskResult, get_task
from uni_agent.tasks.config import _deep_merge

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


def _rewrite_gateway_url(gateway_url: str, proxy_port: int) -> str:
    """Rewrite a gateway URL to the sandbox-internal tunnel (``127.0.0.1:<proxy_port>``).

    Replaces host:port with ``127.0.0.1:<proxy_port>`` and keeps the path, so an
    in-sandbox endpoint reaches the gateway through the reverse tunnel. Example:
    ``http://gateway.example:40169/sessions/abc/v1`` ->
    ``http://127.0.0.1:38197/sessions/abc/v1``.
    """
    return f"http://127.0.0.1:{proxy_port}{urlparse(gateway_url).path}"


def _extract_upstream(gateway_url: str) -> str | None:
    """Extract ``host:port`` from a gateway URL (the tunnel's ``upstream``).

    Returns ``None`` when the URL carries no host or port, so callers can fail
    loudly instead of forwarding a ``None:None`` upstream.
    """
    parsed = urlparse(gateway_url)
    if not parsed.hostname or not parsed.port:
        return None
    return f"{parsed.hostname}:{parsed.port}"


def _inject_gateway_tunnel(task: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Fill the runtime side of an openyuanrong gateway reverse tunnel.

    The sandbox config declares its tunnel port via ``sandbox_kwargs.proxy_port``;
    only the runtime-derived pieces are injected here: ``upstream`` (the gateway
    host:port, so the provider knows where to forward the tunnel) and the agent's
    ``model.base_url`` rewritten to the sandbox-internal tunnel address. The agent
    itself stays tunnel-agnostic -- it just sees a base_url that already points at
    ``127.0.0.1:<proxy_port>``.

    The reverse tunnel is currently supported only on the openyuanrong sandbox;
    configuring ``proxy_port`` on any other provider is rejected loudly instead of
    being silently ignored (which would leave the agent pointed at an unreachable
    ``127.0.0.1`` address).
    """
    provider = (task.get("sandbox") or {}).get("provider")
    if provider != "openyuanrong":
        raise ValueError(
            "the gateway reverse tunnel (sandbox.sandbox_kwargs.proxy_port) is currently "
            f"supported only on 'openyuanrong' sandboxes, got provider={provider!r}; "
            "switch the sandbox provider or drop proxy_port"
        )
    upstream = _extract_upstream(base_url)
    if upstream is None:
        raise ValueError(f"cannot derive gateway tunnel upstream from base_url={base_url!r}")
    proxy_port = task["sandbox"]["sandbox_kwargs"]["proxy_port"]
    return _deep_merge(
        task,
        {
            "sandbox": {"sandbox_kwargs": {"upstream": upstream}},
            "agent": {"model": {"base_url": _rewrite_gateway_url(base_url, proxy_port)}},
        },
    )


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
    / ``sample_index`` / ``tools_kwargs``). The framework's ``raw_prompt`` contains
    the authoritative dataset/source messages and overrides any serialized Task prompt.

    Run-level defaults come from the per-task-name YAML file selected by
    ``task_config_path``. ``TaskConfigResolver`` applies that Task Config, the
    sample values, and the live endpoint in order. When ``report_reward`` is set,
    the task's reward + info are POSTed back to the session's reward-info endpoint;
    the standalone evaluator reads the returned :class:`TaskResult` directly.
    """
    sample_config = tools_kwargs.get("task") if tools_kwargs else None
    if not isinstance(sample_config, dict):
        raise ValueError("run_task requires tools_kwargs['task'] (the serialized Task Config)")
    sample_config = dict(sample_config)
    sample_config["prompt"] = raw_prompt

    resolver = TaskConfigResolver.from_file(task_config_path) if task_config_path else TaskConfigResolver()
    task = resolver.resolve(
        sample_config,
        runtime_model={
            "base_url": session.base_url,
            "api_key": api_key,
            "model_name": model_name,
        },
    )

    # openyuanrong reverse tunnel: the sandbox config pins the in-sandbox tunnel
    # port (sandbox_kwargs.proxy_port); only the gateway upstream + the agent's
    # base_url rewrite are runtime-derived (session.base_url), so fill them in
    # here when a tunnel is configured. The provider check lives inside
    # _inject_gateway_tunnel (rejected loudly for non-Yuanrong sandboxes).
    tunnel_port = (task.get("sandbox") or {}).get("sandbox_kwargs", {}).get("proxy_port")
    if tunnel_port and session.base_url:
        task = _inject_gateway_tunnel(task, session.base_url)

    task_name = task.get("name")
    logger.info("run_task start: task=%s sample_index=%s", task_name, sample_index)

    task_instance = get_task(task)
    result = await task_instance.run()

    reward_posted = False
    if report_reward and session.reward_info_url:
        await _post_reward_info(session.reward_info_url, result)
        reward_posted = True
    logger.info(
        "run_task done: task=%s reward=%s acc=%s finished=%s reward_posted=%s",
        task_name,
        result.reward,
        result.accuracy,
        result.finished,
        reward_posted,
    )
    return result


async def _post_reward_info(reward_info_url: str, result: TaskResult) -> None:
    """Best-effort POST of task reward, accuracy, and Agent completion."""
    import aiohttp

    reward_info = _reward_info_from_result(result)
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(reward_info_url, json={"reward_info": reward_info}) as response:
                response.raise_for_status()
        logger.debug("posted reward_info to %s: %s", reward_info_url, reward_info)
    except Exception as exc:  # noqa: BLE001 - reward-info is best-effort telemetry
        logger.warning("failed to post reward_info to %s: %s: %s", reward_info_url, type(exc).__name__, exc)


def _reward_info_from_result(result: TaskResult) -> dict[str, Any]:
    """Build the session reward payload consumed by the trajectory framework."""
    if result.finished is not None and type(result.finished) is not bool:
        raise ValueError("TaskResult.finished must be a bool or None")
    reward_info: dict[str, Any] = {"reward": result.reward}
    if result.accuracy is not None:
        reward_info["acc"] = result.accuracy
    if result.finished is not None:
        reward_info["finished"] = result.finished
    return reward_info
