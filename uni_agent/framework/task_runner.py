# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from uni_agent.rl_insight.adapter import task_span
from uni_agent.sandbox import Sandbox
from uni_agent.sandbox.registry import get_sandbox_cls
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
    """Return the default Sandbox upstream format."""
    return Sandbox.tunnel_upstream(gateway_url)


def _inject_gateway_tunnel(task: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Fill the runtime side of a provider's gateway reverse tunnel.

    The sandbox config declares its tunnel port via ``sandbox_kwargs.proxy_port``;
    only the runtime-derived pieces are injected here: ``upstream`` (the gateway
    host:port, so the provider knows where to forward the tunnel) and the agent's
    ``model.base_url`` rewritten to the sandbox-internal tunnel address. The agent
    itself stays tunnel-agnostic -- it just sees a base_url that already points at
    ``127.0.0.1:<proxy_port>``.

    Providers declare support with ``supports_reverse_tunnel``; configuring
    ``proxy_port`` on an unsupported provider is rejected loudly instead of
    being silently ignored (which would leave the agent pointed at an unreachable
    ``127.0.0.1`` address).
    """
    provider = (task.get("sandbox") or {}).get("provider")
    sandbox_cls = get_sandbox_cls(provider) if provider else None
    if sandbox_cls is None or not sandbox_cls.supports_reverse_tunnel:
        raise ValueError(
            "the gateway reverse tunnel (sandbox.sandbox_kwargs.proxy_port) is "
            "supported only on providers declaring supports_reverse_tunnel=True, "
            f"got provider={provider!r}; "
            "switch the sandbox provider or drop proxy_port"
        )
    proxy_port = task["sandbox"]["sandbox_kwargs"]["proxy_port"]
    if isinstance(proxy_port, bool) or not isinstance(proxy_port, int) or not 2 <= proxy_port <= 65535:
        raise ValueError("gateway tunnel proxy_port must be an integer between 2 and 65535")
    upstream = sandbox_cls.tunnel_upstream(base_url)
    if upstream is None:
        raise ValueError(f"cannot derive gateway tunnel upstream from base_url={base_url!r}")
    return _deep_merge(
        task,
        {
            "sandbox": {"sandbox_kwargs": {"upstream": upstream}},
            "agent": {"model": {"base_url": _rewrite_gateway_url(base_url, proxy_port)}},
        },
    )


def score_from_runner_result(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any],
    **_reward_manager_kwargs: Any,
) -> dict[str, int | float | bool]:
    """Adapt the managed Runner result for a VERL RewardLoopWorker scorer.

    The Worker may still apply its own post-processing (for example DAPO's
    overlong penalty), so this is an adapter for the Runner payload rather than
    a bypass of the Worker.
    """
    runner_reward_info = extra_info["runner_reward_info"]
    return {**runner_reward_info["metrics"], "score": float(runner_reward_info["reward"])}


# Keep the conventional VERL callback name available for existing configs.
# New configs should prefer the descriptive ``score_from_runner_result`` name.
compute_score = score_from_runner_result


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_config_path: str | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
    **_: Any,
) -> TaskResult:
    """Resolve the sample's task, run it against ``session``, and return its result.

    Satisfies the framework's ``AgentRunner`` contract (``session`` / ``raw_prompt``
    / ``sample_index`` / ``tools_kwargs``). The framework's ``raw_prompt`` contains
    the authoritative dataset/source messages and overrides any serialized Task prompt.

    Run-level defaults come from the per-task-name YAML file selected by
    ``task_config_path``. ``TaskConfigResolver`` applies that Task Config, the
    sample values, and the live endpoint in order.
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

    # The sandbox config pins the in-sandbox reverse tunnel
    # port (sandbox_kwargs.proxy_port); only the gateway upstream + the agent's
    # base_url rewrite are runtime-derived (session.base_url), so fill them in
    # here when a tunnel is configured. The provider check lives inside
    # _inject_gateway_tunnel (rejected for providers without tunnel support).
    tunnel_port = (task.get("sandbox") or {}).get("sandbox_kwargs", {}).get("proxy_port")
    if tunnel_port is not None and session.base_url:
        task = _inject_gateway_tunnel(task, session.base_url)

    task_name = task.get("name")
    logger.info("run_task start: task=%s sample_index=%s", task_name, sample_index)

    prompt = task.get("prompt", [])
    with task_span(tools_kwargs, task_name=task_name, prompt=prompt) as span:
        task_instance = get_task(task)
        result = await task_instance.run()
        span.record_result(result, reward_posted=False)
        logger.info(
            "run_task done: task=%s reward=%s acc=%s finished=%s",
            task_name,
            result.reward,
            result.accuracy,
            result.finished,
        )
    return result
