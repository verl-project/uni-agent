"""Thin adapter between Uni-Agent's generic Task runner and this example.

The adapter has two intentionally small responsibilities:

* import the example-local remote Docker provider; and
* bind this session to a remote Docker host and NPU lock namespace.

Task construction remains owned by the generic runner; its TaskResult is returned
to the framework for scoring and postprocessing. The
``npu_triton_kernel`` Task is loaded lazily from
:mod:`uni_agent.tasks.kernel_bench.task`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from uni_agent.framework.task_runner import run_task

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

from .remote_docker import bind_remote_sandbox, parse_device_ids


async def run_triton_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    remote_docker_hosts: str,
    evaluator_npu_device_ids: str,
    evaluator_npu_lock_dir: str = "/var/lock/triton-agent-npu",
    evaluator_npu_lock_timeout: float = 1200.0,
    max_response_length: int | None = None,
    claude_run_timeout: float | None = None,
    **kwargs: Any,
):
    """Run one Triton task through Uni-Agent's generic Task runner."""

    devices = parse_device_ids(evaluator_npu_device_ids)
    copied = bind_remote_sandbox(
        tools_kwargs or {},
        hosts=remote_docker_hosts,
        session_id=session.session_id,
        devices=devices,
        lock_dir=evaluator_npu_lock_dir,
        lock_timeout=evaluator_npu_lock_timeout,
    )
    task_config = copied.get("task")
    if not isinstance(task_config, dict):
        raise ValueError("run_triton_task requires tools_kwargs['task']")
    if claude_run_timeout is not None:
        if not math.isfinite(claude_run_timeout) or claude_run_timeout <= 0:
            raise ValueError("claude_run_timeout must be finite and positive")
        # Run-level override wins over values serialized in prepared samples.
        task_config.setdefault("agent", {})["run_timeout"] = claude_run_timeout
    if max_response_length is not None:
        extra_env = task_config.setdefault("agent", {}).setdefault("extra_env", {})
        extra_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_response_length)
    metadata = task_config.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("tools_kwargs['task']['metadata'] must be a mapping")
    runtime = metadata.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError("tools_kwargs['task']['metadata']['runtime'] must be a mapping")
    runtime.update({"session_id": session.session_id, "sample_index": sample_index})

    runtime["evaluator_npu_device_count"] = len(devices)

    return await run_task(
        session=session,
        tools_kwargs=copied,
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        **kwargs,
    )
