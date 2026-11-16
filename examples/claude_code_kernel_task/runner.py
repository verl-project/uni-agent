"""Thin adapter between Uni-Agent's generic Task runner and this example.

The adapter has two intentionally small responsibilities:

* import the example-local remote Docker provider; and
* bind this session to a remote Docker host and NPU lock namespace.

Task construction remains owned by the generic runner; its TaskResult is returned
to the framework for scoring and postprocessing. The ``npu_triton_kernel`` Task
is loaded lazily from :mod:`uni_agent.tasks.kernel_bench.task`.
"""

from __future__ import annotations

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
    validation_max_turns: int = 120,
    validation_run_timeout: float = 10800,
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
    if task_config.get("metadata", {}).get("split") == "validation":
        task_config.setdefault("agent", {}).update(max_turns=validation_max_turns, run_timeout=validation_run_timeout)
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
