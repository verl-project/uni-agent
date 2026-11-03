"""Scaffolding shared by the framework CPU test modules.

Not a test module: importing it collects no tests. It holds the fake gateway
manager, the inline-runner registry, the trajectory/prompt builders and the
select-function doubles, shared by ``test_generate_sequences_on_cpu`` and
``test_trajectory_selection_on_cpu``.

The select doubles live here rather than in the test module because the framework
loads them through ``{path, name}``: a real importable module lets a test import
the same object the framework loaded and inspect the call log.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.session import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu

_TEST_INLINE_RUNNERS = {}


async def _async_noop_runner(**kwargs):
    return None


async def _inline_runner_proxy(*, runner_key, **kwargs):
    runner = _TEST_INLINE_RUNNERS[runner_key]
    return await runner(**kwargs)


def _inline_runner_config(
    runner,
    *,
    dispatch_mode: str = "inline_async",
    trajectory_selection: object | None = None,
    extra_config: dict[str, object] | None = None,
) -> dict[str, object]:
    runner_key = f"runner-{len(_TEST_INLINE_RUNNERS)}"
    _TEST_INLINE_RUNNERS[runner_key] = runner
    config = {
        "runner_fqn": f"{__name__}._inline_runner_proxy",
        "runner_kwargs": {"runner_key": runner_key},
        "dispatch_mode": dispatch_mode,
    }
    if trajectory_selection is not None:
        config["trajectory_selection"] = trajectory_selection
    if extra_config:
        config.update(extra_config)
    return config


async def _build_framework_with_agent_runners(
    *,
    agent_runners: dict[str, dict[str, object]],
    gateway_manager,
    reward_loop_worker_handles=None,
    n: int = 1,
    val_n: int = 1,
    log_dir: str | None = None,
    mask_unfinished_episode: bool = False,
    trajectory_postprocessor_fqn: str | None = None,
    trajectory_postprocessor_kwargs: object | None = None,
    reward_config: dict[str, object] | None = None,
):
    agent_framework_cfg: dict[str, object] = {
        "agent_runners": agent_runners,
        "mask_unfinished_episode": mask_unfinished_episode,
    }
    if log_dir is not None:
        agent_framework_cfg["log_dir"] = log_dir
    if trajectory_postprocessor_fqn is not None:
        agent_framework_cfg["trajectory_postprocessor_fqn"] = trajectory_postprocessor_fqn
    if trajectory_postprocessor_kwargs is not None:
        agent_framework_cfg["trajectory_postprocessor_kwargs"] = trajectory_postprocessor_kwargs

    config_dict: dict[str, object] = {
        "actor_rollout_ref": {
            "rollout": {
                "n": n,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "calculate_log_probs": True,
                "val_kwargs": {
                    "n": val_n,
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": -1,
                },
                "custom": {"agent_framework": agent_framework_cfg},
            }
        }
    }
    if reward_config is not None:
        config_dict["reward"] = reward_config
    config = OmegaConf.create(config_dict)
    return GatewayAgentFramework.from_config(
        config=config,
        gateway_manager=gateway_manager,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


class _FakeGatewayManager:
    """Fake runtime that matches session IDs by prefix (``session-sample-{sample}-rollout-{rollout}``)
    to support the real uuid-suffixed IDs produced by the framework."""

    def __init__(self, finalized_by_session_prefix: dict[str, list[Trajectory]]):
        self._finalized_by_prefix = finalized_by_session_prefix
        self.created_sessions = []
        self.created_session_kwargs = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    def _lookup(self, session_id: str) -> list[Trajectory]:
        for prefix, trajectories in self._finalized_by_prefix.items():
            if session_id.startswith(prefix):
                return trajectories
        raise KeyError(f"No prefix match for session_id={session_id}")

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        self.created_session_kwargs.append(dict(kwargs))
        return SessionHandle(
            session_id=session_id,
            base_url=f"http://fake/{session_id}/v1",
        )

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self._lookup(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)


def _build_prompts(
    count: int = 2,
    *,
    global_steps: int | None = 7,
    validate: bool | None = None,
    do_sample: bool | None = None,
):
    non_tensor_dict = {"global_steps": global_steps}
    if validate is not None:
        non_tensor_dict["validate"] = validate
    tensor_dict = {
        "raw_prompt": [[{"role": "user", "content": f"sample {i}"}] for i in range(count)],
        "uid": [f"uid-{i}" for i in range(count)],
        "data_source": ["deepeyes"] * count,
        "reward_model": [{"ground_truth": f"answer-{i}"} for i in range(count)],
        "extra_info": [{"index": i} for i in range(count)],
        "tools_kwargs": [{"tool": i} for i in range(count)],
        "agent_name": ["deepeyes"] * count,
    }
    if do_sample is not None:
        tensor_dict["__do_sample__"] = [do_sample] * count
    return tu.get_tensordict(
        tensor_dict=tensor_dict,
        non_tensor_dict=non_tensor_dict,
    )


def _trajectory(
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    response_mask: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    finished: bool | None = None,
    reward_score: float | None = None,
    reward_metrics: dict[str, object] | None = None,
    num_turns: int = 2,
    routed_experts: object | None = None,
    extra_fields: dict[str, object] | None = None,
    chain_id: int | None = None,
):
    prompt_ids = prompt_ids or [10, 11]
    response_ids = response_ids or [20, 21]
    response_mask = response_mask if response_mask is not None else [1] * len(response_ids)
    return Trajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        finished=finished,
        reward_score=reward_score,
        reward_metrics=dict(reward_metrics or {}),
        num_turns=num_turns,
        chain_id=chain_id,
        routed_experts=routed_experts,
        multi_modal_data={"images": ["raw-image-should-not-be-written"]},
        extra_fields=dict(extra_fields or {}),
    )


#: Select-function doubles, loaded through ``{path, name}`` like user code.
#: Every call recorded as ``(session_id, [id(trajectory), ...])``.
CALLS: list[tuple[str, list[int]]] = []


def record_calls(session_id: str, trajectories: list) -> list:
    """Return every trajectory unchanged while recording the call."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return list(trajectories)


def drop_first(session_id: str, trajectories: list) -> list:
    """Keep everything but the first trajectory."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return list(trajectories[1:])


def reverse_order(session_id: str, trajectories: list) -> list:
    """Return the trajectories in reverse order."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return list(reversed(trajectories))


def drop_everything(session_id: str, trajectories: list) -> list:
    """Deliberately keep nothing so the session follows status="empty"."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return []


def drop_first_in_place(session_id: str, trajectories: list) -> list:
    """Same outcome as ``drop_first`` but mutates the caller's list."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    del trajectories[0]
    return trajectories


def deepcopy_first(session_id: str, trajectories: list) -> list:
    """Keep a deep copy of the first trajectory."""
    import copy

    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return copy.deepcopy(trajectories[:1])


def raise_boom(session_id: str, trajectories: list) -> list:
    """Fail loudly so the framework has to fall back to 'all'."""
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    raise RuntimeError("select boom")


async def async_record_calls(session_id: str, trajectories: list) -> list:
    """Async variant: awaitable results must be awaited, not used raw."""
    import asyncio

    await asyncio.sleep(0)
    CALLS.append((session_id, [id(trajectory) for trajectory in trajectories]))
    return list(trajectories)


#: Deliberately not a function, to prove the loader rejects non-callables.
NOT_CALLABLE = "not-a-function"
