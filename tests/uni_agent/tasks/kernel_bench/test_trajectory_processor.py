"""Recipe selection and R3 slicing, rather than Framework output validation."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.claude_code_kernel_task.trajectory_processor import process_trajectories
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def trajectory(**kwargs):
    return Trajectory(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22, 23, 24],
        response_mask=[1, 1, 0, 1, 1],
        response_logprobs=[-1.0, -2.0, 0.0, -3.0, -4.0],
        reward_score=0.3,
        **kwargs,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_default_best_crops_partial_credit_and_absolute_routes(backend):
    routes = np.arange(7 * 4, dtype=np.int16).reshape(7, 2, 2)
    if backend == "torch":
        routes = torch.from_numpy(routes)
    source = trajectory(routed_experts=routes)
    result = TaskResult(extra_info={"train_best": {"assistant_index": 0}, "metrics": {"correctness_ok": False}})

    selected = process_trajectories((source,), task_result=result)[0]

    assert selected.response_ids == [20, 21]
    assert selected.response_mask == [1, 1]
    assert selected.response_logprobs == [-1.0, -2.0]
    assert selected.routed_experts.tolist() == routes[:4].tolist()  # 2 prompt + 2 response tokens
    selected.routed_experts[0, 0, 0] = -1
    assert routes[0, 0, 0] == 0
    assert source.response_ids == [20, 21, 22, 23, 24]


@pytest.mark.parametrize("hint", [None, {"assistant_index": 99}, {"assistant_index": -1}])
def test_missing_or_invalid_hint_falls_back_to_last_assistant(hint):
    source = replace(trajectory(), response_mask=[1, 1, 0, 1, 0])
    info = {} if hint is None else {"train_best": hint}

    selected = process_trajectories((source,), task_result=TaskResult(extra_info=info))

    assert selected[0].response_ids == [20, 21, 22, 23]
    assert selected[0].extra_fields["trajectory_postprocess_reason"] == "all_final"


def test_multiple_chains_do_not_apply_a_scalar_best_hint():
    sources = (trajectory(chain_id=1), trajectory(chain_id=2))
    result = TaskResult(extra_info={"train_best": {"assistant_index": 0}})

    selected = process_trajectories(sources, task_result=result)

    assert [item.chain_id for item in selected] == [1, 2]
    assert all(item.response_ids == [20, 21, 22, 23, 24] for item in selected)


@pytest.mark.parametrize("selection,chains", [("all_final", [1, 2]), ("final", [2])])
def test_explicit_selection(selection, chains):
    sources = (trajectory(chain_id=1), trajectory(chain_id=2))
    selected = process_trajectories(sources, task_result=TaskResult(), selection=selection)
    assert [item.chain_id for item in selected] == chains


def test_empty_and_context_only_trajectories_produce_no_prefix():
    result = TaskResult()
    assert process_trajectories((), task_result=result) == []
    assert process_trajectories((replace(trajectory(), response_mask=[0] * 5),), task_result=result) == []


def test_token_misalignment_raises_instead_of_filtering():
    with pytest.raises(ValueError, match="response_ids/response_mask misaligned"):
        process_trajectories((replace(trajectory(), response_mask=[1]),), task_result=TaskResult())


def test_missing_implementation_keeps_zero_reward_trajectory():
    source = replace(trajectory(), reward_score=0.0)
    result = process_trajectories(
        (source,), task_result=TaskResult(extra_info={"metrics": {"error_type": "missing_impl"}})
    )
    assert len(result) == 1
    assert result[0].response_ids == source.response_ids
    assert result[0].reward_score == 0.0
