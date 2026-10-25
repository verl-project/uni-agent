import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.framework import MultiAgentGatewayFramework
from examples.multi_policy.reward import compute_score
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils import tensordict_utils as tu


def _framework(*, workers=None, policy_processors=None):
    framework = MultiAgentGatewayFramework.__new__(MultiAgentGatewayFramework)
    framework.reward_loop_worker_handles = workers
    framework._policy_processors = policy_processors or {"p1": None, "p2": None}
    framework._mask_unfinished_episode = False
    framework._custom_reward_function_configured = False
    return framework


def _trajectories():
    return [
        Trajectory([1], [2], [1], extra_fields={"role": role, "policy_name": policy})
        for role, policy in (("searcher", "p1"), ("answerer", "p2"))
    ]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("runner_reward", [None, 0.0, 2.0])
@pytest.mark.asyncio
async def test_reward_precedence_and_runner_context(runner_reward):
    framework = _framework(workers=[SimpleNamespace()])
    framework._score_trajectories = AsyncMock(return_value=[(1.0, {"correct": 1})] * 2)
    context = {"final_result": "Paris", "agent_outputs": {"answerer": "Paris"}}
    task_result = TaskResult(reward=runner_reward, accuracy=0.5, extra_info=context)
    trajectories = _trajectories()

    annotations, source = await framework._select_reward_source(trajectories, {"uid": "u"}, task_result)

    runner_info = {
        "reward": runner_reward,
        "metrics": {"acc": 0.5},
        "reward_context": context,
    }
    for trajectory in trajectories:
        assert trajectory.extra_fields["runner_reward_info"] == runner_info
    if runner_reward is None:
        assert source == "reward_loop_worker"
        assert annotations == [(1.0, {"correct": 1})] * 2
        framework._score_trajectories.assert_awaited_once()
    else:
        assert source == "agent_runner"
        assert annotations == [(float(runner_reward), {})] * 2
        framework._score_trajectories.assert_not_awaited()
    # Context copies are independent per trajectory and from the TaskResult.
    trajectories[0].extra_fields["runner_reward_info"]["reward_context"]["agent_outputs"]["answerer"] = "changed"
    assert trajectories[1].extra_fields["runner_reward_info"]["reward_context"]["agent_outputs"]["answerer"] == "Paris"
    assert task_result.extra_info["agent_outputs"]["answerer"] == "Paris"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_reward_worker_overrides_runner_reward():
    framework = _framework(workers=[SimpleNamespace()])
    framework._custom_reward_function_configured = True
    framework._score_trajectories = AsyncMock(return_value=[(0.25, {"source": "custom"})] * 2)
    trajectories = _trajectories()

    annotations, source = await framework._select_reward_source(
        trajectories,
        {"uid": "u"},
        TaskResult(reward=1.0, accuracy=1.0, extra_info={"final_result": "ok"}),
    )

    assert annotations == [(0.25, {"source": "custom"})] * 2
    assert source == "reward_loop_worker"
    framework._score_trajectories.assert_awaited_once()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_missing_reward_without_workers_stays_unscored():
    framework = _framework(workers=None)
    annotations, source = await framework._select_reward_source(_trajectories(), {"uid": "u"}, TaskResult())
    assert annotations is None
    assert source is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("runner_reward", [None, 0.0, 2.0])
@pytest.mark.asyncio
async def test_reward_context_survives_tq_without_entering_validation_metrics(
    runner_reward,
):
    # Exercise the real verl scorer wrapper without a model or Ray actor.
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "wrong decoded answer")
    manager = NaiveRewardManager(OmegaConf.create({}), tokenizer, compute_score)
    worker_inputs = []

    async def score(data):
        worker_inputs.append(copy.deepcopy(data.non_tensor_batch["extra_info"][0]))
        return await manager.run_single(data)

    framework = _framework(workers=[SimpleNamespace(compute_score=SimpleNamespace(remote=score))])
    context = {
        "final_result": "Paris",
        "agent_outputs": {"searcher": "evidence", "answerer": "Paris"},
        "reward_extra_info": {"diagnostic": {"status": "ok"}},
    }
    task_result = TaskResult(reward=runner_reward, accuracy=0.5, extra_info=context)
    trajectories = _trajectories()
    annotations, _ = await framework._select_reward_source(
        trajectories,
        {
            "uid": "prompt-1",
            "raw_prompt": "Capital of France?",
            "data_source": "qa",
            "reward_model": {"ground_truth": "Paris"},
            "extra_info": {},
        },
        task_result,
    )
    runner_info = {
        "reward": runner_reward,
        "metrics": {"acc": 0.5},
        "reward_context": context,
    }
    if runner_reward is None:
        assert worker_inputs == [{"runner_reward_info": runner_info}]
        expected_score = 1.0
    else:
        assert worker_inputs == []  # Explicit zero is also an authoritative reward.
        expected_score = runner_reward

    annotated = [
        trajectory.__class__(
            **{
                **trajectory.__dict__,
                "reward_score": score,
                "reward_metrics": extra,
            }
        )
        for trajectory, (score, extra) in zip(trajectories, annotations, strict=False)
    ]
    assert all(t.reward_score == expected_score for t in annotated)
    session = {"uid": "prompt-1", "raw_prompt": "Capital of France?"}
    final_field, _ = framework._trajectory_to_tq_field_and_tag(
        trajectory=annotated[-1],
        sample_fields=session,
        session_index=0,
        global_steps=1,
        uid="prompt-1",
    )
    packed = tu.get(
        __import__(
            "uni_agent.framework.framework",
            fromlist=["_list_of_tq_fields_to_tensordict"],
        )._list_of_tq_fields_to_tensordict([final_field])[0],
        "extra_fields",
    )
    assert packed["runner_reward_info"] == runner_info


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("solution", "ground_truth", "score"),
    [
        ("Paris", "Paris", 1.0),
        ("London", "Paris", 0.0),
        ("answer", None, 0.0),
        ("answer", {"answers": ["Paris"]}, 0.0),
    ],
)
def test_scorer_without_runner_context_uses_solution_and_flat_metrics(solution, ground_truth, score):
    assert compute_score("qa", solution, ground_truth) == {
        "score": score,
        "acc": score,
        "num_agent_outputs": 0,
    }
