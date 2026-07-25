import pytest

from uni_agent.framework.task_runner import _reward_info_from_result
from uni_agent.tasks import TaskResult


def test_reward_info_includes_default_trainability():
    result = TaskResult(reward=0.5, accuracy=1.0)

    assert _reward_info_from_result(result) == {
        "reward": 0.5,
        "trainable": True,
        "acc": 1.0,
    }


def test_reward_info_forwards_untrainable_trajectory_tag():
    result = TaskResult(reward=0.0, trainable=False)

    assert _reward_info_from_result(result) == {
        "reward": 0.0,
        "trainable": False,
    }


def test_reward_info_rejects_non_boolean_trainability():
    result = TaskResult(reward=0.0, trainable=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="trainable must be a bool"):
        _reward_info_from_result(result)
