import pytest

from uni_agent.tasks import TaskResult


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, 1.0), ("0.5", 0.5)],
    ids=["boolean", "numeric-string"],
)
def test_task_result_normalizes_numeric_fields(value, expected):
    result = TaskResult(reward=value, accuracy=value)  # type: ignore[arg-type]

    assert result.reward == expected
    assert result.accuracy == expected


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("kwargs", "error_message"),
    [
        ({"reward": "not-a-number"}, r"TaskResult\.reward must be a finite number or numeric string"),
        ({"accuracy": float("inf")}, r"TaskResult\.accuracy must be a finite number or numeric string"),
        ({"finished": 1}, r"TaskResult\.finished must be a bool or None"),
        ({"extra_info": None}, r"TaskResult\.extra_info must be a dict"),
    ],
    ids=["reward", "accuracy", "finished", "extra-info"],
)
def test_task_result_rejects_invalid_fields(kwargs, error_message):
    with pytest.raises(ValueError, match=error_message):
        TaskResult(**kwargs)


@pytest.mark.cpu
@pytest.mark.level0
def test_task_result_represents_missing_reward_explicitly():
    result = TaskResult()

    assert result.reward is None
    assert result.extra_info == {}
