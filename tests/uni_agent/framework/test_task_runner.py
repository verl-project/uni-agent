import pytest

from uni_agent.framework import EpisodeResult
from uni_agent.framework import task_runner as task_runner_module
from uni_agent.framework.task_runner import run_task
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskResult


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.episode_finished is False
    assert result.extra_info == {"reason": "limit"}


def test_episode_result_separates_reward_status_metrics_and_context():
    result = EpisodeResult(
        reward=0.5,
        metrics={"acc": 1.0, "steps": 4},
        episode_finished=False,
        reward_context={"report": {"resolved": 1}},
    )

    assert result.reward == 0.5
    assert result.metrics == {"acc": 1.0, "steps": 4}
    assert result.episode_finished is False
    assert result.reward_context == {"report": {"resolved": 1}}


def test_task_result_uses_episode_finished_name():
    result = TaskResult(reward=0.5, accuracy=1.0, episode_finished=False)

    assert result.episode_finished is False
    assert not hasattr(result, "finished")


def test_episode_result_rejects_non_scalar_metrics():
    with pytest.raises(ValueError, match=r"metrics\['report'\] must be scalar"):
        EpisodeResult(metrics={"report": {"resolved": 1}})


def test_episode_result_rejects_non_numeric_reward():
    with pytest.raises(ValueError, match="reward must be a number or None"):
        EpisodeResult(reward="1.0")  # type: ignore[arg-type]


def test_episode_result_rejects_reserved_reward_metric():
    with pytest.raises(ValueError, match="key 'reward' is reserved"):
        EpisodeResult(metrics={"reward": 0.5})


@pytest.mark.asyncio
async def test_run_task_returns_episode_result(monkeypatch):
    task_result = TaskResult(
        reward=0.5,
        accuracy=1.0,
        episode_finished=False,
        extra_info={"report": {"resolved": 1}},
    )

    class _Resolver:
        def resolve(self, sample_config, runtime_model):
            assert sample_config == {"name": "stub"}
            assert runtime_model["base_url"] == "http://gateway/session/v1"
            return {"name": "stub"}

    class _Task:
        async def run(self):
            return task_result

    monkeypatch.setattr(task_runner_module, "TaskConfigResolver", _Resolver)
    monkeypatch.setattr(task_runner_module, "get_task", lambda task: _Task())

    result = await run_task(
        session=SessionHandle(session_id="session", base_url="http://gateway/session/v1"),
        tools_kwargs={"task": {"name": "stub"}},
    )

    assert result == EpisodeResult(
        reward=0.5,
        metrics={"acc": 1.0},
        episode_finished=False,
        reward_context={"report": {"resolved": 1}},
    )
