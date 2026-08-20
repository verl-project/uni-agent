import pytest

from uni_agent.framework import task_runner
from uni_agent.framework.task_runner import _reward_info_from_result
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskConfig, TaskResult


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.finished is False
    assert result.extra_info == {"reason": "limit"}
    assert result.effective_messages is None


def test_task_result_effective_messages_is_trailing_positional_field():
    messages = [{"role": "user", "content": "Effective prompt"}]

    result = TaskResult(0.5, 1.0, True, None, messages)

    assert result.effective_messages == messages


def test_reward_info_omits_unknown_agent_completion():
    result = TaskResult(reward=0.5, accuracy=1.0)

    assert _reward_info_from_result(result) == {
        "reward": 0.5,
        "acc": 1.0,
    }


@pytest.mark.parametrize("finished", [True, False])
def test_reward_info_forwards_agent_completion(finished):
    result = TaskResult(reward=0.0, finished=finished)

    assert _reward_info_from_result(result) == {
        "reward": 0.0,
        "finished": finished,
    }


def test_reward_info_rejects_non_boolean_agent_completion():
    result = TaskResult(reward=0.0, finished=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finished must be a bool or None"):
        _reward_info_from_result(result)


def test_reward_info_never_includes_effective_messages():
    result = TaskResult(
        reward=1.0,
        accuracy=1.0,
        finished=True,
        effective_messages=[{"role": "user", "content": "private prompt"}],
    )

    assert _reward_info_from_result(result) == {
        "reward": 1.0,
        "acc": 1.0,
        "finished": True,
    }


@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_and_returns_effective_messages(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
  prompt_template:
    - role: system
      content: Recipe instructions
    - role: user
      content: "Issue: {prompt}"
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                prompt_template=config["prompt_template"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    result = await task_runner.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
            reward_info_url=None,
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "prompt": [{"role": "user", "content": "STALE NESTED PROMPT"}],
                "metadata": {
                    "problem_statement": "METADATA PROBLEM",
                    "patch": "SECRET PATCH",
                },
            }
        },
        task_config_path=str(config_path),
    )

    expected = [
        {"role": "system", "content": "Recipe instructions"},
        {"role": "user", "content": "Issue: Canonical source problem"},
    ]
    assert captured["config"].prompt == expected
    assert result.effective_messages == expected
    assert "STALE NESTED PROMPT" not in str(result.effective_messages)
    assert "METADATA PROBLEM" not in str(result.effective_messages)
    assert "SECRET PATCH" not in str(result.effective_messages)
