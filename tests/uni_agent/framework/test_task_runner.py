import pytest

from uni_agent.framework import task_runner
from uni_agent.framework.task_runner import (
    _extract_upstream,
    _inject_gateway_tunnel,
    _reward_info_from_result,
    _rewrite_gateway_url,
)
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskConfig, TaskResult


def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert _rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1", 38197) == (
        "http://127.0.0.1:38197/sessions/abc/v1"
    )


def test_rewrite_gateway_url_custom_proxy_port():
    assert _rewrite_gateway_url("http://gateway:8000/v1", 4242) == "http://127.0.0.1:4242/v1"


def test_extract_upstream_returns_host_port():
    assert _extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


def test_extract_upstream_none_without_port():
    assert _extract_upstream("http://gateway/v1") is None


def test_inject_gateway_tunnel_rewrites_upstream_and_base_url():
    task = {
        "sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197, "image": "x"}},
        "agent": {"step_limit": 10},
    }
    merged = _inject_gateway_tunnel(task, "http://gateway.example:40169/sessions/abc/v1")

    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "gateway.example:40169"
    assert merged["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
    # The agent receives the tunnel-rewritten base_url; unrelated keys are preserved.
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"
    assert merged["agent"]["step_limit"] == 10


def test_inject_gateway_tunnel_raises_without_port():
    task = {"sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="cannot derive gateway tunnel upstream"):
        _inject_gateway_tunnel(task, "http://gateway.example/v1")


def test_inject_gateway_tunnel_rejects_non_yuanrong_sandbox():
    task = {"sandbox": {"provider": "local", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="supported only on 'openyuanrong'"):
        _inject_gateway_tunnel(task, "http://gateway.example:40169/v1")


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.finished is False
    assert result.extra_info == {"reason": "limit"}


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


@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_to_sample_task_config(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    await task_runner.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
            reward_info_url=None,
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "metadata": {"problem_statement": "METADATA PROBLEM"},
            }
        },
        task_config_path=str(config_path),
    )

    assert captured["config"].prompt == source_prompt
