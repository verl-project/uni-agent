from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from examples.multi_policy.framework import SessionGroupHandle
from examples.multi_policy.external_mas import task_runner as mod
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskResult


def _handle(roles=("a",), session_group_id="r1"):
    return SessionGroupHandle(
        session_group_id,
        {
            role: f"http://gateway/sessions/{session_group_id}:{role}/v1"
            for role in roles
        },
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_uses_native_task_resolution(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "answer",
            "reward_info": {"reward": 2, "finished": False, "correct": 1},
        }
    )
    monkeypatch.setattr(mod, "run_external_mas", launcher)
    tools = {
        "task": {"name": "multi_policy_external_mas", "metadata": {"dataset": "test"}}
    }
    original = deepcopy(tools)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    session = _handle(roles=("answerer", "verifier"))
    result = await mod.run_multi_agent_task(
        session=session,
        raw_prompt=messages,
        sample_index=3,
        tools_kwargs=tools,
        execution={"backend": "local_process"},
        command={"argv": ["python"]},
        config={"template_path": "template.yaml"},
    )
    assert isinstance(result, TaskResult)
    assert result.reward == 2
    assert result.finished is False
    assert result.extra_info["final_result"] == "answer"
    request = launcher.call_args.kwargs["request"]
    assert request.role_urls == session.role_base_urls
    assert request.session_group_id == "r1"
    assert request.prompt == "question"
    assert tools == original
    assert messages[0]["content"] == "rules"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_normalizes_string_prompt(monkeypatch):
    launcher = AsyncMock(return_value={"reward_info": {}})
    monkeypatch.setattr(mod, "run_external_mas", launcher)
    result = await mod.run_multi_agent_task(
        session=_handle(),
        raw_prompt="question",
        sample_index=0,
        execution={},
        command={},
        config={},
    )
    assert launcher.call_args.kwargs["request"].prompt == "question"
    assert result.finished is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_requires_role_urls(monkeypatch):
    monkeypatch.setattr(mod, "run_task", AsyncMock())
    with pytest.raises(ValueError, match="role_base_urls"):
        await mod.run_multi_agent_task(
            session=SessionHandle("r", "http://gateway/v1"),
            raw_prompt="q",
            sample_index=0,
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "reward_info", [{"reward": float("nan")}, {"finished": "false"}]
)
@pytest.mark.asyncio
async def test_invalid_task_result_is_rejected(monkeypatch, reward_info):
    monkeypatch.setattr(
        mod, "run_external_mas", AsyncMock(return_value={"reward_info": reward_info})
    )
    with pytest.raises(ValueError):
        await mod.run_multi_agent_task(
            session=_handle(),
            raw_prompt="q",
            sample_index=0,
            execution={},
            command={},
            config={},
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("tools", ["invalid", [1], [], "", 0])
@pytest.mark.asyncio
async def test_runner_rejects_non_dictionary_tools_kwargs(monkeypatch, tools):
    run_task = AsyncMock()
    monkeypatch.setattr(mod, "run_task", run_task)
    with pytest.raises(ValueError, match="tools_kwargs must be a dictionary or None"):
        await mod.run_multi_agent_task(
            session=_handle(), raw_prompt="q", sample_index=0, tools_kwargs=tools
        )
    run_task.assert_not_awaited()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("tools", [None, {}, {"metadata": "preserved"}])
@pytest.mark.asyncio
async def test_runner_supplies_missing_task(monkeypatch, tools):
    run_task = AsyncMock(return_value=TaskResult())
    monkeypatch.setattr(mod, "run_task", run_task)
    original = deepcopy(tools)
    await mod.run_multi_agent_task(
        session=_handle(),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={"backend": "local_process"},
    )
    forwarded = run_task.call_args.kwargs["tools_kwargs"]
    assert forwarded["task"]["name"] == mod.TASK_NAME
    assert forwarded["task"]["role_urls"] == _handle().role_base_urls
    assert (
        forwarded["task"]["launcher_kwargs"]["execution"]["backend"] == "local_process"
    )
    assert tools == original
    if tools:
        assert forwarded["metadata"] == tools["metadata"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_task_defaults_are_composed_once_by_native_runner(monkeypatch, tmp_path):
    from uni_agent.tasks.config import TaskConfigResolver

    defaults_path = tmp_path / "task.yaml"
    defaults_path.write_text(
        "name: multi_policy_external_mas\n"
        "launcher_kwargs:\n"
        "  execution: {backend: local_process}\n"
        "  command: {argv: [python], env: {DEFAULT: yes}}\n"
        "  config: {template_path: template.yaml}\n",
        encoding="utf-8",
    )
    launcher = AsyncMock(return_value={})
    monkeypatch.setattr(mod, "run_external_mas", launcher)
    calls = []
    original_resolve = TaskConfigResolver.resolve

    def resolve(self, *args, **kwargs):
        calls.append(kwargs)
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(TaskConfigResolver, "resolve", resolve)
    tools = {"task": {"launcher_kwargs": {"command": {"env": {"SAMPLE": "value"}}}}}
    await mod.run_multi_agent_task(
        session=_handle(),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        task_config_path=str(defaults_path),
    )
    assert len(calls) == 1
    assert launcher.call_args.kwargs["command"]["argv"] == ["python"]
    assert launcher.call_args.kwargs["command"]["env"]["SAMPLE"] == "value"
    assert "task_config_path" not in launcher.call_args.kwargs


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_legacy_launcher_defaults_merge_without_mutating_samples(monkeypatch):
    launcher = AsyncMock(return_value={})
    monkeypatch.setattr(mod, "run_external_mas", launcher)
    tools = {"task": {"launcher_kwargs": {"command": {"env": {"SAMPLE": "yes"}}}}}
    original = deepcopy(tools)
    await mod.run_multi_agent_task(
        session=_handle(),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        command={"argv": ["python"], "env": {"DEFAULT": "yes"}},
        execution={},
        config={},
    )
    assert launcher.call_args.kwargs["command"] == {
        "argv": ["python"],
        "env": {"DEFAULT": "yes", "SAMPLE": "yes"},
    }
    assert tools == original
