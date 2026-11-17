from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from examples.multi_policy.external_mas import task as task_mod
from examples.multi_policy.external_mas.runner import run_multi_agent_task
from examples.multi_policy.framework import SessionGroupHandle, multi_agent_runner
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskResult

TASK_NAME = "multi_policy_external_mas"


def _handle(roles=("a",), session_group_id="r1"):
    return SessionGroupHandle(
        session_group_id,
        {role: f"http://gateway/sessions/{session_group_id}:{role}/v1" for role in roles},
    )


def _task(metadata=None, **extra):
    task = {"name": TASK_NAME}
    if metadata is not None:
        task["metadata"] = metadata
    task.update(extra)
    return {"task": task}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_uses_native_task_resolution(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "answer",
            "reward_info": {"final_answer": "answer", "finished": False, "correct": 1},
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"dataset": "test", "reward_model": {"ground_truth": "answer"}})
    original = deepcopy(tools)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    session = _handle(roles=("answerer", "verifier"))
    result = await run_multi_agent_task(
        session=session,
        raw_prompt=messages,
        sample_index=3,
        tools_kwargs=tools,
        execution={"backend": "local_process"},
        command={"argv": ["python"]},
        config={"template_path": "template.yaml"},
    )
    assert isinstance(result, TaskResult)
    assert result.reward == 1.0
    assert result.accuracy == 1.0
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
async def test_runner_scores_exact_match(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "Paris",
            "reward_info": {"final_answer": "Paris"},
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"reward_model": {"ground_truth": "Paris"}})
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.extra_info["final_answer"] == "Paris"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_uses_top_level_ground_truth(monkeypatch):
    launcher = AsyncMock(return_value={"reward_info": {"final_answer": "Paris"}})
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=_task(ground_truth=["Paris"]),
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.extra_info["expected_answers"] == ["Paris"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_ignores_external_reward_field(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "Paris",
            "reward_info": {"final_answer": "Paris", "reward": 2, "score": 2},
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"reward_model": {"ground_truth": "Paris"}})
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.accuracy == 1.0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_prefers_final_answer_over_response_and_result(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "London",
            "reward_info": {
                "final_answer": "Paris",
                "final_response": "<answer>London</answer>",
                "final_result": "London",
            },
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"reward_model": {"ground_truth": "Paris"}})
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.extra_info["final_answer"] == "Paris"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_extracts_answer_from_final_response(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "<answer>Paris</answer>",
            "reward_info": {
                "final_answer": None,
                "final_response": "<answer>Paris</answer>",
            },
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"reward_model": {"ground_truth": "Paris"}})
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.extra_info["final_answer"] == "Paris"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_ignores_approved_and_does_not_substring_match(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "York",
            "reward_info": {"final_answer": "York", "approved": True},
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task({"reward_model": {"ground_truth": "New York"}})
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 0.0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_scores_clean_answers_from_ground_truth_raw(monkeypatch):
    launcher = AsyncMock(
        return_value={
            "final_result": "NBC",
            "reward_info": {"final_answer": "NBC"},
        }
    )
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task(
        {
            "reward_model": {"ground_truth": ["['NBC' 'Telemundo']"]},
            "ground_truth_raw": {"target": ["NBC", "Telemundo"]},
        }
    )
    result = await run_multi_agent_task(
        session=_handle(roles=("answerer",)),
        raw_prompt="q",
        sample_index=0,
        tools_kwargs=tools,
        execution={},
        command={},
        config={},
    )
    assert result.reward == 1.0
    assert result.accuracy == 1.0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_normalizes_string_prompt(monkeypatch):
    launcher = AsyncMock(return_value={"reward_info": {}})
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    result = await run_multi_agent_task(
        session=_handle(),
        raw_prompt="question",
        sample_index=0,
        tools_kwargs=_task(),
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
    monkeypatch.setattr(multi_agent_runner, "run_task", AsyncMock())
    with pytest.raises(ValueError, match="role_base_urls"):
        await run_multi_agent_task(
            session=SessionHandle("r", "http://gateway/v1"),
            raw_prompt="q",
            sample_index=0,
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("reward_info", [{"finished": "false"}, {"finished": 1}])
@pytest.mark.asyncio
async def test_invalid_task_result_is_rejected(monkeypatch, reward_info):
    monkeypatch.setattr(task_mod, "run_external_mas", AsyncMock(return_value={"reward_info": reward_info}))
    with pytest.raises(ValueError):
        await run_multi_agent_task(
            session=_handle(),
            raw_prompt="q",
            sample_index=0,
            tools_kwargs=_task(),
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
    monkeypatch.setattr(multi_agent_runner, "run_task", run_task)
    with pytest.raises(ValueError, match="tools_kwargs must be a dictionary or None"):
        await run_multi_agent_task(session=_handle(), raw_prompt="q", sample_index=0, tools_kwargs=tools)
    run_task.assert_not_awaited()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("tools", [None, {}, {"metadata": "preserved"}])
@pytest.mark.asyncio
async def test_runner_requires_serialized_task_config(monkeypatch, tools):
    run_task = AsyncMock(return_value=TaskResult())
    monkeypatch.setattr(multi_agent_runner, "run_task", run_task)
    with pytest.raises(ValueError, match=r"tools_kwargs\['task'\]"):
        await run_multi_agent_task(session=_handle(), raw_prompt="q", sample_index=0, tools_kwargs=tools)
    run_task.assert_not_awaited()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_runner_preserves_task_config_and_binds_session(monkeypatch):
    run_task = AsyncMock(return_value=TaskResult())
    monkeypatch.setattr(multi_agent_runner, "run_task", run_task)
    tools = _task({"dataset": "test"})
    original = deepcopy(tools)

    await run_multi_agent_task(
        session=_handle(roles=("a", "b"), session_group_id="g1"),
        raw_prompt="question",
        sample_index=0,
        tools_kwargs=tools,
        execution={"backend": "local_process"},
    )

    forwarded = run_task.call_args.kwargs["tools_kwargs"]
    task = forwarded["task"]
    assert task["name"] == TASK_NAME
    assert task["metadata"] == {"dataset": "test"}
    assert task["session_group_id"] == "g1"
    assert task["role_urls"] == _handle(roles=("a", "b"), session_group_id="g1").role_base_urls
    assert task["launcher_kwargs"]["execution"]["backend"] == "local_process"
    assert tools == original
    bound = run_task.call_args.kwargs["session"]
    assert isinstance(bound, SessionHandle)
    assert bound.session_id == "g1"
    assert bound.base_url is None


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
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    calls = []
    original_resolve = TaskConfigResolver.resolve

    def resolve(self, *args, **kwargs):
        calls.append(kwargs)
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(TaskConfigResolver, "resolve", resolve)
    tools = _task(launcher_kwargs={"command": {"env": {"SAMPLE": "value"}}})
    await run_multi_agent_task(
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
    monkeypatch.setattr(task_mod, "run_external_mas", launcher)
    tools = _task(launcher_kwargs={"command": {"env": {"SAMPLE": "yes"}}})
    original = deepcopy(tools)
    await run_multi_agent_task(
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
