"""SWE scoring must distinguish a retired workspace from an unfinished agent."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.mini_swe_agent.agent import MiniSweAgentAgent, MiniSweAgentConfig
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.tasks.swe_bench.task import SWEBenchTask
from uni_agent.tasks.swe_bench_multilingual.task import SWEBenchMultilingualTask
from uni_agent.tasks.swe_rebench.task import SWEREBenchTask


@pytest.fixture(params=[SWEBenchTask, SWEREBenchTask, SWEBenchMultilingualTask])
def episode(request, monkeypatch):
    task_cls = request.param
    task = task_cls(
        task_cls.config_model(
            sandbox={"provider": "docker"},
            metadata={"base_commit": "test-base"},
            prompt=[{"role": "user", "content": "Fix it"}],
        )
    )
    sandbox = DockerSandbox()
    agent = MiniSweAgentAgent(
        MiniSweAgentConfig(
            model=ModelConfig(base_url="http://unused.invalid"),
            tool_python="python",
            run_agent_script="/mini_swe_entrypoint.py",
        )
    )
    scoring = AsyncMock(return_value={"resolved": True, "eval_completed": True})
    # No evaluator is needed: these tests assert whether the Task invokes it.
    reward_module = task_cls.__module__.rsplit(".", 1)[0] + ".reward"
    monkeypatch.setitem(sys.modules, reward_module, SimpleNamespace(compute_reward=scoring))
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    monkeypatch.setattr(task, "build_agent", lambda: agent)
    return task, sandbox, scoring


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("outcome", ["timeout", "nonzero", "success"])
def test_swe_scores_only_when_agent_workspace_was_not_retired(episode, monkeypatch, outcome):
    task, sandbox, scoring = episode
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "container":
            return ExecResult(0, "owned-id\n", "")
        if args[0] == "exec" and "/mini_swe_entrypoint.py" in args[-1]:
            if outcome == "timeout":
                raise asyncio.TimeoutError
            if outcome == "nonzero":
                return ExecResult(1, "", "agent failed")
            return ExecResult(0, '{"exit_status": "Submitted"}', "")
        return ExecResult(0, "", "")

    monkeypatch.setattr(sandbox, "_run_docker", docker)
    result = asyncio.run(task.run())
    assert ("rm", "-f", "owned-id") in calls
    assert sandbox._container_name is None
    assert sandbox._cleanup_label is None
    if outcome == "timeout":
        scoring.assert_not_awaited()
        assert result.reward == result.accuracy == 0.0
        assert result.finished is False
        assert result.extra_info["resolved"] is False
        assert result.extra_info["eval_completed"] is False
        assert result.extra_info["eval_skipped_reason"] == "sandbox_retired_after_timeout"
    else:
        scoring.assert_awaited_once()
        assert result.reward == result.accuracy == 1.0
        assert result.finished is (outcome == "success")
        assert "eval_skipped_reason" not in result.extra_info


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("failure", ["cleanup", "cancel", "scoring"])
def test_swe_does_not_convert_infrastructure_errors_or_cancellation_to_zero(episode, monkeypatch, failure):
    task, sandbox, scoring = episode

    async def docker(*args, **kwargs):
        if args[0] == "container":
            return ExecResult(0, "owned-id\n", "")
        if args[0] == "rm" and failure == "cleanup":
            return ExecResult(1, "", "daemon unavailable")
        if args[0] == "exec" and "/mini_swe_entrypoint.py" in args[-1]:
            if failure == "cleanup":
                raise asyncio.TimeoutError
            if failure == "cancel":
                raise asyncio.CancelledError
        return ExecResult(0, "", "")

    monkeypatch.setattr(sandbox, "_run_docker", docker)
    if failure == "scoring":
        scoring.side_effect = RuntimeError("scoring failed")
    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        asyncio.run(task.run())
    if failure != "scoring":
        scoring.assert_not_awaited()


@pytest.mark.cpu
@pytest.mark.level0
def test_swe_still_scores_nonretiring_provider_after_timeout(episode, monkeypatch):
    from uni_agent.sandbox.local import LocalSandbox

    task, _, scoring = episode
    sandbox = LocalSandbox()

    async def execute(argv, **kwargs):
        if "/mini_swe_entrypoint.py" in argv[-1]:
            raise asyncio.TimeoutError
        return ExecResult(0, "", "")

    monkeypatch.setattr(sandbox, "_exec", execute)
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    result = asyncio.run(task.run())
    assert not sandbox.retired_after_timeout
    assert result.finished is False
    assert result.reward == 1.0
    scoring.assert_awaited_once()
