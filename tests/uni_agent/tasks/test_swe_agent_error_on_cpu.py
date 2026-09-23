from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.tasks.swe_bench import reward
from uni_agent.tasks.swe_bench.task import SWEBenchTask


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_error", [None, "", " ", "RuntimeError: upload destination must be absolute"])
async def test_swe_task_preserves_explicit_agent_error_without_changing_reward(monkeypatch, agent_error):
    sandbox = object()

    @asynccontextmanager
    async def build_sandbox():
        yield sandbox

    agent = SimpleNamespace(run=AsyncMock(return_value=AgentResult(finished=False, info={"error": agent_error})))
    task = object.__new__(SWEBenchTask)
    task.config = SimpleNamespace(
        metadata={"instance_id": "test"},
        model_dump=lambda **kwargs: {},
        run_oracle_solution=False,
        prompt=[{"role": "user", "content": "test"}],
        eval_timeout=60,
    )
    task.build_sandbox = build_sandbox
    task.build_agent = lambda: agent
    scorer_result = {"resolved": False, "eval_report": {"found_eval_status": True}}
    scorer = AsyncMock(return_value=scorer_result)
    monkeypatch.setattr(reward, "compute_reward", scorer)

    result = await task.run()

    scorer.assert_awaited_once_with(task.config.metadata, sandbox, eval_timeout=60)
    assert result.reward == result.accuracy == 0.0
    assert result.finished is False
    assert result.extra_info["eval_report"] == scorer_result["eval_report"]
    assert "agent_error" not in scorer_result
    if agent_error and agent_error.strip():
        assert result.extra_info["agent_error"] == agent_error
    else:
        assert "agent_error" not in result.extra_info
