from __future__ import annotations

import asyncio

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.tasks.swe_bench.task import SWEBenchTask, SWEBenchTaskConfig


class _FakeSandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _UnfinishedAgent:
    async def run(self, *, sandbox, messages, workdir=None):
        return AgentResult(finished=False)


@pytest.mark.cpu
@pytest.mark.level0
def test_swe_bench_does_not_reward_an_unfinished_agent(monkeypatch):
    eval_calls = []

    async def fake_compute_reward(*args, **kwargs):
        eval_calls.append((args, kwargs))
        return {"resolved": True}

    monkeypatch.setattr("uni_agent.tasks.swe_bench.reward.compute_reward", fake_compute_reward)

    task = SWEBenchTask(
        SWEBenchTaskConfig(
            name="swe_bench",
            sandbox={"provider": "local"},
            agent={"name": "codex", "tool_script": "/opt/codex/bin/run_agent.sh"},
            prompt=[{"role": "user", "content": "fix"}],
            metadata={"instance_id": "example", "base_commit": "deadbeef"},
        )
    )
    task.build_sandbox = lambda: _FakeSandbox()
    task.build_agent = lambda: _UnfinishedAgent()

    result = asyncio.run(task.run())

    assert eval_calls == []
    assert result.reward == 0.0
    assert result.accuracy == 0.0
    assert result.finished is False
    assert result.extra_info == {"resolved": False, "eval_skipped": "agent_unfinished"}
