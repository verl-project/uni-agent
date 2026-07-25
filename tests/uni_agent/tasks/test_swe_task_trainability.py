from __future__ import annotations

import sys
from types import ModuleType

import pytest

from uni_agent.agents import AgentConfig, AgentResult
from uni_agent.tasks.swe_bench.task import SWEBenchTask, SWEBenchTaskConfig
from uni_agent.tasks.swe_rebench.task import SWEREBenchTask, SWEREBenchTaskConfig


class _FakeSandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def exec_shell(self, script: str, *, workdir: str):
        return None


class _FakeAgent:
    def __init__(self, finished: bool):
        self.finished = finished

    async def run(self, *, sandbox, messages) -> AgentResult:
        return AgentResult(finished=self.finished)


@pytest.mark.parametrize(
    ("task_cls", "config_cls", "reward_module_name"),
    [
        (SWEBenchTask, SWEBenchTaskConfig, "uni_agent.tasks.swe_bench.reward"),
        (SWEREBenchTask, SWEREBenchTaskConfig, "uni_agent.tasks.swe_rebench.reward"),
    ],
)
@pytest.mark.parametrize(
    ("mask_unfinished_trajectories", "finished", "expected"),
    [
        (False, False, True),
        (True, True, True),
        (True, False, False),
    ],
)
@pytest.mark.asyncio
async def test_swe_task_decides_trainability_from_agent_completion(
    monkeypatch,
    task_cls,
    config_cls,
    reward_module_name,
    mask_unfinished_trajectories,
    finished,
    expected,
):
    async def compute_reward(metadata, sandbox):
        return {"resolved": False}

    reward_module = ModuleType(reward_module_name)
    reward_module.compute_reward = compute_reward
    monkeypatch.setitem(sys.modules, reward_module_name, reward_module)

    config = config_cls(
        sandbox={"provider": "local"},
        agent=AgentConfig(name="test"),
        mask_unfinished_trajectories=mask_unfinished_trajectories,
    )
    task = task_cls(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: _FakeSandbox())
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent(finished))

    result = await task.run()

    assert result.trainable is expected


def test_unfinished_trajectory_masking_defaults_to_disabled():
    config = SWEBenchTaskConfig(sandbox={"provider": "local"})

    assert config.mask_unfinished_trajectories is False
