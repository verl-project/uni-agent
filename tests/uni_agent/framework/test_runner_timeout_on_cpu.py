from __future__ import annotations

import asyncio
import types

import pytest

from uni_agent.framework import framework as framework_module
from uni_agent.framework.framework import GatewayAgentFramework
from uni_agent.gateway.session.types import Trajectory
from uni_agent.tasks import TaskResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_runner_timeout_uses_ray_wait(monkeypatch):
    object_ref = object()
    wait_calls = []

    def never_ready_wait(refs, *, num_returns, timeout):
        wait_calls.append((refs, num_returns, timeout))
        assert refs == [object_ref]
        assert num_returns == 1
        assert timeout == 0.01
        return [], refs

    monkeypatch.setattr(framework_module, "ray", types.SimpleNamespace(wait=never_ready_wait))
    framework = object.__new__(GatewayAgentFramework)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await framework._await_runner_result(object_ref, "session-1", 0.01)

    asyncio.run(run())

    assert wait_calls == [([object_ref], 1, 0.01)]


def test_cancel_runner_escalates_after_ray_wait_grace_period(monkeypatch):
    object_ref = object()
    cancel_calls = []

    def never_ready_wait(refs, *, num_returns, timeout):
        assert refs == [object_ref]
        assert num_returns == 1
        assert timeout == 0.01
        return [], refs

    def cancel(ref, *, force=False):
        cancel_calls.append((ref, force))

    monkeypatch.setattr(framework_module, "ray", types.SimpleNamespace(wait=never_ready_wait, cancel=cancel))
    framework = object.__new__(GatewayAgentFramework)
    framework._RUNNER_CANCEL_GRACE_SECONDS = 0.01

    asyncio.run(framework._cancel_runner_task(object_ref, "session-1"))

    assert cancel_calls == [(object_ref, False), (object_ref, True)]


def test_reward_dataproto_supplies_default_data_source():
    data = framework_module._trajectory_to_reward_dataproto(
        Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1]),
        {},
        TaskResult(reward=1.0, accuracy=1.0, finished=True),
    )

    assert data.non_tensor_batch["data_source"].tolist() == ["uni_agent"]
    assert data.non_tensor_batch["reward_model"].tolist() == [{"ground_truth": None}]
    assert data.non_tensor_batch["extra_info"].tolist()[0]["runner_reward_info"] == {
        "reward": 1.0,
        "metrics": {"acc": 1.0},
        "reward_context": {},
    }
