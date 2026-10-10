"""Adapter completion boundaries and worker capacity configuration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from uni_agent.framework import entry as entry_module
from uni_agent.framework.base import AgentFramework

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
async def test_custom_framework_requires_explicit_admission_support():
    class CustomFramework(AgentFramework):
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

        async def generate_sequences(self, prompts):
            prompts.append("generated")

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.framework = CustomFramework()
    prompts = []
    with pytest.raises(AttributeError, match="submit_sessions"):
        await worker.submit_sessions(prompts)
    assert prompts == []  # No fallback to full generation.
    await worker.generate_sequences(prompts)
    assert prompts == ["generated"]


@pytest.mark.parametrize(
    "runner_limits, expected_concurrency",
    [([], 1000), ([4, 6], 1000), ([0, None], 1000), ([600, 700, 0, None], 1300)],
)
def test_adapter_derives_worker_concurrency_from_session_limits(monkeypatch, runner_limits, expected_concurrency):
    worker_class = Mock()
    worker_class.options.return_value = worker_class
    gateway = object()
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", worker_class)
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_: gateway)
    runners = {
        str(index): {} if limit is None else {"max_concurrent_sessions": limit}
        for index, limit in enumerate(runner_limits)
    }
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"n": 8, "custom": {"agent_framework": {"agent_runners": runners}}}}}
    )
    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
    worker_class.options.assert_called_once_with(max_concurrency=expected_concurrency)
    assert worker_class.remote.call_args.kwargs["gateway_manager"] is gateway
    assert adapter.framework_worker is worker_class.remote.return_value


@pytest.mark.parametrize(
    "method, rpc, waits",
    [
        ("generate_sequences", "generate_sequences", False),
        ("generate_sequences_and_wait", "generate_sequences", True),
        ("submit_sessions", "submit_sessions", True),
    ],
)
def test_adapter_completion_boundaries(monkeypatch, method, rpc, waits):
    worker = SimpleNamespace(generate_sequences=Mock(), submit_sessions=Mock())
    join = Mock()
    monkeypatch.setattr(entry_module.ray, "get", join)
    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = worker
    prompts = object()
    getattr(adapter, method)(prompts)
    selected, unused = (
        getattr(worker, rpc),
        getattr(worker, "submit_sessions" if rpc == "generate_sequences" else "generate_sequences"),
    )
    selected.remote.assert_called_once_with(prompts)
    unused.remote.assert_not_called()
    if waits:
        join.assert_called_once_with(selected.remote.return_value)
    else:
        join.assert_not_called()
