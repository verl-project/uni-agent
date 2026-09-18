"""Real Ray scheduling and session limits without model or TransferQueue setup."""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

import verl
from uni_agent.framework import entry
from uni_agent.framework import framework as framework_module
from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig
from verl.utils import tensordict_utils as tu


@pytest.mark.cpu
@pytest.mark.level0
def test_worker_starts_over_1000_batches_with_runner_limits(monkeypatch):
    def source_paths():
        return (sys.executable, entry.__file__, inspect.getfile(GatewayAgentFramework), verl.__file__)

    class ControlledFramework(GatewayAgentFramework):
        def __init__(self, runners):
            super().__init__(
                None,
                runner_registry={name: _RunnerConfig.from_config(name, cfg) for name, cfg in runners.items()},
                rollout_config=OmegaConf.create(
                    {"n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "calculate_log_probs": False}
                ),
            )
            self.release = asyncio.Event()
            self.state = {"batches_started": 0, "episodes_completed": 0}
            for name in runners:
                self.state.update({f"active/{name}": 0, f"peak/{name}": 0})

        async def generate_sequences(self, prompts):
            self.state["batches_started"] += 1
            await super().generate_sequences(
                tu.get_tensordict(
                    tensor_dict={"uid": [str(index) for index in range(len(prompts))], "agent_name": prompts},
                    non_tensor_dict={"global_steps": 0},
                )
            )

        async def _collect_prompt_rollouts(self, *, tasks, **_kwargs):
            # Replace output persistence only; use the production session scheduler.
            await asyncio.gather(*tasks)
            return {
                "num_success_sessions": len(tasks),
                "num_failed_sessions": 0,
                "num_success_outputs": len(tasks),
                "num_unfinished_episodes": 0,
                "num_failed_uids": 0,
                "failure_reasons": [],
            }

        async def _run_agent_episode(self, *, runner_name, sample_fields, **kwargs):
            active, peak = f"active/{runner_name}", f"peak/{runner_name}"
            self.state[active] += 1
            self.state[peak] = max(self.state[peak], self.state[active])
            try:
                await self.release.wait()
                self.state["episodes_completed"] += 1
                return [], sample_fields
            finally:
                self.state[active] -= 1

    class ControlledWorker(entry.AgentFrameworkWorker.__ray_metadata__.modified_class):
        def __init__(self, *, config, gateway_manager, reward_loop_worker_handles=None):
            async def write_status(**_kwargs):
                pass

            assert source_paths() == gateway_manager, "Ray worker must use the driver Python and product sources"
            framework_module.tq = SimpleNamespace(async_kv_put=write_status)
            self.framework = ControlledFramework(config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners)

        @ray.method(concurrency_group="control")
        def release(self):
            self.framework._semaphore_loop.call_soon_threadsafe(self.framework.release.set)

        @ray.method(concurrency_group="control")
        def read_state(self):
            return dict(self.framework.state)

    runners = {
        name: {"runner_fqn": "unused.runner", "dispatch_mode": "ray_task", "max_concurrent_sessions": 501}
        for name in ("a", "b")
    }
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": {"agent_runners": runners}}}}}
    )
    monkeypatch.setattr(entry, "build_gateway_manager", lambda **_: source_paths())
    # Test-only controls observe and release batches while all generation slots are occupied.
    monkeypatch.setattr(entry, "AgentFrameworkWorker", ray.remote(concurrency_groups={"control": 1})(ControlledWorker))

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_cpus=1, include_dashboard=False)
    worker = None
    try:
        worker = entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=None).framework_worker
        batches = [worker.generate_sequences.remote([name, name]) for _ in range(501) for name in runners]
        deadline = time.monotonic() + 60
        while True:
            state = ray.get(worker.read_state.remote(), timeout=10)
            if state["batches_started"] == 1002 and all(state[f"active/{name}"] == 501 for name in runners):
                break
            assert time.monotonic() < deadline, state
            time.sleep(0.02)
        assert not ray.wait(batches, num_returns=1, timeout=0)[0]
        assert state["episodes_completed"] == 0
        assert all(state[f"peak/{name}"] == 501 for name in runners)

        ray.get(worker.release.remote(), timeout=10)
        assert ray.get(batches, timeout=30) == [None] * 1002
        state = ray.get(worker.read_state.remote(), timeout=10)
        assert state["episodes_completed"] == 2004
        assert all(state[f"peak/{name}"] == 501 and state[f"active/{name}"] == 0 for name in runners)
    finally:
        if worker is not None:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
def test_ray_admission_rpc_keeps_rollouts_alive_and_backpressures_next_batch():
    class Gateway:
        async def create_session(self, session_id, **kwargs):
            return None

        async def finalize_session(self, session_id):
            return []

    class ControlledFramework(GatewayAgentFramework):
        def __init__(self):
            runner_config = _RunnerConfig("unused.runner", {}, "ray_task", 1)
            super().__init__(
                Gateway(),
                runner_registry={"runner": runner_config},
                rollout_config=OmegaConf.create(
                    {"n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "calculate_log_probs": False}
                ),
            )
            runner_config.dispatch_mode = "inline_async"
            self._inline_runners["runner"] = self.run
            self.releases = {}
            self.started = []
            self.completed = []

        async def run(self, *, raw_prompt, **_kwargs):
            release = self.releases[raw_prompt] = asyncio.Event()
            self.started.append(raw_prompt)
            await release.wait()
            self.completed.append(raw_prompt)

        async def _collect_prompt_rollouts(self, *, tasks, **_kwargs):
            # Replace output persistence only; admission and lifetime use the real scheduler.
            await asyncio.gather(*tasks)
            return {
                "num_success_sessions": len(tasks),
                "num_failed_sessions": 0,
                "num_success_outputs": len(tasks),
                "num_unfinished_episodes": 0,
                "num_failed_uids": 0,
                "failure_reasons": [],
            }

    class ControlledWorker(entry.AgentFrameworkWorker.__ray_metadata__.modified_class):
        def __init__(self, expected_sources):
            async def write_status(**_kwargs):
                pass

            assert (sys.executable, entry.__file__, verl.__file__) == expected_sources
            framework_module.tq = SimpleNamespace(async_kv_put=write_status)
            self.framework = ControlledFramework()

        async def release(self, name):
            self.framework.releases[name].set()

        async def read_state(self):
            return self.framework.started, self.framework.completed, len(self.framework._rollout_tasks)

    def prompts(name):
        return tu.get_tensordict(
            tensor_dict={"uid": [name], "raw_prompt": [name]},
            non_tensor_dict={"global_steps": 0},
        )

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_cpus=1, include_dashboard=False)
    worker = None
    try:
        worker = ray.remote(ControlledWorker).remote((sys.executable, entry.__file__, verl.__file__))
        assert ray.get(worker.submit_sessions.remote(prompts("first")), timeout=30) is None
        assert ray.get(worker.read_state.remote(), timeout=10) == (["first"], [], 1)

        second = worker.submit_sessions.remote(prompts("second"))
        assert not ray.wait([second], timeout=0.1)[0]
        assert ray.get(worker.read_state.remote(), timeout=10) == (["first"], [], 1)
        ray.get(worker.release.remote("first"), timeout=10)
        assert ray.get(second, timeout=10) is None
        assert ray.get(worker.read_state.remote(), timeout=10)[:2] == (["first", "second"], ["first"])

        ray.get(worker.release.remote("second"), timeout=10)
        complete = worker.generate_sequences.remote(prompts("standalone"))
        assert not ray.wait([complete], timeout=0.1)[0]
        assert ray.get(worker.read_state.remote(), timeout=10)[:2] == (
            ["first", "second", "standalone"],
            ["first", "second"],
        )
        ray.get(worker.release.remote("standalone"), timeout=10)
        assert ray.get(complete, timeout=10) is None
        assert ray.get(worker.read_state.remote(), timeout=10) == (
            ["first", "second", "standalone"],
            ["first", "second", "standalone"],
            0,
        )
    finally:
        if worker is not None:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()
