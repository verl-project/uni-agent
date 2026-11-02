"""Real Ray scheduling and session limits without model or TransferQueue setup."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

import verl
from uni_agent.framework import entry
from uni_agent.framework import framework as framework_module
from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult
from verl.utils import tensordict_utils as tu


@pytest.mark.cpu
@pytest.mark.level0
def test_ray_admission_rpc_keeps_rollouts_alive_and_backpressures_next_batch():
    class Gateway:
        async def create_session(self, session_id, **kwargs):
            return None

        async def finalize_session(self, session_id):
            return [Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1])]

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
            return TaskResult(reward=1.0)

        async def _write_session_trajectories_to_tq(self, **_kwargs):
            # Replace only persistence; prompt admission and ownership stay real.
            pass

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
        # Both owners exist already; the second waits for a permit without an episode.
        assert ray.get(worker.read_state.remote(), timeout=10) == (["first"], [], 2)
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
