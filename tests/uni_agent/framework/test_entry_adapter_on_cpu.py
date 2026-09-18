"""CPU coverage for the Laminar worker and classic TransferQueue path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.uni_agent.support import FakeTokenizer
from uni_agent.framework import entry as entry_module
from uni_agent.framework.base import AgentFramework
from uni_agent.sandbox.registry import get_sandbox_cls
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.replica import TokenOutput


class _InProcessGatewayManager:
    def __init__(self, actor):
        self._actor = actor
        self.created_session_kwargs = []

    async def create_session(self, session_id: str, **kwargs):
        self.created_session_kwargs.append(dict(kwargs))
        return await self._actor.create_session(session_id, **kwargs)

    async def finalize_session(self, session_id: str):
        return await self._actor.finalize_session(session_id)

    async def abort_session(self, session_id: str) -> None:
        await self._actor.abort_session(session_id)


class _VersionedBackend:
    def __init__(self):
        self.calls = []

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        mm_processor_kwargs=None,
        weight_version=None,
    ) -> TokenOutput:
        self.calls.append({"request_id": request_id, "weight_version": weight_version})
        text = r"\boxed{Paris}"
        return TokenOutput(
            token_ids=[ord(char) for char in text],
            log_probs=[-0.1] * len(text),
            stop_reason="completed",
            extra_fields={
                "min_global_steps": weight_version,
                "max_global_steps": weight_version,
            },
        )


def _versioned_prompts(task_config):
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": "sample 0"}]],
            "uid": ["uid-0"],
            "data_source": ["local-echo"],
            "reward_model": [{"ground_truth": "ok"}],
            "extra_info": [{"index": 0}],
            "tools_kwargs": [{"task": task_config}],
        },
        non_tensor_dict={
            "global_steps": 7,
        },
    )


def _build_framework(gateway_manager):
    from uni_agent.framework.framework import GatewayAgentFramework

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": True,
                    "val_kwargs": {"n": 1, "temperature": 0, "top_p": 1.0, "top_k": -1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "mem_agent": {
                                    "runner_fqn": "uni_agent.framework.task_runner.run_task",
                                    "runner_kwargs": {
                                        "model_name": "test-model",
                                        "task_config_path": str(
                                            Path(__file__).resolve().parents[3] / "examples/mem_agent/task_config.yaml"
                                        ),
                                    },
                                    "dispatch_mode": "inline_async",
                                }
                            }
                        }
                    },
                }
            }
        }
    )
    return GatewayAgentFramework.from_config(config=config, gateway_manager=gateway_manager)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_framework_requires_explicit_admission_support():
    release = asyncio.Event()
    started = asyncio.Event()

    class CustomFramework(AgentFramework):
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

        async def generate_sequences(self, _prompts):
            started.set()
            await release.wait()

    worker_class = entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.framework = CustomFramework()
    with pytest.raises(AttributeError, match="submit_sessions"):
        await worker.submit_sessions(object())
    assert not started.is_set()

    task = asyncio.create_task(worker.generate_sequences(object()))
    await asyncio.wait_for(started.wait(), 2)
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, 2)
    assert task.result() is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("runner_limits", "expected_options"),
    [
        ([1536], {"max_concurrency": 1536}),
        ([600, 700], {"max_concurrency": 1300}),
        ([4, 6], {"max_concurrency": 1000}),
        ([0], {"max_concurrency": 1000}),
        ([None], {"max_concurrency": 1000}),
        ([1536, 0], {"max_concurrency": 1536}),
        ([], {"max_concurrency": 1000}),
    ],
)
def test_adapter_derives_worker_concurrency_from_session_limits(monkeypatch, runner_limits, expected_options):
    options = {}
    worker = object()
    gateway_manager = object()

    class _FrameworkWorkerClass:
        @classmethod
        def options(cls, **kwargs):
            options.update(kwargs)
            return cls

        @staticmethod
        def remote(**kwargs):
            assert kwargs["gateway_manager"] is gateway_manager
            return worker

    runners = {
        f"runner_{index}": {} if limit is None else {"max_concurrent_sessions": limit}
        for index, limit in enumerate(runner_limits)
    }
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"n": 8, "custom": {"agent_framework": {"agent_runners": runners}}}}}
    )
    monkeypatch.setattr(entry_module, "build_gateway_manager", lambda **_: gateway_manager)
    monkeypatch.setattr(entry_module, "AgentFrameworkWorker", _FrameworkWorkerClass)

    adapter = entry_module.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())

    assert options == expected_options
    assert adapter.framework_worker is worker


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "method, expected_refs",
    [
        ("generate_sequences", []),
        ("generate_sequences_and_wait", ["completion"] * 3),
        ("submit_sessions", ["admission"] * 3),
    ],
)
def test_adapter_forwards_selected_versions_and_unversioned_batches(monkeypatch, method, expected_refs):
    submitted_versions = []
    joined_refs = []

    class _RemoteGeneration:
        def __init__(self, name):
            self.name = name

        def remote(self, prompts):
            submitted_versions.append(tu.get(prompts, "global_steps"))
            return self.name

    class _FrameworkWorker:
        generate_sequences = _RemoteGeneration("completion")

    if method == "submit_sessions":
        _FrameworkWorker.submit_sessions = _RemoteGeneration("admission")

    monkeypatch.setattr(entry_module.ray, "get", joined_refs.append)
    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = _FrameworkWorker()
    generate = getattr(adapter, method)

    training_prompts = _versioned_prompts({})
    generate(training_prompts)

    validation_prompts = _versioned_prompts({})
    tu.assign_non_tensor_data(validation_prompts, "validate", True)
    generate(validation_prompts)
    generate(tu.get_tensordict(tensor_dict={"raw_prompt": [["unversioned"]]}))

    assert tu.get(training_prompts, "global_steps") == 7
    assert tu.get(validation_prompts, "global_steps") == 7
    assert submitted_versions == [7, 7, None]
    assert joined_refs == expected_refs


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_local_task_writes_versioned_classic_tq_contract(monkeypatch):
    from uni_agent.framework import framework as framework_module
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    sandbox_lifecycle = []
    batch_writes = []
    group_writes = []

    local_sandbox_cls = get_sandbox_cls("local")
    original_start = local_sandbox_cls.start
    original_stop = local_sandbox_cls.stop

    async def tracked_start(self):
        sandbox_lifecycle.append("start")
        await original_start(self)

    async def tracked_stop(self):
        sandbox_lifecycle.append("stop")
        await original_stop(self)

    monkeypatch.setattr(local_sandbox_cls, "start", tracked_start)
    monkeypatch.setattr(local_sandbox_cls, "stop", tracked_stop)

    class _FakeTransferQueue:
        async def async_kv_put(self, *, key, partition_id, tag):
            group_writes.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})

        async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
            batch_writes.append(
                {
                    "keys": list(keys),
                    "partition_id": partition_id,
                    "fields": fields,
                    "tags": [dict(item) for item in tags],
                }
            )

    monkeypatch.setattr(framework_module, "tq", _FakeTransferQueue())

    backend = _VersionedBackend()
    gateway_actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), backend)
    await gateway_actor.start()
    runtime = _InProcessGatewayManager(gateway_actor)
    task_config = {
        "name": "hotpotqa",
        "prompt": [{"role": "user", "content": "What is the capital of France?"}],
        "ground_truth": ["Paris"],
        "metadata": {"instance_id": "cpu-local", "chunks": []},
    }

    try:
        await _build_framework(runtime).generate_sequences(_versioned_prompts(task_config))
    finally:
        await gateway_actor.shutdown()

    assert sandbox_lifecycle == ["start", "stop"]
    assert runtime.created_session_kwargs[0]["weight_version"] == 7
    assert backend.calls[0]["weight_version"] == 7
    assert batch_writes[0]["keys"] == ["uid-0_0_0"]
    assert batch_writes[0]["tags"][0]["status"] == "success"
    assert tu.get(batch_writes[0]["fields"], "rm_scores")[0][-1].item() == 1.0
    assert group_writes == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "running"}},
        {
            "key": "uid-0",
            "partition_id": "train",
            "tag": {
                "status": "finished",
            },
        },
    ]
