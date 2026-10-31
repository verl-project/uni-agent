"""Owned dispatch preserves native resumption and remote cancellation semantics."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

from uni_agent.integrations.verl_v1.client import (
    CancellableFullyAsyncLLMServerClient,
    CancellableLLMServerClient,
    GenerationCancellationError,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput


def config(name="vllm"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"name": name, "response_length": 6, "full_determinism": True}},
            "async_training": {"partial_rollout": True},
        }
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_native_resume_uses_owned_dispatch_and_keeps_response_budget(monkeypatch):
    calls, released = [], []

    async def generate(**kwargs):
        calls.append(deepcopy(kwargs))
        n = min(2, kwargs["sampling_params"]["max_tokens"])
        return TokenOutput(
            token_ids=[7] * n,
            log_probs=[-0.1] * n,
            stop_reason="aborted" if len(calls) < 3 else "length",
            extra_fields={"global_steps": len(calls)},
        )

    server = SimpleNamespace(generate=SimpleNamespace(remote=lambda **kwargs: asyncio.create_task(generate(**kwargs))))
    client = CancellableFullyAsyncLLMServerClient(config())

    async def acquire(*args, **kwargs):
        return "server", server

    client._acquire_server = acquire
    client._release_server = lambda *args, **kwargs: released.append(1)
    assert client.generate.__func__ is FullyAsyncLLMServerClient.generate
    output = await client.generate("resume", prompt_ids=[1], sampling_params={})
    assert len(output.token_ids) == len(output.log_probs) == 6
    assert [call["sampling_params"]["max_tokens"] for call in calls] == [6, 4, 2]
    assert len(released) == 3
    assert output.extra_fields["min_global_steps"] == 1
    assert output.extra_fields["max_global_steps"] == 3


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_cancel_timeout_does_not_release_an_unconfirmed_reservation(monkeypatch):
    entered = asyncio.Event()
    remote = asyncio.Future()
    releases, cancellations = [], []
    client = CancellableLLMServerClient(config(), cancellation_timeout=0.01)

    def submit(**kwargs):
        entered.set()
        return remote

    async def acquire(*args, **kwargs):
        return "server", SimpleNamespace(generate=SimpleNamespace(remote=submit))

    client._acquire_server = acquire
    client._release_server = lambda *args, **kwargs: releases.append(1)
    monkeypatch.setattr(ray, "cancel", lambda ref: cancellations.append(ref))

    def no_ack(*args, **kwargs):
        raise ray.exceptions.GetTimeoutError("No remote acknowledgement")

    monkeypatch.setattr(ray, "get", no_ack)
    task = asyncio.create_task(client.generate("timeout", prompt_ids=[1], sampling_params={}))
    await entered.wait()
    task.cancel()
    with pytest.raises(GenerationCancellationError):
        await task
    assert cancellations == [remote]
    assert not releases


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_cancelled_acquire_releases_the_late_allocation_once():
    allocated, deliver = asyncio.Event(), asyncio.Event()
    releases = []
    client = CancellableLLMServerClient(config())

    async def acquire(*args, **kwargs):
        allocated.set()
        await deliver.wait()
        return "server", object()

    client._acquire_server = acquire
    client._release_server = lambda *args, **kwargs: releases.append((args, kwargs))
    task = asyncio.create_task(client.generate("late", prompt_ids=[1], sampling_params={}))
    await allocated.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    deliver.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert releases == [(("server",), {"request_id": "late"})]


@ray.remote
class BlockingServer:
    def __init__(self):
        self.active, self.cancelled = set(), set()

    async def generate(self, request_id, **kwargs):
        self.active.add(request_id)
        try:
            await asyncio.Event().wait()
        finally:
            self.active.discard(request_id)
            self.cancelled.add(request_id)

    async def state(self):
        return sorted(self.active), sorted(self.cancelled)


@pytest.mark.cpu
@pytest.mark.level0
def test_real_ray_cancellation_stops_only_the_target_request():
    async def run():
        server = BlockingServer.remote()
        client = CancellableLLMServerClient(config())

        async def acquire(*args, **kwargs):
            return "server", server

        client._acquire_server = acquire
        client._release_server = lambda *args, **kwargs: None
        tasks = [
            asyncio.create_task(client.generate(name, prompt_ids=[1], sampling_params={}))
            for name in ("cancelled", "retained")
        ]

        async def ready():
            while (await server.state.remote())[0] != ["cancelled", "retained"]:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(ready(), 30)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(tasks[0], 10)
            assert await server.state.remote() == (["retained"], ["cancelled"])
            assert not tasks[1].done()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            ray.kill(server)

    owned = not ray.is_initialized()
    if owned:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=128 * 1024**2)
    try:
        asyncio.run(run())
    finally:
        if owned:
            ray.shutdown()
