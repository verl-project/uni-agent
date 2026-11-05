# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registration ownership and engine metadata survive pool changes and failures."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver, notify_lifecycle
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState
from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

from ._helpers import _FakeCollectorManager, _router_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def local_ray(monkeypatch):
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, **kwargs: ref)
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.util.get_node_ip_address", lambda: "127.0.0.1")


def test_replaced_registration_rejects_unknown_old_instance_over_real_control_transport(local_ray):
    sinks = []

    def handle(state):
        def register(sink):
            sinks.append(dict(sink))
            return {"state": state.to_dict(), "sources": {}}

        return SimpleNamespace(
            register_cache_lifecycle_sink=SimpleNamespace(remote=register),
            unregister_cache_lifecycle_sink=SimpleNamespace(remote=Mock(return_value=None)),
        )

    old = handle(CacheLifecycleState("old-engine"))
    paused = CacheLifecycleState("replacement", serving=False)
    replacement = handle(paused)
    router = KVCAwareBalancer({"a": old}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        obsolete_sink = sinks[0]
        router.add_servers({"a": replacement})
        delayed = CacheLifecycleState("old-engine-restarted-before-removal")
        reply = asyncio.run(notify_lifecycle(obsolete_sink, delayed.to_dict(), timeout=1))
        assert not reply["applied"]
        assert router._store.cache_lifecycle_state("a") == paused
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("must-stay-paused")
        resumed = CacheLifecycleState("replacement", revision=1)
        assert asyncio.run(notify_lifecycle(sinks[-1], resumed.to_dict(), timeout=1))["applied"]
        assert router.acquire_server("resumed")[1] is replacement
    finally:
        router.close()


def test_delayed_unregister_cannot_detach_readded_registration(local_ray):
    from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle

    control, cleanups = CacheLifecycle(), []

    def unregister(token, *, generation):
        cleanups.append((token, generation))

    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=lambda sink: {"state": control.bind(sink), "sources": {}}),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=unregister),
    )
    router = KVCAwareBalancer({"a": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.remove_servers(["a"])
        control.state = CacheLifecycleState(control.state.instance_id, revision=1, serving=False)
        router.add_servers({"a": handle})
        token, obsolete_generation = cleanups[0]
        current_sink = dict(control.sinks[token])
        assert current_sink["generation"] != obsolete_generation
        asyncio.run(control.unbind(token, generation=obsolete_generation))
        assert control.sinks[token] == current_sink
        asyncio.run(control.finish(True))
        assert router.acquire_server("new-registration-still-live")[1] is handle
    finally:
        router.close()


def test_repeated_registration_keeps_current_inflight_notification(local_ray):
    ready, paused = CacheLifecycleState("engine"), CacheLifecycleState("engine", revision=1, serving=False)
    sinks = []

    def register(sink):
        if sinks:
            # Publication captured the descriptor before bind() takes its RPC
            # snapshot. Re-registering the same handle must keep it valid.
            assert sink == sinks[0]
            reply = asyncio.run(notify_lifecycle(sinks[0], paused.to_dict(), timeout=1))
            assert not reply["applied"] and reply["reason"] == "lifecycle_handshake_pending"
        sinks.append(sink)
        return {"state": ready.to_dict(), "sources": {}}

    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=register),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
    )
    router = KVCAwareBalancer({"a": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.add_servers({"a": handle})
        assert router._store.cache_lifecycle_state("a") == paused
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("must-stay-paused")
    finally:
        router.close()


@pytest.mark.parametrize("metadata_source", ["notification", "handshake", "new-instance"])
def test_registration_keeps_instance_metadata_when_newer_pause_has_none(local_ray, layout, metadata_source):
    raw = layout((("sliding_window", 4), ("full_attention", 8)), unit=None)
    original = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=lambda: {**raw, "profile": "original"}))
    ready, paused = CacheLifecycleState("new-engine"), CacheLifecycleState("new-engine", revision=1, serving=False)
    if metadata_source == "new-instance":
        paused = CacheLifecycleState("engine-after-snapshot", revision=2, serving=False)
    metadata = {**raw, "profile": "fresh"}
    router = KVCAwareBalancer({"a": original}, _router_config(), provider_factory=_FakeCollectorManager)

    def factory(value, unit):
        profile = value.get("profile")
        return SimpleNamespace(unit=unit, root=profile.encode(), profile_key=profile) if profile else None

    def register(sink):
        if metadata_source == "notification":
            reply = asyncio.run(notify_lifecycle(sink, ready.to_dict(), cache_metadata=metadata))
            assert not reply["applied"] and reply["reason"] == "lifecycle_handshake_pending"
        reply = asyncio.run(
            notify_lifecycle(
                sink,
                paused.to_dict(),
                cache_metadata=metadata if metadata_source == "new-instance" else None,
            )
        )
        assert not reply["applied"] and reply["reason"] == "lifecycle_handshake_pending"
        if metadata_source == "new-instance":
            obsolete = CacheLifecycleState("unknown-earlier-engine", revision=1)
            reply = asyncio.run(notify_lifecycle(sink, obsolete.to_dict(), cache_metadata={**raw, "profile": "stale"}))
            assert not reply["applied"] and reply["reason"] == "lifecycle_handshake_pending"
        response = {"state": ready.to_dict(), "sources": {}}
        if metadata_source == "handshake":
            response["cache_metadata"] = metadata
        return response

    added = SimpleNamespace(
        get_kv_cache_metadata=SimpleNamespace(remote=lambda: {**raw, "profile": "pre-handshake"}),
        register_cache_lifecycle_sink=SimpleNamespace(remote=register),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
    )
    try:
        router._store.set_native_hash_codec_factory(factory)
        router.add_servers({"b": added})
        assert router._store.cache_lifecycle_state("b") == paused
        assert router._store._kv._native_by_node["b"].codec.profile_key == "fresh"
        assert router._store._kv._native_by_node["a"].codec.profile_key == "original"
        if metadata_source == "new-instance":
            sink = router._lifecycle_receiver.descriptor("b")
            reply = asyncio.run(
                notify_lifecycle(sink, CacheLifecycleState("another-earlier-engine", revision=1).to_dict())
            )
            assert not reply["applied"]
            assert router._store.cache_lifecycle_state("b") == paused
    finally:
        router.close()


@pytest.mark.parametrize("failure", ["discovery", "factory", "start"])
def test_failed_construction_detaches_sink_and_stops_created_resources(monkeypatch, local_ray, failure):
    receivers, providers = [], []
    unregister = Mock(return_value=None)
    state = CacheLifecycleState("engine")

    def receiver(*args, **kwargs):
        created = LifecycleControlReceiver(*args, **kwargs)
        receivers.append(created)
        return created

    class Provider(_FakeCollectorManager):
        def __init__(self, *args, **kwargs):
            if failure == "factory":
                raise ValueError("provider factory failed")
            super().__init__(*args, **kwargs)
            providers.append(self)

        def start(self):
            super().start()
            raise ValueError("provider start failed")

    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=lambda sink: {"state": state.to_dict(), "sources": {}}),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=unregister),
    )
    if failure == "discovery":
        handle.get_server_address = SimpleNamespace(remote=Mock(side_effect=ValueError("discovery failed")))
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.LifecycleControlReceiver", receiver)
    try:
        with pytest.raises(ValueError, match="failed"):
            KVCAwareBalancer({"a": handle}, _router_config(), provider_factory=Provider)
        unregister.assert_called_once_with(receivers[0].token, generation=ANY)
        assert not receivers[0]._thread.is_alive()
        assert all(provider.stopped for provider in providers)
    finally:
        for created in receivers:
            created.close()
        for provider in providers:
            provider.stop()


@pytest.mark.parametrize("delivery", ["notification", "handshake"])
def test_same_handle_new_engine_refreshes_native_profile(monkeypatch, local_ray, layout, stored, parse, delivery):
    vllm = pytest.importorskip("vllm")
    import vllm.envs as envs

    from uni_agent.agent_aware_router.collectors.parse.vllm.native_hash import VLLMHashCodec

    raw = layout((("sliding_window", 4), ("full_attention", 8)), unit=None, window=9)

    def metadata(root):
        return {
            **raw,
            "native_hash_config": {
                "schema_version": 1,
                "vllm_version": vllm.__version__,
                "algorithm": "sha256",
                "root_hash_hex": root * 32,
                "unit": 4,
                "pickle_protocol": pickle.HIGHEST_PROTOCOL,
                "int_hashes": envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES,
                "lookahead_tokens": 0,
            },
        }

    first, second = metadata("11"), metadata("22")
    current, state, sinks = first, CacheLifecycleState("engine-1"), []

    def register(sink):
        sinks.append(sink)
        return {"state": state.to_dict(), "sources": {}, "cache_metadata": current}

    handle = SimpleNamespace(
        get_kv_cache_metadata=SimpleNamespace(remote=lambda: current),
        register_cache_lifecycle_sink=SimpleNamespace(remote=register),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
    )
    router = KVCAwareBalancer({"a": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        previous = router._store._kv._native_by_node["a"]
        current, state = second, CacheLifecycleState("engine-2", revision=1)
        if delivery == "notification":
            response = asyncio.run(notify_lifecycle(sinks[-1], state.to_dict(), cache_metadata=current))
            assert response["applied"]
        else:
            router.add_servers({"a": handle})
        codec = VLLMHashCodec.from_metadata(current, 4)
        parent, digests = codec.root, []
        for start in range(0, 16, 4):
            parent = codec.hash_block(parent, list(range(start, start + 4)))
            digests.append(parent)
        coarse = codec.coarse_keys(digests, 8)
        events = [
            stored(1, "full_attention", 8, hashes=list(coarse)),
            stored(0, "sliding_window", 4, hashes=[codec.external_key(digest) for digest in digests[1:]]),
        ]
        assert router._store.apply_kv_update(KVCacheUpdate("a", parse(events), lifecycle=state))
        hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(17)), 4, 0, 0)[0]]
        assert router._store.estimate_prefixes(["a"], hashes, 17)["a"].estimated_hit_tokens == 16
        current_index = router._store._kv._native_by_node["a"]
        assert current_index is not previous and current_index.enabled
    finally:
        router.close()
