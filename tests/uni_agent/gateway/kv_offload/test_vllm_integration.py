"""Real vLLM 0.23.0 scheduler-side lifecycle; no manager/policy/spec stubs.

requirements-test.txt pins the version used by Linux CPU CI. These tests do
not launch a GPU worker or validate physical GPU/CPU transfers.
"""

import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def api():
    if sys.platform == "win32":
        pytest.skip("vLLM 0.23.0 requires Linux; exercised in the Linux CPU CI job")
    # Fail on a missing/wrong installation in CI rather than silently skipping.
    assert version("vllm") == "0.23.0"
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    from uni_agent.gateway.kv_offload.priority_policy import PriorityCPUOffloadingManager, PriorityGPUOffloadingSpec

    assert PriorityCPUOffloadingManager.__bases__ == (CPUOffloadingManager,)
    assert PriorityGPUOffloadingSpec.__bases__ == (CPUOffloadingSpec,)
    return SimpleNamespace(
        manager=PriorityCPUOffloadingManager, spec=PriorityGPUOffloadingSpec, context=ReqContext, key=make_offload_key
    )


def _hint(api, priority=90):
    return api.context("request", {"agent_hint": {"kv_priority": priority, "lease_until": 10**12}})


def test_real_manager_admission_store_load_and_reset(api):
    manager = api.manager(num_blocks=2, min_offload_priority=50, admit_without_hint=False, enable_events=True)
    keys = [api.key(b"a", 0), api.key(b"b", 0)]
    context = _hint(api)
    manager.on_new_request(context)
    assert manager.lookup(keys[0], context) is False
    for rejected in [api.context("no-hint"), _hint(api, 30)]:
        assert manager.prepare_store(keys, rejected).keys_to_store == []
        assert manager.lookup(keys[0], context) is False
    output = manager.prepare_store(keys, context)
    assert output.keys_to_store == keys
    assert output.evicted_keys == []
    assert len(set(output.store_spec.block_ids.tolist())) == 2
    assert manager.lookup(keys[0], context) is None
    manager.complete_store(keys, context)
    assert manager.lookup(keys[0], context) is True
    manager.touch(keys, context)
    assert all(manager._policy.meta[key].priority == 90 for key in keys)
    load = manager.prepare_load(keys, context)
    assert load.block_ids.tolist() == output.store_spec.block_ids.tolist()
    assert all(manager._policy.get(key).ref_cnt == 1 for key in keys)
    extra = api.key(b"c", 0)
    assert manager.prepare_store([extra], context) is None
    manager.complete_load(keys, context)
    assert all(manager._policy.get(key).ref_cnt == 0 for key in keys)
    # The original input protects b, forcing eviction of a.
    replacement = manager.prepare_store([keys[1], extra], context)
    assert replacement.keys_to_store == [extra]
    assert replacement.evicted_keys == [keys[0]]
    manager.complete_store([extra], context)
    events = list(manager.take_events())
    assert [(event.keys, event.removed) for event in events] == [(keys, False), ([keys[0]], True), ([extra], False)]
    manager.reset_cache()
    assert manager._policy.snapshot()["entries"] == 0
    assert manager.lookup(keys[1], context) is False
    assert manager.prepare_store(keys, context).keys_to_store == keys


def test_real_manager_failed_store_releases_block(api):
    manager = api.manager(num_blocks=1)
    context = _hint(api)
    key, replacement = api.key(b"a", 0), api.key(b"b", 0)
    output = manager.prepare_store([key], context)
    assert manager.lookup(key, context) is None
    manager.complete_store([key], context, success=False)
    assert manager.lookup(key, context) is False
    retry = manager.prepare_store([replacement], context)
    assert retry.store_spec.block_ids.tolist() == output.store_spec.block_ids.tolist()
    manager.complete_store([replacement], context)
    assert manager.lookup(replacement, context) is True


def test_real_manager_evicts_suffix_after_ordered_touch(api):
    manager = api.manager(num_blocks=3)
    context = _hint(api)
    keys = [api.key(bytes([index]), 0) for index in range(3)]
    manager.prepare_store(keys, context)
    manager.complete_store(keys, context)
    manager.touch(keys, context)
    extra = api.key(b"new", 0)
    output = manager.prepare_store([extra], context)
    assert output.evicted_keys == [keys[-1]]
    assert all(manager.lookup(key, context) is True for key in keys[:-1])


def test_expired_admission_does_not_evict_live_cache(api, monkeypatch):
    import uni_agent.gateway.kv_offload.priority_policy as policy_module

    monkeypatch.setattr(policy_module.time, "time", lambda: 200)
    manager = api.manager(num_blocks=1, min_offload_priority=50, admit_without_hint=False)
    live_key, stale_key = api.key(b"live", 0), api.key(b"stale", 0)
    live = _hint(api)
    manager.prepare_store([live_key], live)
    manager.complete_store([live_key], live)
    stale = api.context("stale", {"agent_hint": {"kv_priority": 100, "lease_until": 200}})
    output = manager.prepare_store([stale_key], stale)
    assert output.keys_to_store == output.evicted_keys == []
    assert manager.lookup(live_key, live) is True
    assert manager.lookup(stale_key, stale) is False


def test_real_manager_preserves_native_store_threshold(api):
    manager = api.manager(num_blocks=1, store_threshold=2)
    key, context = api.key(b"a", 0), _hint(api)
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == []
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == [key]


def test_spec_constructs_and_caches_real_manager(api):
    # Bypass GPU sizing only; exercise our factory against the real constructor.
    spec = api.spec.__new__(api.spec)
    spec._manager = None
    spec.num_blocks = 2
    spec.eviction_policy = "lru"
    spec.extra_config = {"store_threshold": 2, "max_tracker_size": 16}
    spec.min_offload_priority = 50
    spec.admit_without_hint = False
    spec.vllm_config = SimpleNamespace(kv_events_config=SimpleNamespace(enable_kv_cache_events=True))
    manager = spec.get_manager()
    assert spec.get_manager() is manager
    assert isinstance(manager, api.manager)
    assert manager.store_threshold == 2
    assert manager.max_tracker_size == 16
    assert manager.min_offload_priority == 50
    assert manager.admit_without_hint is False
    assert manager.events == []
