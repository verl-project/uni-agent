"""Real vLLM 0.23.0 scheduler-side lifecycle; no manager/policy/spec stubs.

requirements-test.txt pins the version used by Linux CPU CI. These tests do
not launch a GPU worker or validate physical GPU/CPU transfers.
"""

import pytest

from uni_agent.gateway.agent_hint import AgentRuntimeHint
from uni_agent.gateway.kv_offload.hints import AgentHintConfig


def _hint(api, hot=True, request_id="request"):
    runtime = AgentRuntimeHint(
        tools_available=hot,
        has_active_chain=hot,
        received_tool_result=hot,
        context_tokens=40_000 if hot else 0,
        trajectory_capacity=65_536,
        remaining_capacity=20_000,
        rollback_applied=False,
        active_chain_count=1,
        trajectory_id="trajectory",
    )
    return api.context(request_id, {"agent_hint": runtime.to_dict()})


@pytest.mark.cpu
@pytest.mark.level0
def test_real_manager_admission_store_load_and_reset(api):
    manager = api.manager(
        num_blocks=2,
        agent_hint_config=AgentHintConfig(min_offload_priority=50, admit_without_hint=False),
        enable_events=True,
    )
    keys = [api.key(b"a", 0), api.key(b"b", 0)]
    context = _hint(api)
    manager.on_new_request(context)
    assert manager.lookup(keys[0], context) is False
    for rejected in [api.context("no-hint"), _hint(api, hot=False, request_id="cold")]:
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
    assert all(manager._policy.meta[key].priority == 87 for key in keys)
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
def test_expired_admission_does_not_evict_live_cache(api, monkeypatch):
    import uni_agent.gateway.kv_offload.priority_policy as policy_module

    monkeypatch.setattr(policy_module.time, "monotonic", lambda: 100)
    manager = api.manager(
        num_blocks=1, agent_hint_config=AgentHintConfig(min_offload_priority=50, admit_without_hint=False)
    )
    live_key, stale_key = api.key(b"live", 0), api.key(b"stale", 0)
    stale = _hint(api, request_id="stale")
    manager.on_new_request(stale)
    monkeypatch.setattr(policy_module.time, "monotonic", lambda: 200)
    live = _hint(api)
    manager.prepare_store([live_key], live)
    manager.complete_store([live_key], live)
    monkeypatch.setattr(policy_module.time, "monotonic", lambda: 400)
    output = manager.prepare_store([stale_key], stale)
    assert output.keys_to_store == output.evicted_keys == []
    assert manager.lookup(live_key, live) is True
    assert manager.lookup(stale_key, stale) is False


@pytest.mark.cpu
@pytest.mark.level0
def test_real_manager_preserves_native_store_threshold(api):
    manager = api.manager(num_blocks=1, store_threshold=2)
    key, context = api.key(b"a", 0), _hint(api)
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == []
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == [key]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("offload_configs", [1, 2], indirect=True)
def test_spec_constructs_and_caches_real_manager(api, offload_configs):
    config, caches = offload_configs
    spec = api.factory.create_spec(config, caches)
    factor = config.kv_transfer_config.kv_connector_extra_config["block_size"] // 16
    page_size = caches.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    assert isinstance(spec, api.spec)
    assert spec.num_blocks == 4
    assert spec.block_size_factor == factor
    assert spec.gpu_block_size == (16,)
    assert spec.hash_block_size == 16
    assert spec.kv_bytes_per_offloaded_block == spec.cpu_page_size_per_worker == page_size * factor
    assert spec._handlers is None
    assert spec.agent_hint_config.active_lease_seconds == 45.0
    assert type(spec).get_handlers is api.native_spec.get_handlers
    assert type(spec).create_handlers is api.native_spec.create_handlers
    manager = spec.get_manager()
    assert spec.get_manager() is manager
    assert isinstance(manager, api.manager)
    assert manager.store_threshold == 2
    assert manager.max_tracker_size == 16
    assert manager.min_offload_priority == 50
    assert manager.admit_without_hint is False
    assert manager.events == []
    # Exercise the manager obtained through the actual constructor chain.
    context, key = _hint(api), api.key(b"constructed", 0)
    manager.on_new_request(context)
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == []
    assert manager.lookup(key, context) is False
    assert manager.prepare_store([key], context).keys_to_store == [key]
    manager.complete_store([key], context)
    assert manager.lookup(key, context) is True
    manager.prepare_load([key], context)
    manager.complete_load([key], context)
    manager.on_request_finished(context)
    assert context.req_id not in manager._request_hints


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("target", "option", "value", "error"),
    [
        ("policy", "active_lease_seconds", float("nan"), "finite and positive"),
        ("policy", "min_offload_priority", 101, "between 0 and 100"),
        ("policy", "priority_mode", "static", "Unsupported agent_hint_config"),
        ("connector", "min_offload_priority", 50, "Put policy settings in agent_hint_config"),
    ],
)
def test_factory_rejects_invalid_policy_config(api, offload_configs, target, option, value, error):
    config, caches = offload_configs
    extra = config.kv_transfer_config.kv_connector_extra_config
    settings = extra["agent_hint_config"] if target == "policy" else extra
    settings[option] = value
    with pytest.raises(ValueError, match=error):
        api.factory.create_spec(config, caches)


@pytest.mark.cpu
@pytest.mark.level0
def test_factory_requires_native_cpu_buffer_size(api, offload_configs):
    config, caches = offload_configs
    del config.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"]
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        api.factory.create_spec(config, caches)
