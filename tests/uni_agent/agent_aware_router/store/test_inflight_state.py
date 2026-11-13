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

"""DataStore must account for facts without a contextual parser or updater."""

import struct

import pytest

from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import InflightUpdate, MetricKey
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def store(monkeypatch):
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        monkeypatch.setattr(cls, "_instance", None)
    return DataStore()


def assert_booking(store, node, count, tokens, turn):
    for key, value in [
        (MetricKey.INFLIGHT_COUNT, count),
        (MetricKey.INFLIGHT_TOKENS, tokens),
        (MetricKey.INFLIGHT_TURN_SUM, turn),
    ]:
        assert store.get_metric(node, key) == value


@pytest.mark.parametrize("rid", ["r", ""])
@pytest.mark.parametrize("target", ["s0", "s1"])
@pytest.mark.parametrize("block_size", [None, 2])
def test_store_owns_turn_replacement_and_release_amounts(store, rid, target, block_size):
    if block_size:
        store.set_block_size(block_size)
    store.requests.set_prefix_cache(rid, {"hash_strs": ["memo"], "parent_hash": 1})
    # Facts contain no prospective turn or booked length; create all of them
    # before application. Only the store knows what to unwind at each step.
    first = InflightUpdate("acquire", "s0", rid, (1, 2, 3, 4))
    replacement = InflightUpdate("acquire", target, rid, (5, 6))
    release = InflightUpdate("release", target, rid)
    store.inflight.apply(first)
    assert_booking(store, "s0", 1, 4, 1)
    result = store.inflight.apply(replacement)
    assert [(m.node_id, m.deltas[MetricKey.INFLIGHT_TURN_SUM]) for m in result.metrics] == [("s0", -1), (target, 2)]
    assert_booking(store, target, 1, 2, 2)
    if target != "s0":
        assert_booking(store, "s0", 0, 0, 0)
        assert store.inflight.apply(InflightUpdate("release", "s0", rid)).metrics == []
    assert store.requests.get_turn(rid) == 2
    assert store.requests.get_prompt_length(rid) == 2
    assert store.requests.get_prefix_cache(rid)["hash_strs"] == ["memo"]
    store.inflight.apply(release)
    assert_booking(store, target, 0, 0, 0)
    assert store.inflight_block_count(target) == store.inflight_block_count("s0") == 0
    assert store.get_marker(rid) is None
    assert store.inflight.apply(release).metrics == []
    assert store.get_metric(target, MetricKey.COMPLETED_COUNT) == 1
    assert store.inflight.apply(InflightUpdate("release", target, "unknown")).metrics == []


def test_store_resolves_using_block_size_learned_after_fact_creation(store):
    fact = InflightUpdate("acquire", "s0", "r", (1, 2, 3, 4))
    store.set_block_size(2)
    result = store.inflight.apply(fact)
    hashes, _ = get_prefix_hashes_incremental([1, 2, 3, 4], 2, 0, 0)
    assert result.blocks_recorded == 2
    assert store.get_marker("r") == {"replica": "s0", "hashes": [str(h) for h in hashes]}
    # A later memo change must not affect the acquire-time release amount.
    store.requests.book_prompt_length("r", 999)
    store.requests.book_turn("r", 99)
    store.inflight.apply(InflightUpdate("release", "s0", "r"))
    assert_booking(store, "s0", 0, 0, 0)
    assert store.inflight_block_count("s0") == 0


@pytest.mark.parametrize("existing", [False, True])
def test_failed_cross_replica_resolution_leaves_entire_old_dispatch_intact(store, existing):
    store.set_block_size(2)
    if existing:
        store.inflight.apply(InflightUpdate("acquire", "s0", "r", (1, 2, 3, 4)))
    before = store.get_metrics("s0")
    marker = store.get_marker("r")
    with pytest.raises(struct.error):
        store.inflight.apply(InflightUpdate("acquire", "s1", "r", (-1, 2)))
    assert store.get_metrics("s0") == before
    assert store.get_metrics("s1") == {}
    assert store.get_marker("r") == marker
    assert store.requests.get_turn("r") == int(existing)
    assert store.requests.get_prompt_length("r") == (4 if existing else 0)
    store.inflight.apply(InflightUpdate("release", "s0", "r"))
    assert_booking(store, "s0", 0, 0, 0)
    assert store.inflight_block_count("s0") == 0


def test_legacy_unattributed_release_remains_count_only(store):
    store.set_block_size(2)
    store.inflight.apply(InflightUpdate("acquire", "s0", prompt_ids=(1, 2)))
    store.inflight.apply(InflightUpdate("release", "s0"))
    assert_booking(store, "s0", 0, 2, 0)
    assert store.get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
    assert store.inflight_block_count("s0") == 0
