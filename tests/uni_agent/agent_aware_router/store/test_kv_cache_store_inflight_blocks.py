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

"""CPU unit tests for the in-flight held-block ref-count table (``pin``/``unpin``)."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types import Layer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

PREFIX = ["h1", "h2", "h3", "h4", "h5"]


def _store() -> KVCacheStore:
    """A fresh non-singleton store."""
    return KVCacheStore()


def test_held_set_ref_counts_shared_blocks() -> None:
    """Shared blocks count once and leave only with their last holder's unpin; hits are held like misses."""
    store = _store()
    store.add_blocks("s0", PREFIX[:2], layer=Layer.GPU)  # h1/h2 already resident

    store.pin_inflight_blocks("s0", PREFIX)  # r1
    store.pin_inflight_blocks("s0", PREFIX[:3])  # r2 shares r1's prefix
    assert store.inflight_block_count("s0") == 5  # not 8: shared count once, hits held too

    store.unpin_inflight_blocks("s0", PREFIX)  # r1 (superset) leaves
    assert store.inflight_block_count("s0") == 3  # r2 still holds its subset

    store.unpin_inflight_blocks("s0", PREFIX[:3])  # r2 leaves
    assert store.inflight_block_count("s0") == 0


def test_pin_table_edges_and_cleanup() -> None:
    """Pins are per replica; unknown hashes are ignored; clear_replica keeps pins."""
    store = _store()

    store.pin_inflight_blocks("s0", [])  # unresolvable prompt pins nothing
    assert store.inflight_block_count("s0") == 0

    store.pin_inflight_blocks("s0", PREFIX)
    store.pin_inflight_blocks("s1", PREFIX)  # the same hash is still new on s1
    assert store.inflight_block_count("s0") == 5
    assert store.inflight_block_count("s1") == 5

    store.unpin_inflight_blocks("s0", ["never-pinned", "h1", "h1"])  # unknown + doubled unpin
    assert store.inflight_block_count("s0") == 4  # h1 really released, never negative

    store.unpin_inflight_blocks("ghost-replica", PREFIX)  # replica with no pin table
    assert store.inflight_block_count("ghost-replica") == 0

    store.clear_replica("s0")
    assert store.inflight_block_count("s0") == 4  # a cache flush must not drop live reservations
    assert store.inflight_block_count("s1") == 5  # clear_replica must not touch s1
