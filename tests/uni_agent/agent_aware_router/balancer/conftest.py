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

"""Balancer test isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_store_singletons():
    from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
    from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
    from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore

    for cls in (PerReplicaStore, KVCacheStore, PerRequestStore):
        cls._instance = None
    yield
    for cls in (PerReplicaStore, KVCacheStore, PerRequestStore):
        cls._instance = None
