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

"""KvEvents mounting: the ``vllm`` backend is overridden, the name stays stock.

Two invariants: (1) the dynamic ``RolloutReplicaRegistry`` maps ``vllm`` to
KvEventsReplica after import, and (2) every *name-based* lookup that runs in
other processes keeps resolving verl's own vLLM entries — the static
``_ROLLOUT_REGISTRY`` used by ``CheckpointEngineWorker`` Ray actors (a new
registered name would crash there: driver-side registrations can't reach actor
processes), and ``ServerAdapter``'s actor-name lookup, which rebuilds the name
from ``rollout.name`` (so the replica must NOT change the name prefix).
"""

import pytest

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

vllm = pytest.importorskip("vllm")  # noqa: F841  (server import chain needs it)


def test_import_overrides_vllm_replica():
    import uni_agent.agent_aware_router.server  # noqa: F401  (import side-effect overrides)
    from uni_agent.agent_aware_router.server.replica import KvEventsReplica
    from verl.workers.rollout.replica import RolloutReplicaRegistry, get_rollout_replica_class

    assert RolloutReplicaRegistry.get("vllm") is KvEventsReplica
    assert get_rollout_replica_class("vllm") is KvEventsReplica


def test_name_based_lookups_stay_stock_vllm():
    import uni_agent.agent_aware_router.server  # noqa: F401  (override active)
    from verl.workers.rollout.base import get_rollout_class

    # Static-table lookup (CheckpointEngineWorker actors): must resolve the
    # stock ServerAdapter, not depend on any driver-side registration.
    adapter = get_rollout_class("vllm", "async")
    assert adapter.__module__.startswith("verl.workers.rollout.vllm_rollout")


def test_replica_keeps_stock_actor_name_prefix():
    from uni_agent.agent_aware_router.server.replica import KvEventsReplica
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    # ServerAdapter rebuilds actor names from rollout.name ("vllm_"); a prefix
    # override here would break its ray.get_actor lookup in colocated flows.
    assert KvEventsReplica._get_server_name_prefix is vLLMReplica._get_server_name_prefix


def test_http_server_surface():
    from uni_agent.agent_aware_router.server.http_server import KvEventsHttpServer
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    # Subclass contract: the port-allocation hooks and the kv-events getter
    # exist (get_rollout_config ships in verl's vLLMHttpServer instead).
    assert issubclass(KvEventsHttpServer, vLLMHttpServer)
    for name in (
        "get_kv_events_endpoints",
        "_preprocess_engine_kwargs",
        "_assign_kv_events_ports",
        "_release_kv_events_ports",
        "run_server",
        "run_headless",
    ):
        assert hasattr(KvEventsHttpServer, name), f"missing {name}"
