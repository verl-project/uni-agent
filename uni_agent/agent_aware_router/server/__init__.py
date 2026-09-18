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

"""kv-events-aware vLLM rollout server for verl: overrides the ``vllm`` backend.

Importing this module replaces verl's ``vllm`` entry in ``RolloutReplicaRegistry``
with :class:`KvEventsReplica`, so every ``rollout.name: vllm`` server in the
importing process becomes a :class:`KvEventsHttpServer`. That is a behavioral
superset: with no ``kv-events-config`` in engine kwargs it behaves identically
to stock vLLM, and when ``enable_kv_cache_events`` is set it additionally
allocates per-replica port blocks (see ``http_server``) and exposes
``get_kv_events_endpoints`` for the router (``get_rollout_config`` ships in
verl itself).

``rollout.name`` stays the stock ``"vllm"`` on purpose: verl also resolves the
name through the static ``_ROLLOUT_REGISTRY`` inside ``CheckpointEngineWorker``
Ray actors (driver-side registrations can't reach those processes) and through
``ServerAdapter``'s actor-name lookup, both of which must keep hitting verl's
own vLLM entries. A *new* registered name would crash the former
("Rollout <name> with mode async not found").
"""

from verl.workers.rollout.replica import RolloutReplicaRegistry

__all__ = ["KvEventsHttpServer", "KvEventsReplica"]

# Lazy exports (PEP 562): these modules import vllm, while ``net_utils`` below
# them stays importable (and testable) without it.
_LAZY_EXPORTS = {
    "KvEventsHttpServer": "uni_agent.agent_aware_router.server.http_server",
    "KvEventsReplica": "uni_agent.agent_aware_router.server.replica",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_kv_events_replica():
    from uni_agent.agent_aware_router.server.replica import KvEventsReplica

    return KvEventsReplica


RolloutReplicaRegistry.register("vllm", _load_kv_events_replica)
