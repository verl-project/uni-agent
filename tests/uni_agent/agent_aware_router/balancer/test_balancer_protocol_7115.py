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

"""verl #7115 protocol conformance — field declarations, release bookkeeping,
``clear_sticky_cache`` / ``get_total_inflight``.

Upstream #7115 changed the plugin contract: ``acquire_server`` only receives
the kwargs declared via ``require_acquire_fields`` (so the balancer must
declare ``prompt_ids``), ``release_server`` matches the Protocol signature
``(server_id, request_id)`` — the in-flight token gauge is balanced by the
inflight collector from acquire-time per-request bookkeeping — and trainers
call ``clear_sticky_cache`` / ``get_total_inflight`` on the actor.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.ut, pytest.mark.cpu, pytest.mark.level0]


class TestFieldDeclarations:
    """verl serializes only the declared fields into the acquire/release RPCs."""

    def test_require_acquire_fields_declares_prompt_ids(self):
        balancer = _make_balancer()
        assert balancer.require_acquire_fields() == ["prompt_ids"]

    def test_require_release_fields_declares_request_id(self):
        balancer = _make_balancer()
        assert balancer.require_release_fields() == ["request_id"]


class TestReleaseTokenBalance:
    """#7115 releases carry no token list — the collector's bookkeeping balances it."""

    def test_token_gauge_stays_balanced(self):
        balancer = _make_balancer({"s0": "h0"})
        sid, _ = balancer.acquire_server("r1", [1, 2, 3])
        balancer.release_server(sid, request_id="r1")
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0

    def test_bookkeeping_refreshes_each_turn(self):
        # Turn 2's prompt is longer; release must subtract turn 2's length, not
        # turn 1's (acquire overwrites the bookkeeping row).
        balancer = _make_balancer({"s0": "h0"})
        sid, _ = balancer.acquire_server("r1", [1, 2, 3])
        balancer.release_server(sid, request_id="r1")
        sid, _ = balancer.acquire_server("r1", [1, 2, 3, 4, 5, 6])
        balancer.release_server(sid, request_id="r1")
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0

    def test_release_without_any_identity_leaves_token_gauge_unchanged(self):
        # Degraded path: no request_id → nothing to look up; the token gauge
        # keeps acquire's contribution (documented behavior).
        balancer = _make_balancer({"s0": "h0"})
        sid, _ = balancer.acquire_server("r1", [1, 2, 3])
        balancer.release_server(sid)
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 3
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0


class TestClearStickyCache:
    """Trainer-facing sticky reset (phase switches / fully-async rebalancing)."""

    def test_clears_bindings_and_reports_loads(self):
        balancer = _make_balancer({"s0": "h0"})
        sid, _ = balancer.acquire_server("r1", [1])
        assert balancer._store.get_sticky_binding("r1") == sid
        result = balancer.clear_sticky_cache()
        assert result["cleared_entries"] == 1
        assert result["server_loads"] == {"s0": 1}
        assert balancer._store.get_sticky_binding("r1") is None

    def test_empty_cache_reports_zero(self):
        balancer = _make_balancer()
        assert balancer.clear_sticky_cache()["cleared_entries"] == 0

    def test_prefix_hash_checkpoint_survives_clear(self):
        # Design decision: hash checkpoints are replica-agnostic (shared across
        # replicas, append-only) — clearing sticky must NOT drop them.
        balancer = _make_balancer({"s0": "h0"})
        balancer._store.set_block_size(16)
        balancer.acquire_server("r1", list(range(32)))
        assert balancer._store.get_per_request("r1", "prefix_hashes") is not None
        balancer.clear_sticky_cache()
        assert balancer._store.get_per_request("r1", "prefix_hashes") is not None
        assert balancer._store.get_sticky_binding("r1") is None


class TestGetTotalInflight:
    """Balancer-side in-flight counters (drain polling)."""

    def test_tracks_acquire_release(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})
        assert balancer.get_total_inflight() == 0
        sid, _ = balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        assert balancer.get_total_inflight() == 2
        balancer.release_server(sid, request_id="r1")
        assert balancer.get_total_inflight() == 1

    def test_unknown_server_release_does_not_go_negative(self):
        balancer = _make_balancer({"s0": "h0"})
        balancer.acquire_server("r1", [1])
        balancer.release_server("nope", request_id="r1")
        assert balancer.get_total_inflight() == 1

    def test_remove_servers_drops_counter_row(self):
        balancer = _make_balancer({"s0": "h0"})
        balancer.acquire_server("r1", [1])
        balancer.remove_servers(["s0"])
        assert balancer.get_total_inflight() == 0
        # A late release for an in-flight request on the removed server is
        # tolerated (floor guard), never negative.
        balancer.release_server("s0", request_id="r1")
        assert balancer.get_total_inflight() == 0

    def test_get_status_exposes_total_inflight(self):
        balancer = _make_balancer({"s0": "h0"})
        balancer.acquire_server("r1", [1])
        assert balancer.get_status()["total_inflight"] == 1
