"""Tests for StickySessionTable — request_id → replica_id LRU mapping.

Covers the sticky-session affinity table the Balancer owns and threads into
``route()`` → ``strategy.score()``. Mirrors verl ``GlobalRequestLoadBalancer``
sticky semantics: access refreshes recency, LRU evicts cold entries, replica
removal bulk-clears stale bindings.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.strategies.sticky_session import (
    DEFAULT_STICKY_MAX_SIZE,
    StickySessionTable,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestStickySessionTable:
    """S01-Snn: StickySessionTable construction + access semantics."""

    def test_s01_get_missing_returns_none(self):
        """Feature: cold-start get returns None (no binding yet).
        Description: get() on an empty table for any request_id
        Expectation: returns None
        """
        t = StickySessionTable()
        assert t.get("r1") is None

    def test_s02_put_then_get_returns_replica(self):
        """Feature: put then get returns the bound replica_id.
        Description: put("r1","s0"); get("r1")
        Expectation: returns "s0"
        """
        t = StickySessionTable()
        t.put("r1", "s0")
        assert t.get("r1") == "s0"
        assert len(t) == 1

    def test_s03_put_refresh_updates_bound_replica(self):
        """Feature: re-put for same request_id updates the bound replica.
        Description: put("r1","s0"); put("r1","s1"); get("r1")
        Expectation: returns "s1" (overload-fallback routed elsewhere)
        """
        t = StickySessionTable()
        t.put("r1", "s0")
        t.put("r1", "s1")
        assert t.get("r1") == "s1"
        assert len(t) == 1

    def test_s04_get_refreshes_lru_recency(self):
        """Feature: get() on a hot key prevents its LRU eviction.
        Description: fill to max_size=2; touch r1; add r3; r1 still present, r2 evicted
        Expectation: r1 bound, r2 None (r2 was coldest)
        """
        t = StickySessionTable(max_size=2)
        t.put("r1", "s0")
        t.put("r2", "s1")
        # touch r1 so r2 becomes the coldest
        assert t.get("r1") == "s0"
        t.put("r3", "s2")  # evicts coldest (r2)
        assert t.get("r1") == "s0"  # still bound
        assert t.get("r2") is None  # evicted
        assert t.get("r3") == "s2"

    def test_s05_lru_evicts_coldest_when_full(self):
        """Feature: inserting past max_size evicts the least-recently-used.
        Description: max_size=2; put r1,r2,r3 in order
        Expectation: r1 evicted (coldest), r2/r3 bound
        """
        t = StickySessionTable(max_size=2)
        t.put("r1", "s0")
        t.put("r2", "s1")
        t.put("r3", "s2")  # evicts r1
        assert t.get("r1") is None
        assert t.get("r2") == "s1"
        assert t.get("r3") == "s2"

    def test_s06_invalidate_drops_single_binding(self):
        """Feature: invalidate(request_id) drops one binding.
        Description: put r1,s0; invalidate r1; get r1
        Expectation: get returns None, len 0
        """
        t = StickySessionTable()
        t.put("r1", "s0")
        t.invalidate("r1")
        assert t.get("r1") is None
        assert len(t) == 0

    def test_s07_invalidate_missing_is_noop(self):
        """Feature: invalidate on a missing key is a no-op.
        Description: invalidate("rX") on empty table
        Expectation: no error, len 0
        """
        t = StickySessionTable()
        t.invalidate("rX")  # must not raise
        assert len(t) == 0

    def test_s08_invalidate_replica_clears_all_bound(self):
        """Feature: invalidate_replica clears every binding to that replica.
        Description: r1→s0, r2→s1, r3→s0; invalidate_replica("s0")
        Expectation: r1/r3 gone, r2 still bound
        """
        t = StickySessionTable()
        t.put("r1", "s0")
        t.put("r2", "s1")
        t.put("r3", "s0")
        t.invalidate_replica("s0")
        assert t.get("r1") is None
        assert t.get("r3") is None
        assert t.get("r2") == "s1"

    def test_s09_invalidate_replica_missing_is_noop(self):
        """Feature: invalidate_replica on a replica with no bindings is a no-op.
        Description: no bindings point to sX; invalidate_replica("sX")
        Expectation: no error, table unchanged
        """
        t = StickySessionTable()
        t.put("r1", "s0")
        t.invalidate_replica("sX")
        assert t.get("r1") == "s0"
        assert len(t) == 1

    def test_s10_status_reports_max_size_and_current_size(self):
        """Feature: status() returns max_size and current size.
        Description: max_size=5; put 2 entries; status()
        Expectation: {"max_size": 5, "size": 2}
        """
        t = StickySessionTable(max_size=5)
        t.put("r1", "s0")
        t.put("r2", "s1")
        s = t.status()
        assert s == {"max_size": 5, "size": 2}

    def test_s11_default_max_size_matches_verl(self):
        """Feature: default max_size is 10000 (verl DEFAULT_ROUTING_CACHE_SIZE).
        Description: StickySessionTable() with no args
        Expectation: max_size == 10000
        """
        t = StickySessionTable()
        assert t.max_size == DEFAULT_STICKY_MAX_SIZE == 10000

    def test_s12_invalid_max_size_raises(self):
        """Feature: max_size <= 0 raises ValueError.
        Description: StickySessionTable(max_size=0) and (-1)
        Expectation: both raise ValueError
        """
        with pytest.raises(ValueError):
            StickySessionTable(max_size=0)
        with pytest.raises(ValueError):
            StickySessionTable(max_size=-1)
