"""Minimal unit tests for PD decode-node selection strategies (dr/uni-agent).

Run (from the uni-agent repo root; ``-t .`` avoids the ``tests/uni_agent`` namespace shadow):

    PYTHONPATH=$PWD python3 -m unittest discover -t . -s tests -p 'test_decode_selectors.py' -v
"""

import random
import types
import unittest

from uni_agent.rollout.decode_selectors import (
    CacheAwareDecodeSelector,
    ConsistentHashDecodeSelector,
    LeastLoadDecodeSelector,
    RandomDecodeSelector,
    RendezvousHashDecodeSelector,
    SessionStickyDecodeSelector,
)


def cands(*keys):
    return [types.SimpleNamespace(key=k, handle=None) for k in keys]


class _StubPressure:
    """Provide observe/snapshot methods with preset KV and queue pressure."""

    def __init__(self, kv_by_key=None, wait_by_key=None):
        self.kv_by_key = dict(kv_by_key or {})
        self.wait_by_key = dict(wait_by_key or {})
        self.observed = ()

    def observe(self, keys):
        self.observed = tuple(keys)

    def snapshot(self, key):
        if key not in self.kv_by_key and key not in self.wait_by_key:
            return None
        return types.SimpleNamespace(
            ts=0.0,
            kv_usage=self.kv_by_key.get(key),
            running=None,
            wait_capacity=self.wait_by_key.get(key),
        )


class TestLeastLoad(unittest.TestCase):
    """Sample k candidates and choose the one with the fewest in-flight calls."""

    def setUp(self):
        random.seed(0)
        self.pool = cands("a", "b", "c")

    def test_counts_track_selection_and_release(self):
        sel = LeastLoadDecodeSelector(sample_size=2)
        picks = [sel.select({}, self.pool).key for _ in range(9)]
        self.assertEqual(sum(sel._pending_requests), 9)
        self.assertGreaterEqual(len(set(picks)), 2, "least_load should spread load, not pin one peer")
        sel.release(0)
        self.assertEqual(sum(sel._pending_requests), 8)

    def test_avoids_loaded_peer(self):
        sel = LeastLoadDecodeSelector(sample_size=3)  # Compare all three candidates to find the minimum load.
        sel._pending_requests = [5, 0, 0]
        sel._candidate_keys = ("a", "b", "c")
        self.assertIn(sel.select({}, self.pool).key, ("b", "c"))
        self.assertEqual(sel._pending_requests[0], 5, "loaded peer must not be bumped")

    def test_release_never_goes_negative(self):
        sel = LeastLoadDecodeSelector()
        sel.select({}, self.pool)
        for i in range(5):
            sel.release(i)
        self.assertTrue(all(v >= 0 for v in sel._pending_requests))

    def test_candidate_set_change_resets_counts(self):
        sel = LeastLoadDecodeSelector()
        sel.select({}, self.pool)
        sel.select({}, self.pool)
        other = cands("x", "y", "z")  # Same count, different candidate identities.
        sel.select({}, other)
        self.assertEqual(sel._candidate_keys, ("x", "y", "z"))
        self.assertEqual(sum(sel._pending_requests), 1)


class TestCacheAware(unittest.TestCase):
    """Check prefix affinity, load fallback and in-flight request accounting."""

    def setUp(self):
        random.seed(0)
        self.pool = cands("a", "b", "c")

    def _req(self, prompt):
        # Supply both identity and prompt content for the selector's request contract.
        return {"routing_key": f"k-{prompt}", "prompt_ids": [prompt, prompt + 1]}

    def test_counts_and_release(self):
        sel = CacheAwareDecodeSelector(cache_threshold=0.9)  # A high match threshold favors load fallback.
        picks = [sel.select(self._req(100 + i), self.pool).key for i in range(6)]
        self.assertEqual(sum(sel._pending_requests), 6)
        self.assertGreaterEqual(len(set(picks)), 2, "no-hit requests must not all pin index 0")
        sel.release(0)
        self.assertEqual(sum(sel._pending_requests), 5)

    def test_candidate_set_change_rebuilds(self):
        sel = CacheAwareDecodeSelector()
        sel.select(self._req(1), self.pool)
        other = cands("x", "y", "z")
        sel.select(self._req(2), other)
        self.assertEqual(sel._candidate_keys, ("x", "y", "z"))
        self.assertEqual(sum(sel._pending_requests), 1)


class TestSessionSticky(unittest.TestCase):
    """session_sticky: routing_key stickiness + hash spread + overload fallback."""

    def setUp(self):
        random.seed(0)
        self.pool = cands("a", "b", "c")

    def _req(self, key, tokens=100):
        return {"routing_key": key, "prompt_ids": [i for i in range(tokens)]}

    def test_same_session_keeps_one_replica(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        picks = set()
        for i in range(20):
            picks.add(sel.select(self._req("sess-A"), self.pool).key)
            sel.select(self._req(f"sess-other-{i}"), self.pool)  # interleave other sessions
        self.assertEqual(len(picks), 1, f"one session must stay on one replica: {picks}")
        self.assertGreaterEqual(sel.stats["sticky"], 20)

    def test_sessions_spread_across_replicas(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        counts = {"a": 0, "b": 0, "c": 0}
        for i in range(300):
            counts[sel.select(self._req(f"sess-{i}"), self.pool).key] += 1
        self.assertTrue(all(v > 0 for v in counts.values()), counts)
        self.assertGreaterEqual(min(counts.values()), 60, f"each replica should get >=20%: {counts}")
        self.assertGreaterEqual(sel.stats["sticky"], 300)

    def test_overloaded_owner_is_skipped(self):
        pool = cands("a", "b")
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        owner_key = None
        for i in range(200):
            key = f"sess-{i}"
            if sel.select({"routing_key": key, "prompt_ids": [1]}, pool).key == "a":
                owner_key = key
                break
        self.assertIsNotNone(owner_key, "expected some session to hash onto 'a'")
        sel._pressure = _StubPressure({"a": 0.99})  # 'a' above kv_hard_stop
        picked = sel.select({"routing_key": owner_key, "prompt_ids": [1]}, pool).key
        self.assertEqual(picked, "b", "overloaded sticky owner must not be used")
        self.assertGreaterEqual(sel.stats["rebalance"], 1)

    def test_wait_queue_pressure_also_marks_overloaded(self):
        pool = cands("a", "b")
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        owner_key = None
        for i in range(200):
            key = f"sess-{i}"
            if sel.select({"routing_key": key, "prompt_ids": [1]}, pool).key == "a":
                owner_key = key
                break
        sel._pressure = _StubPressure({"a": 0.1}, {"a": 64})  # 64 >= wait_budget(32)
        self.assertEqual(sel.select({"routing_key": owner_key, "prompt_ids": [1]}, pool).key, "b")

    def test_all_overloaded_still_advances(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure({"a": 0.99, "b": 0.99, "c": 0.99}))
        picked = sel.select(self._req("sess-x"), self.pool).key
        self.assertIn(picked, ("a", "b", "c"))
        self.assertEqual(sel.stats["overload_all"], 1)

    def test_without_routing_key_falls_back_to_load(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        picks = {sel.select({"prompt_ids": [1]}, self.pool).key for _ in range(30)}
        self.assertGreaterEqual(len(picks), 2, "no session id -> still spread by token load")
        self.assertGreaterEqual(sel.stats["no_key"], 30)

    def test_token_accounting_and_release(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        sel.select(self._req("sess-y", tokens=50), self.pool)
        self.assertEqual(sum(sel._pending_tokens), 50)
        for i in range(3):
            sel.release(i)
        self.assertEqual(sum(sel._pending_tokens), 0)

    def test_candidate_set_change_resets_counters(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        sel.select(self._req("sess-z", tokens=1), self.pool)
        sel.select(self._req("sess-z", tokens=1), cands("x", "y", "z"))
        self.assertEqual(sel._candidate_keys, ("x", "y", "z"))
        self.assertEqual(sum(sel._pending_tokens), 1)

    def test_reset_keeps_mapping_and_counters(self):
        sel = SessionStickyDecodeSelector(pressure_cache=_StubPressure())
        first = sel.select(self._req("sess-r"), self.pool).key
        pending = sum(sel._pending_tokens)
        sel.reset()
        self.assertEqual(sum(sel._pending_tokens), pending, "in-flight accounting must survive reset")
        self.assertEqual(sel.select(self._req("sess-r"), self.pool).key, first)


class TestStatelessStrategies(unittest.TestCase):
    """Check random candidate membership and deterministic hash-policy routing."""

    def setUp(self):
        random.seed(0)
        self.pool = cands("a", "b", "c")

    def test_random_returns_member(self):
        sel = RandomDecodeSelector()
        for _ in range(20):
            self.assertIn(sel.select({"routing_key": "k", "prompt_ids": [1]}, self.pool).key, ("a", "b", "c"))

    def test_consistent_hash_is_deterministic(self):
        sel = ConsistentHashDecodeSelector(virtual_nodes=16)
        req = {"routing_key": "stable-key", "prompt_ids": [1, 2, 3]}
        first = sel.select(req, self.pool).key
        for _ in range(5):
            self.assertEqual(sel.select(req, self.pool).key, first)

    def test_rendezvous_hash_is_deterministic(self):
        sel = RendezvousHashDecodeSelector()
        req = {"routing_key": "stable-key", "prompt_ids": [1, 2, 3]}
        first = sel.select(req, self.pool).key
        for _ in range(5):
            self.assertEqual(sel.select(req, self.pool).key, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
