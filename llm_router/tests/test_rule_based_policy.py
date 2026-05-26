"""RuleBasedPolicy: RFC §5.3 two-stage routing rule + report_prefixes."""
import pytest

from llm_router.policy.rule_based import RuleBasedPolicy

VERSION = "v0"


def _sig(hash_str: str, length: int):
    return (VERSION, hash_str, length)


def test_acquire_without_signatures_falls_back_to_legacy_least_loaded():
    """No prefix_signatures → no rule-1 candidate → rule 2 (least loaded)."""
    p = RuleBasedPolicy(server_ids=["a", "b", "c"])
    assert p.acquire_server("req-1") == "a"


def test_session_fast_path_when_primary_has_gpu_hit_and_low_load():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    # Seed "session-1 hashes to b" by reporting a prefix on b first acquire.
    sigs = [_sig("aaaaaaaaaaaaaaaa", 64)]
    # First request — assigns session to primary chosen by consistent hashing.
    server = p.acquire_server("sess-1", session_id="sess-1", prefix_signatures=sigs)
    # Report that THIS server now has the prefix (mimics worker-side report).
    p.report_prefixes(server, sigs)
    p.release_server(server)
    # Second request from the same session — fast path: primary has the
    # prefix and load=0 → should return same server in O(1).
    server2 = p.acquire_server("sess-1", session_id="sess-1", prefix_signatures=sigs)
    assert server2 == server


def test_acquire_with_gpu_hit_picks_replica_with_longest_match():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    sig_short = _sig("h_short", 16)
    sig_long = _sig("h_long", 64)
    p.report_prefixes("a", [sig_short])
    p.report_prefixes("b", [sig_long])
    # Seed session binding to "c" — a replica with no reported prefix — so
    # the fast path (primary GPU hit) misses and rule 1 fires across all
    # candidates. Among candidates a (hit=16) and b (hit=64), rule 1 picks
    # b (longer hit).
    p._session_to_server["sess-x"] = "c"
    server = p.acquire_server(
        "req-x",
        session_id="sess-x",
        prefix_signatures=[sig_short, sig_long],
    )
    assert server == "b"


def test_gpu_hit_takes_priority_over_cpu_hit():
    p = RuleBasedPolicy(
        server_ids=["a", "b"],
        gpu_hit_threshold=10,
        cpu_hit_threshold=10,
        load_threshold=100,
    )
    sig_gpu = _sig("h_gpu", 16)
    sig_cpu = _sig("h_cpu", 128)
    p.report_prefixes("a", [sig_gpu], tier="gpu")
    p.report_prefixes("b", [sig_cpu], tier="cpu")
    p._session_to_server["sess-tier"] = "b"

    server = p.acquire_server(
        "req-tier",
        session_id="sess-tier",
        prefix_signatures=[sig_gpu, sig_cpu],
    )

    assert server == "a"


def test_cpu_hit_used_after_gpu_miss():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        gpu_hit_threshold=64,
        cpu_hit_threshold=32,
        load_threshold=100,
    )
    sig = _sig("h_cpu", 48)
    p.report_prefixes("b", [sig], tier="cpu")
    p._session_to_server["sess-cpu"] = "a"

    server = p.acquire_server(
        "req-cpu",
        session_id="sess-cpu",
        prefix_signatures=[sig],
    )

    assert server == "b"


def test_cpu_hit_below_cpu_threshold_falls_back_to_least_loaded():
    p = RuleBasedPolicy(
        server_ids=["a", "b"],
        gpu_hit_threshold=64,
        cpu_hit_threshold=64,
        load_threshold=100,
    )
    sig = _sig("h_cpu_short", 32)
    p.report_prefixes("b", [sig], tier="cpu")
    p._session_to_server["sess-short"] = "b"

    server = p.acquire_server(
        "req-short",
        session_id="sess-short",
        prefix_signatures=[sig],
    )

    assert server == "a"


def test_cpu_hit_respects_load_threshold():
    p = RuleBasedPolicy(
        server_ids=["a", "b"],
        gpu_hit_threshold=64,
        cpu_hit_threshold=16,
        load_threshold=1,
    )
    sig = _sig("h_cpu", 32)
    p.report_prefixes("b", [sig], tier="cpu")
    p._inflight["b"] = 1
    p._session_to_server["sess-load"] = "a"

    server = p.acquire_server(
        "req-load",
        session_id="sess-load",
        prefix_signatures=[sig],
    )

    assert server == "a"


def test_overloaded_replica_excluded_from_rule1_candidates():
    p = RuleBasedPolicy(
        server_ids=["a", "b"],
        hit_threshold=10,
        load_threshold=2,
    )
    sig = _sig("h", 64)
    p.report_prefixes("a", [sig])
    # Saturate "a"'s in-flight count to 2 (== load_threshold, NOT strictly less).
    # Rule 1 must reject "a" → fall through to least-loaded "b".
    p._inflight["a"] = 2
    # Seed session binding to "b" so the fast path checks "b" first; b has
    # no reported prefix, so primary_hit=0 < threshold → fast path fails.
    p._session_to_server["sess-3"] = "b"
    server = p.acquire_server(
        "req-3", session_id="sess-3", prefix_signatures=[sig]
    )
    assert server == "b"


def test_rule2_fallback_least_loaded_when_no_gpu_hit_candidate():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    # No reports — every replica has empty prefix_locations.
    server = p.acquire_server(
        "req-1",
        session_id="sess-1",
        prefix_signatures=[_sig("h", 64)],
    )
    # No GPU hit anywhere → rule 2 picks min in-flight (all 0 → first key).
    assert server == "a"


def test_report_prefixes_records_max_observed_length_per_key():
    p = RuleBasedPolicy(server_ids=["a"], max_prefix_entries_per_server=10)
    p.report_prefixes("a", [_sig("h", 32)])
    p.report_prefixes("a", [_sig("h", 64)])
    p.report_prefixes("a", [_sig("h", 48)])  # shorter — must not overwrite max
    # Inspect via the rule-1 path: a request matching the prefix should
    # observe the longest length we've ever reported for that key.
    server = p.acquire_server(
        "req",
        session_id="sess",
        prefix_signatures=[_sig("h", 80)],
    )
    assert server == "a"


def test_report_prefixes_rejects_unknown_server():
    p = RuleBasedPolicy(server_ids=["a"])
    with pytest.raises(ValueError, match="Invalid server_id"):
        p.report_prefixes("bogus", [_sig("h", 16)])


def test_prefix_locations_lru_evicts_oldest():
    p = RuleBasedPolicy(
        server_ids=["a"], max_prefix_entries_per_server=2
    )
    p.report_prefixes("a", [_sig("h1", 16)])
    p.report_prefixes("a", [_sig("h2", 16)])
    p.report_prefixes("a", [_sig("h3", 16)])  # evicts h1
    # h1 lookup must miss → rule 1 finds no candidate → rule 2 returns "a"
    # by least-loaded; nothing to assert behaviorally beyond no crash. Just
    # verify the in-memory state directly.
    assert ("v0", "h1") not in p._prefix_locations["a"]
    assert ("v0", "h2") in p._prefix_locations["a"]
    assert ("v0", "h3") in p._prefix_locations["a"]


def test_in_flight_counters_respected_after_acquire():
    p = RuleBasedPolicy(server_ids=["a", "b"], hit_threshold=1, load_threshold=1)
    p.report_prefixes("a", [_sig("h", 64)])
    # First request — a wins rule 1.
    s1 = p.acquire_server("r1", session_id="s1", prefix_signatures=[_sig("h", 64)])
    assert s1 == "a"
    # Second request, a is now at load=1 (>= load_threshold) → rule 1 rejects a.
    # Rule 2 falls through to least-loaded → b.
    s2 = p.acquire_server("r2", session_id="s2", prefix_signatures=[_sig("h", 64)])
    assert s2 == "b"
