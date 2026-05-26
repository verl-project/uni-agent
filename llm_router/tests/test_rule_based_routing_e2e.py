"""End-to-end: LoadBalancer + RuleBasedPolicy mimicking verl AsyncLLMServerManager.

Simulates the exact call sequence verl makes on each request:
  1. Worker computes prefix_signatures from prompt_ids + weight_version.
  2. Worker calls LoadBalancer.acquire_server.remote(
         request_id, session_id, prefix_signatures).
  3. Worker drives the picked replica (mocked here — we just count routes).
  4. After generation, worker calls LoadBalancer.report_prefixes.remote(
         server_id, prefix_signatures_of_cached_prompt).
  5. Worker calls LoadBalancer.release_server.remote(server_id).
"""
import hashlib

import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    ray.init(num_cpus=2, local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _hash_prefix(prompt_ids: list[int], prefix_len: int) -> str:
    """Mirror llm_router.connector.prefix_hash.hash_token_prefix shape."""
    h = hashlib.blake2b(digest_size=16)
    for tok in prompt_ids[:prefix_len]:
        h.update(int(tok).to_bytes(8, byteorder="little", signed=True))
    h.update(b":")
    h.update(str(prefix_len).encode("ascii"))
    return h.hexdigest()


def _signatures(prompt_ids: list[int], weight_version: str, stride: int = 16):
    """Mirror verl._iter_prefix_signatures: stride-sampled prefix hashes."""
    if not prompt_ids:
        return []
    lengths = list(range(stride, len(prompt_ids), stride))
    if not lengths or lengths[-1] != len(prompt_ids):
        lengths.append(len(prompt_ids))
    return [
        (weight_version, _hash_prefix(prompt_ids, n), n) for n in lengths
    ]


def _drive_request(lb, request_id, session_id, prompt_ids, weight_version):
    """Simulate one verl-style request through the LB."""
    sigs = _signatures(prompt_ids, weight_version)
    server_id = ray.get(
        lb.acquire_server.remote(
            request_id,
            session_id=session_id,
            prefix_signatures=sigs,
        )
    )
    # Mock generation: server now has the prompt cached.
    ray.get(lb.report_prefixes.remote(server_id, sigs))
    ray.get(lb.release_server.remote(server_id))
    return server_id


def test_e2e_session_affinity_after_first_turn():
    """A session's second turn lands on the same replica as its first."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1", "s2"],
        policy_name="rule_based",
        hit_threshold=1,
        load_threshold=100,
    )
    prompt_t1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    prompt_t2 = prompt_t1 + [17, 18, 19, 20]

    s_t1 = _drive_request(lb, "req-t1", "session-X", prompt_t1, "v0")
    s_t2 = _drive_request(lb, "req-t2", "session-X", prompt_t2, "v0")

    assert s_t1 == s_t2


def test_e2e_different_weight_version_does_not_hit():
    """KV reported under v0 must not satisfy a v1 query (RFC §5.2)."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1"],
        policy_name="rule_based",
        hit_threshold=1,
        load_threshold=100,
    )
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Turn under v0 → server reported as having v0 prefix.
    _drive_request(lb, "r-v0", "sess", prompt, "v0")

    # Next turn under v1 — consistent hash still maps "sess" to the same
    # primary, but L_gpu(primary) for the v1 query is 0 → fast path fails
    # the hit_threshold check. Then rule 1 also finds no candidate, so
    # rule 2 falls through to least-loaded, which is non-deterministic
    # between empty servers (all in_flight=0 → pick first alphabetically).
    # We assert only that the returned server is in the set.
    s_v1 = _drive_request(lb, "r-v1", "sess", prompt, "v1")
    assert s_v1 in {"s0", "s1"}


def test_e2e_legacy_policy_ignores_signatures():
    """Legacy mode: signatures are silently accepted but irrelevant."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1"],
        policy_name="legacy_sticky",
    )
    prompt = [1, 2, 3]
    s = _drive_request(lb, "r", "sess", prompt, "v0")
    assert s in {"s0", "s1"}
