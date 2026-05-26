"""LoadBalancer actor wraps a RouterPolicy and exposes Ray-remote methods."""
import pytest
import ray

from llm_router.load_balancer import LoadBalancer, _build_policy


@pytest.fixture(scope="module", autouse=True)
def ray_init_and_shutdown():
    ray.init(num_cpus=2, local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_loadbalancer_acquire_and_release():
    lb = LoadBalancer.remote(server_ids=["a", "b"], policy_name="legacy_sticky")
    s1 = ray.get(lb.acquire_server.remote("req-1"))
    assert s1 in ("a", "b")
    ray.get(lb.release_server.remote(s1))


def test_loadbalancer_sticky_binding():
    lb = LoadBalancer.remote(server_ids=["a", "b"], policy_name="legacy_sticky")
    s1 = ray.get(lb.acquire_server.remote("req-1"))
    ray.get(lb.release_server.remote(s1))
    s2 = ray.get(lb.acquire_server.remote("req-1"))
    assert s1 == s2


def test_loadbalancer_rule_based_policy_loads():
    lb = LoadBalancer.remote(server_ids=["a"], policy_name="rule_based")
    s = ray.get(lb.acquire_server.remote("req-1"))
    assert s == "a"


def test_loadbalancer_rejects_unknown_policy():
    # Ray's local_mode crashes at the C++ layer when a ValueError is raised from
    # an actor's __init__ (std::bad_optional_access abort), so we cannot rely on
    # pytest.raises catching it via __ray_ready__. The validation logic lives in
    # the module-level _build_policy helper that the actor delegates to; assert
    # on that directly. The validation invariant the test exercises ("policy_name
    # unknown -> ValueError") is identical.
    with pytest.raises(ValueError, match="unknown policy"):
        _build_policy(server_ids=["a"], policy_name="bogus", routing_cache_size=10)


def test_loadbalancer_report_prefixes_passthrough():
    lb = LoadBalancer.remote(server_ids=["a", "b"], policy_name="rule_based")
    # No exception, returns None.
    ret = ray.get(
        lb.report_prefixes.remote("a", [("v0", "deadbeef", 64)])
    )
    assert ret is None


def test_loadbalancer_report_then_acquire_rule1_path():
    """End-to-end through the actor: report a hit, then acquire with
    matching signatures, expect that server."""
    lb = LoadBalancer.remote(
        server_ids=["a", "b", "c"],
        policy_name="rule_based",
        hit_threshold=10,
        load_threshold=100,
    )
    ray.get(lb.report_prefixes.remote("b", [("v0", "h_long", 64)]))
    server = ray.get(
        lb.acquire_server.remote(
            "req-1",
            session_id="sess-1",
            prefix_signatures=[("v0", "h_long", 64)],
        )
    )
    assert server == "b"


def test_loadbalancer_report_prefixes_supports_cpu_tier_and_aliases():
    lb = LoadBalancer.remote(
        server_ids=["replica-0", "replica-1"],
        policy_name="rule_based",
        gpu_hit_threshold=100,
        cpu_hit_threshold=16,
        load_threshold=100,
        server_aliases={"node-b": ["replica-1"]},
    )
    ray.get(lb.report_prefixes.remote("node-b", [("v0", "h_cpu", 32)], tier="cpu"))
    server = ray.get(
        lb.acquire_server.remote(
            "req",
            session_id="sess",
            prefix_signatures=[("v0", "h_cpu", 32)],
        )
    )
    assert server == "replica-1"


def test_loadbalancer_legacy_report_prefixes_is_noop():
    """Legacy policy accepts report_prefixes (no-op) without affecting routing."""
    lb = LoadBalancer.remote(server_ids=["a"], policy_name="legacy_sticky")
    ray.get(lb.report_prefixes.remote("a", [("v0", "h", 64)]))
    s = ray.get(lb.acquire_server.remote("r1"))
    assert s == "a"
