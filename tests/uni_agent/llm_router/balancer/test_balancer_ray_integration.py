"""End-to-end integration tests for KVCAwareBalancer via VeRL's drop-in path.

These exercise the **wiring** — VeRL ``get_router_handle`` → ``ray.remote`` →
KVCAwareBalancer Protocol methods across the Ray actor boundary — NOT the
routing quality. ``route()`` is a placeholder ranking (input order); the real
KV-aware algorithm lands with the strategy-module design. So acquire tests
assert handle-correctness for whatever id comes back, not which id is chosen.
Per detailed_balancer.md §5.3B.
"""

from __future__ import annotations

import pytest
import ray

from verl.workers.config.rollout import RouterConfig
from verl.workers.rollout.router import get_router_handle

pytestmark = [pytest.mark.st, pytest.mark.cpu]


# Package-relative router config. VeRL resolves pkg:// via importlib (see
# detailed_balancer.md §2.1), so this is robust to CWD / install location and
# matches how the router is configured in production.
_ROUTER_YAML = "pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml"


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def _router_config() -> RouterConfig:
    return RouterConfig(router_strategy="plugin_extension", router_config_path=_ROUTER_YAML)


class TestPluginExtensionEndToEnd:
    """I01-I05: full VeRL drop-in flow over a Ray actor (flow, not scheduling)."""

    def test_i01_actor_lists_servers_and_acquires_valid_handle(self, ray_runtime):
        """
        Feature: a freshly created actor lists its servers and acquire returns a valid handle
        Description: get_router_handle → get_all_servers; then acquire → (id, handle)
        Expectation: get_all_servers returns the provided ids; acquire returns an id in the pool
                     whose handle matches
        """
        servers = {"s0": "h0", "s1": "h1", "s2": "h2"}
        handle = get_router_handle(servers, _router_config())
        assert set(ray.get(handle.get_all_servers.remote())) == set(servers)
        server_id, actor_handle = ray.get(handle.acquire_server.remote("r1", [1, 2, 3]))
        assert server_id in servers
        assert actor_handle == servers[server_id]

    def test_i02_acquire_release_acquire_over_ray(self, ray_runtime):
        """
        Feature: release is a no-op and does not break subsequent routing
        Description: acquire → release → acquire over the Ray actor
        Expectation: release returns None; both acquires return valid (id, handle)
        """
        handle = get_router_handle({"s0": "h0"}, _router_config())
        sid1, h1 = ray.get(handle.acquire_server.remote("r1", [1, 2]))
        rel = ray.get(handle.release_server.remote(sid1))
        sid2, h2 = ray.get(handle.acquire_server.remote("r2", [1, 2]))
        assert rel is None
        assert {sid1, sid2} <= {"s0"}
        assert h1 == "h0" and h2 == "h0"

    def test_i03_add_remove_reflected_in_pool(self, ray_runtime):
        """
        Feature: pool mutations are visible through the actor
        Description: add_servers then get_all_servers; remove_servers then get_all_servers
        Expectation: pool reflects the add and the remove
        """
        handle = get_router_handle({"s0": "h0"}, _router_config())
        ray.get(handle.add_servers.remote({"s3": "h3"}))
        assert "s3" in ray.get(handle.get_all_servers.remote())
        ray.get(handle.remove_servers.remote(["s0"]))
        assert "s0" not in ray.get(handle.get_all_servers.remote())

    def test_i04_concurrent_acquires_both_valid(self, ray_runtime):
        """
        Feature: a single actor handles concurrent acquire calls without crashing
        Description: two concurrent acquire_server.remote() resolved together
        Expectation: both return a valid (id, handle) pair (v1 does not track inflight)
        """
        servers = {"s0": "h0", "s1": "h1"}
        handle = get_router_handle(servers, _router_config())
        (sid1, h1), (sid2, h2) = ray.get(
            [handle.acquire_server.remote("r1", [1]), handle.acquire_server.remote("r2", [2])]
        )
        for sid, h in [(sid1, h1), (sid2, h2)]:
            assert sid in servers
            assert h == servers[sid]

    def test_i05_construction_state_and_route_invocation(self, ray_runtime):
        """
        Feature: construction wires provider/strategies, and acquire invokes route()
        Description: get_router_handle → get_status (construction) → acquire → get_status (route called)
        Expectation: before acquire, provider=RouteDataProvider, strategy materialized, route_calls=0;
                     after one acquire, route_calls=1
        """
        handle = get_router_handle({"s0": "h0", "s1": "h1"}, _router_config())
        status = ray.get(handle.get_status.remote())
        assert status["provider"] == "RouteDataProvider"
        assert status["strategies"] == [{"type": "KVCacheAwareStrategy", "weight": 1.0}]
        assert set(status["servers"]) == {"s0", "s1"}
        assert status["route_calls"] == 0
        ray.get(handle.acquire_server.remote("r1", [1, 2, 3]))
        assert ray.get(handle.get_status.remote())["route_calls"] == 1
