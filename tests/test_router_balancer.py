"""CPU tests for the uni-agent-to-verl router protocol adapter."""

import unittest
from importlib.resources import files

from omegaconf import OmegaConf

from uni_agent.rollout.router_balancer import SelectorRouter


class _NoPressure:
    def observe(self, keys):
        pass

    def snapshot(self, key):
        return None


class TestSelectorRouter(unittest.TestCase):
    def setUp(self):
        self.servers = {"http://decode-a:1": object(), "http://decode-b:1": object()}
        self.router = SelectorRouter(
            self.servers,
            {
                "router_class": "uni_agent.rollout.router_balancer.SelectorRouter",
                "selector_class": "uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector",
                "pressure_cache": _NoPressure(),
            },
        )

    def test_session_sticky_acquire_and_release(self):
        first_id, first_handle = self.router.acquire_server(
            "session-a",
            prompt_ids=list(range(32)),
            reservation_id="turn-1",
        )
        second_id, second_handle = self.router.acquire_server(
            "session-a",
            prompt_ids=list(range(32)),
            reservation_id="turn-2",
        )
        self.assertEqual(first_id, second_id)
        self.assertIs(first_handle, second_handle)
        self.assertEqual(self.router.get_total_inflight(), 2)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 64)
        self.router.release_server(first_id, "session-a", reservation_id="turn-1")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 32)

    def test_server_pool_can_be_updated(self):
        new_handle = object()
        self.router.add_servers({"http://decode-c:1": new_handle})
        self.assertIn("http://decode-c:1", self.router.get_all_servers())
        self.router.remove_servers(["http://decode-a:1"])
        self.assertNotIn("http://decode-a:1", self.router.get_all_servers())

    def test_session_sticky_yaml_is_upstream_router_compatible(self):
        config = OmegaConf.to_container(
            OmegaConf.create(files("uni_agent.rollout.configs").joinpath("session_sticky.yaml").read_text()),
            resolve=True,
        )
        config["pressure_cache"] = _NoPressure()
        router = SelectorRouter(self.servers, config)
        server_id, _ = router.acquire_server("yaml-session", prompt_ids=[1, 2, 3])
        self.assertIn(server_id, self.servers)

    def test_declares_upstream_protocol_fields(self):
        self.assertEqual(self.router.require_acquire_fields(), ["prompt_ids", "reservation_id"])
        self.assertEqual(self.router.require_release_fields(), ["request_id", "reservation_id"])

    def test_all_packaged_policies_load(self):
        for policy in (
            "random",
            "least_load",
            "power_of_two",
            "consistent_hash",
            "rendezvous_hash",
            "cache_aware",
            "session_sticky",
        ):
            with self.subTest(policy=policy):
                config = OmegaConf.to_container(
                    OmegaConf.create(files("uni_agent.rollout.configs").joinpath(f"{policy}.yaml").read_text()),
                    resolve=True,
                )
                if policy == "session_sticky":
                    config["pressure_cache"] = _NoPressure()
                router = SelectorRouter(self.servers, config)
                self.assertEqual(router.get_total_inflight(), 0)
                server_id, handle = router.acquire_server("test-session", [1, 2, 3])
                self.assertIs(handle, self.servers[server_id])
                self.assertEqual(router.get_total_inflight(), 1)
                expected_tokens = 3 if policy == "session_sticky" else None
                self.assertEqual(router.get_status()["pending_prompt_tokens"], expected_tokens)
                router.release_server(server_id, "test-session")
                self.assertEqual(router.get_total_inflight(), 0)

    def test_reset_preserves_pending_tokens(self):
        self.router.acquire_server("a", [1, 2, 3])
        self.router.clear_sticky_cache()
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 3)

    def test_missing_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selector_class"):
            SelectorRouter(self.servers, {})

    def test_out_of_order_release_returns_the_finished_request_weight(self):
        self.router = SelectorRouter(
            {"d": object()},
            {
                "selector_class": "uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector",
                "pressure_cache": _NoPressure(),
            },
        )
        self.router.acquire_server("long", [0] * 10000)
        self.router.acquire_server("short", [0] * 1000)
        self.router.release_server("d", "short")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 10000)
        self.router.release_server("d", "long")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_same_session_uses_distinct_reservations(self):
        long_id, _ = self.router.acquire_server("session", [0] * 10000, reservation_id="long")
        short_id, _ = self.router.acquire_server("session", [0] * 1000, reservation_id="short")
        self.assertEqual(long_id, short_id)
        self.router.release_server(short_id, "session", reservation_id="short")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 10000)
        self.router.release_server(long_id, "session", reservation_id="long")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_duplicate_or_unknown_release_does_not_release_another_request(self):
        first, _ = self.router.acquire_server("session", [1, 2], reservation_id="first")
        second, _ = self.router.acquire_server("session", [3], reservation_id="second")
        self.router.release_server(first, reservation_id="first")
        self.router.release_server(first, reservation_id="first")
        self.router.release_server(second, reservation_id="unknown")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.router.release_server(second, reservation_id="second")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_duplicate_acquire_is_rejected_before_incrementing_load(self):
        self.router.acquire_server("same", [1, 2])
        with self.assertRaisesRegex(ValueError, "already active"):
            self.router.acquire_server("same", [3, 4, 5])
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 2)

    def test_zero_token_release_does_not_consume_a_nonempty_request(self):
        first, _ = self.router.acquire_server("session", [1, 2, 3], reservation_id="nonempty")
        empty, _ = self.router.acquire_server("session", [], reservation_id="empty")
        self.assertEqual(self.router.get_total_inflight(), 2)
        self.router.release_server(empty, reservation_id="empty")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 3)
        self.router.release_server(first, reservation_id="nonempty")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_wrong_server_release_keeps_the_reservation(self):
        server_id, _ = self.router.acquire_server("request", [1, 2])
        wrong_id = next(key for key in self.servers if key != server_id)
        with self.assertRaisesRegex(ValueError, "belongs to"):
            self.router.release_server(wrong_id, "request")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 2)
        self.router.release_server(server_id, "request")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_pool_membership_requires_drain_and_preserves_pending_state(self):
        server_id, _ = self.router.acquire_server("request", [1, 2])
        with self.assertRaisesRegex(RuntimeError, "drain"):
            self.router.add_servers({"new": object()})
        with self.assertRaisesRegex(RuntimeError, "drain"):
            self.router.remove_servers([server_id])
        self.assertEqual(self.router.get_all_servers(), list(self.servers))
        self.router.clear_sticky_cache()
        self.router.release_server(server_id, "request")
        self.router.add_servers({"new": object()})
        self.router.remove_servers([server_id])
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_duplicate_release_is_also_idempotent_for_request_count_policy(self):
        router = SelectorRouter(
            {"d": object()}, {"selector_class": "uni_agent.rollout.decode_selectors.LeastLoadDecodeSelector"}
        )
        router.acquire_server("one", [1])
        router.acquire_server("two", [2])
        router.release_server("d", "one")
        router.release_server("d", "one")
        self.assertEqual(router.get_total_inflight(), 1)
        router.release_server("d", "two")
        self.assertEqual(router.get_total_inflight(), 0)

    def test_status_separates_requests_from_zero_and_nonzero_token_weights(self):
        first, _ = self.router.acquire_server("session", [1, 2, 3], reservation_id="nonempty")
        empty, _ = self.router.acquire_server("session", [], reservation_id="empty")
        self.assertEqual(self.router.get_status()["total_inflight"], 2)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 3)
        self.router.release_server(first, reservation_id="nonempty")
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 0)
        self.router.release_server(empty, reservation_id="empty")
        self.assertEqual(self.router.get_total_inflight(), 0)

    def test_same_id_replacement_after_drain_uses_new_handle_and_resets_affinity(self):
        first_handle, old_handle, new_handle = object(), object(), object()
        router = SelectorRouter(
            {"d0": first_handle, "d1": old_handle},
            {"selector_class": "uni_agent.rollout.decode_selectors.CacheAwareDecodeSelector"},
        )
        self.assertEqual(router.acquire_server("busy", [9])[0], "d0")
        self.assertEqual(router.acquire_server("warm", [1, 2, 3])[0], "d1")
        router.release_server("d0", "busy")
        router.release_server("d1", "warm")
        router.add_servers({"d1": new_handle})
        self.assertEqual(router.get_all_servers(), ["d0", "d1"])
        # The replacement cannot inherit the old engine's prefix affinity.
        server_id, handle = router.acquire_server("next", [1, 2, 3])
        self.assertEqual(server_id, "d0")
        self.assertIs(handle, first_handle)
        replacement_id, replacement_handle = router.acquire_server("other", [4, 5])
        self.assertEqual(replacement_id, "d1")
        self.assertIs(replacement_handle, new_handle)
        router.release_server(server_id, "next")
        router.release_server(replacement_id, "other")
        self.assertEqual(router.get_total_inflight(), 0)

    def test_active_handle_replacement_rejects_the_entire_pool_update(self):
        server_id, _ = self.router.acquire_server("request", [1, 2])
        replacement = object()
        with self.assertRaisesRegex(RuntimeError, "drain"):
            self.router.add_servers({server_id: replacement, "new": object()})
        self.assertEqual(self.router._servers, self.servers)
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.router.release_server(server_id, "request")
        self.router.add_servers({server_id: replacement})
        self.assertIs(self.router._servers[server_id], replacement)

    def test_readding_the_same_handle_does_not_change_active_accounting(self):
        server_id, handle = self.router.acquire_server("request", [1, 2])
        self.router.add_servers({server_id: handle})
        self.assertEqual(self.router.get_total_inflight(), 1)
        self.assertEqual(self.router.get_status()["pending_prompt_tokens"], 2)
        self.router.release_server(server_id, "request")
        self.assertEqual(self.router.get_total_inflight(), 0)


if __name__ == "__main__":
    unittest.main()
