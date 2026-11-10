# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
from unittest.mock import patch

from pycape.telemetry_contract import CollectionConfig, LatestSnapshots, RegistrationSet


class TelemetryContractTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.registry = RegistrationSet("scheduler", "host", clock=lambda: self.now)

    def test_default_off_and_strict_configuration(self):
        self.assertFalse(CollectionConfig().enabled)
        self.assertTrue(CollectionConfig().basic)
        self.assertFalse(CollectionConfig().profiling)
        self.assertEqual(CollectionConfig().interval_seconds, 5)
        for data in ({"enabled": 1}, {"profiling": "true"}, {"token": "secret"},
                     {"interval_seconds": float("nan")}, {"interval_seconds": True},
                     {"interval_seconds": 0}, {"stale_periods": 3.0}, []):
            with self.subTest(data=data), self.assertRaises(ValueError):
                CollectionConfig.parse(data)

    def test_agent_default_does_not_probe_nvml(self):
        from pycape.agent import NodeAgentClient
        from pycape.models import NodeResources
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient("http://127.0.0.1:1", "n", "p", NodeResources(1, 1, ()),
                                    log_dir=root, state_path=root + "/state.json")
            with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
                self.assertEqual(agent.metrics_collector.collect_gpu_metrics(), [])

    def test_union_filters_and_independent_scheduler(self):
        self.registry.register("scheduler", "a", "epoch-a", "shared", ["GPU-A", "GPU-A"])
        self.registry.register("scheduler", "b", "epoch-b", "exclusive", ["GPU-A", "MIG-B"])
        self.assertEqual(self.registry.gpu_uuids(), ["GPU-A", "MIG-B"])
        self.assertEqual(self.registry.gpu_uuids(pool_id="shared"), ["GPU-A"])
        self.assertEqual(self.registry.gpu_uuids(node_id="missing"), [])
        with self.assertRaises(ValueError):
            self.registry.register("other", "c", "e", "p", ["GPU-A"])
        other = RegistrationSet("other", "host")
        self.assertEqual(other.gpu_uuids(), [])

    def test_missing_optional_helper_is_visible_without_breaking_status(self):
        from pycape.agent import NodeAgentClient
        from pycape.models import NodeResources
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", CAPE_METRIC_HELPER_BIN="/missing/cape-metric-helper"):
            agent = NodeAgentClient("http://127.0.0.1:1", "n", "p", NodeResources(1, 1, ()),
                                    log_dir=root, state_path=root + "/state.json", metrics_enabled=True)
            self.assertEqual(agent.collect_status()["metrics"]["state"], "error")
            agent.metrics_link.close()

    def test_old_exit_cannot_remove_new_incarnation(self):
        self.registry.register("scheduler", "a", "old", "p", ["GPU-A"])
        with self.assertRaises(ValueError):
            self.registry.register("scheduler", "a", "new", "p", ["GPU-B"])
        self.now = 40  # Previous registration expires.
        self.registry.register("scheduler", "a", "new", "p", ["GPU-B"])
        self.assertFalse(self.registry.remove("a", "old"))
        self.assertEqual(self.registry.gpu_uuids(), ["GPU-B"])
        self.assertTrue(self.registry.remove("a", "new"))

    def test_renewal_and_other_holder_survive(self):
        self.registry.register("scheduler", "a", "a", "p", ["GPU-A"])
        self.now = 20
        self.registry.register("scheduler", "b", "b", "p", ["GPU-A"])
        self.now = 40
        self.assertEqual(self.registry.expire(), 1)
        self.assertEqual(self.registry.gpu_uuids(), ["GPU-A"])
        self.now = 50
        self.assertEqual(self.registry.gpu_uuids(), [])

    def test_limits_and_validation_do_not_partially_register(self):
        registry = RegistrationSet("s", "h", capacity=1, clock=lambda: self.now)
        registry.register("s", "a", "e", "p", ["GPU-A"])
        for node, gpus in (("b", ["GPU-B"]), ("a", ["GPU-B", "bad\nlabel"]),
                           ("a", "GPU-A")):
            with self.assertRaises(ValueError):
                registry.register("s", node, "e", "p", gpus)
        self.assertEqual(registry.gpu_uuids(), ["GPU-A"])

    def test_snapshot_expiry_no_replay_refresh_and_no_history(self):
        snapshots = LatestSnapshots(clock=lambda: self.now)
        self.assertEqual(snapshots.get("GPU-A", 15)["state"], "unavailable")
        self.assertTrue(snapshots.put("GPU-A", 1000, {"sm_active": 0, "tensor_active": None}))
        self.now = 24
        self.assertFalse(snapshots.put("GPU-A", 1000, {"sm_active": 99}))
        self.assertEqual(snapshots.get("GPU-A", 15)["values"]["sm_active"], 0)
        self.now = 25
        self.assertEqual(snapshots.get("GPU-A", 15)["state"], "stale")
        self.assertEqual(snapshots.get("GPU-A", 15)["values"], {})
        snapshots.put("GPU-A", 1001, {"sm_active": 1})
        returned = snapshots.get("GPU-A", 15)
        returned["values"]["sm_active"] = 99
        self.assertEqual(snapshots.get("GPU-A", 15)["values"]["sm_active"], 1)
        snapshots.retain([])
        self.assertEqual(snapshots.get("GPU-A", 15)["state"], "unavailable")

    def test_invalid_samples_do_not_replace_latest(self):
        snapshots = LatestSnapshots(capacity=1, clock=lambda: self.now)
        snapshots.put("GPU-A", 1, {"value": 1})
        for value in (float("inf"), float("nan"), True, "1"):
            with self.assertRaises(ValueError):
                snapshots.put("GPU-A", 2, {"value": value})
        with self.assertRaises(ValueError):
            snapshots.put("GPU-B", 2, {})
        self.assertEqual(snapshots.get("GPU-A", 15)["sampled_at"], 1)


if __name__ == "__main__":
    unittest.main()
