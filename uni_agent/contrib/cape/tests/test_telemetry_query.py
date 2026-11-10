# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest

from pycape.telemetry_control import initial_state
from pycape.telemetry_query import prepare_query, sanitize_result, fetch_query


class TelemetryQueryTests(unittest.TestCase):
    def setUp(self):
        self.state = {"scheduler_id": "scheduler", "pools": {"pool-id": {"name": "shared"}},
                      "nodes": {"node-a": {"pool": "pool-id", "resources": {"gpu_uuids": ["GPU-A"]}}},
                      "telemetry": initial_state()}
        self.state["telemetry"]["bindings"]["node-a"] = "service"

    def plan(self, **changes):
        return prepare_query(self.state, dict(metric="sm_active_ratio", **changes), 1000)

    def test_fixed_selector_scope_and_staleness(self):
        plan = self.plan(node_id="node-a", pool="shared")
        self.assertEqual(plan["gpus"], ["GPU-A"])
        self.assertIn('scheduler_id="scheduler"', plan["params"]["query"])
        self.assertIn('service_id="service"', plan["params"]["query"])
        self.assertIn("sample_expires_at_seconds", plan["params"]["query"])
        self.assertIn("on(scheduler_id,service_id,gpu_uuid)", plan["params"]["query"])

    def test_no_promql_url_or_unbounded_history(self):
        for changes in ({"query": "up"}, {"url": "http://other"}, {"metric": "up"},
                        {"gpu_uuid": "GPU-secret"}, {"pool": "missing"}, {"node_id": "missing"},
                        {"start": 1}, {"mode": "history", "start": 1, "end": 10, "step": 0},
                        {"mode": "history", "start": "NaN", "end": 10, "step": 1},
                        {"mode": "history", "start": 0, "end": 1000000, "step": 1}):
            params = dict(metric="sm_active_ratio")
            params.update(changes)
            with self.subTest(params=params), self.assertRaises(ValueError):
                prepare_query(self.state, params, 1000)

    def test_disabled_and_unavailable_not_zero(self):
        result = asyncio.run(fetch_query(self.plan(), None))
        self.assertEqual(result["state"], "disabled")
        self.state["telemetry"]["default"]["enabled"] = True
        result = asyncio.run(fetch_query(self.plan(), None))
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["series"], [])

    def test_sanitization_and_nonfinite_as_missing(self):
        plan = self.plan(node_id="node-a")
        body = {"status": "success", "data": {"resultType": "vector", "result": [
            {"metric": {"scheduler_id": "scheduler", "service_id": "service", "gpu_uuid": "GPU-A", "secret": "hidden"},
             "value": [10, "NaN"]}]}}
        result = sanitize_result(plan, body)
        self.assertNotIn("hidden", str(result))
        self.assertIsNone(result["series"][0]["samples"][0][1])
        body["data"]["result"][0]["metric"]["scheduler_id"] = "other"
        with self.assertRaises(ValueError):
            sanitize_result(plan, body)


if __name__ == "__main__":
    unittest.main()
