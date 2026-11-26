"""Resource plotting must preserve measurement gaps and state uncertainty."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/agentenv_offload/plot_results.py"
SPEC = importlib.util.spec_from_file_location("offload_plot", PATH)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class CurveTests(unittest.TestCase):
    def test_preresume_annotation_comes_from_manifest_and_keeps_native_restore_scope(self):
        self.assertIsNone(plot.resume_lead_annotation({}))
        self.assertIsNone(plot.resume_lead_annotation({"resume_lead_seconds": 0}))
        for seconds in (1.5, 2.25):
            text = plot.resume_lead_annotation({"resume_lead_seconds": seconds})
            self.assertIn(f"{seconds:g} 秒", text)
            self.assertIn("已知合成期限", text)
            self.assertIn("底层恢复未加速", text)

    def test_integer_raw_values_clip_only_boundaries_and_integrate_known_area(self):
        times = np.array([-1.0, 1.0, 3.0])
        values = np.array([0, 2 * plot.GIB, 4 * plot.GIB], dtype=np.int64)
        x, y = plot.clip_observations(times, values, 2.0)
        np.testing.assert_allclose(x, [0, 1, 2])
        np.testing.assert_allclose(y / plot.GIB, [1, 2, 3])
        np.testing.assert_allclose(plot.cumulative_pss(x, y), [0, 1.5, 4.0])

    def test_missing_interval_is_not_bridged_or_added_as_zero_cost(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([plot.GIB, plot.GIB, np.nan, plot.GIB, plot.GIB])
        cumulative = plot.cumulative_pss(x, y)
        np.testing.assert_allclose(cumulative[:2], [0, 1])
        self.assertTrue(np.isnan(cumulative[2:]).all())
        clipped_x, clipped_y = plot.clip_observations(x, y, 3.5)
        self.assertTrue(math.isnan(clipped_y[2]))
        self.assertEqual(clipped_x[-1], 3.5)
        self.assertEqual(clipped_y[-1], plot.GIB)

    def test_missing_window_start_is_not_replaced_by_first_later_sample(self):
        x, y = plot.clip_observations(np.array([0.2, 1.0, 2.0]), np.array([10, 20, 30]), 1.5)
        self.assertTrue(math.isnan(y[0]))
        self.assertTrue(np.isnan(plot.cumulative_pss(x, y)).all())


class StateAndAlignmentTests(unittest.TestCase):
    def test_vm_phase_does_not_count_a_warm_process_or_fill_transition_gaps(self):
        result = {"measurement_started_monotonic_seconds": 100, "create_seconds": 1}
        events = [
            {
                "event": "node_observation",
                "phase": "before_create",
                "monotonic_seconds": 100,
                "nodes": [{"sandboxCount": 0}],
            },
            {"event": "created", "monotonic_seconds": 102},
            {
                "event": "firecracker_identity",
                "monotonic_seconds": 102.1,
                "processes": [{"pid": 20, "starttime_ticks": 7}],
            },
            {"event": "pause_requested", "monotonic_seconds": 104},
            {"event": "resources_released", "monotonic_seconds": 105},
            {"event": "resume_requested", "monotonic_seconds": 108},
        ]
        transitions = plot.lifecycle_transitions(result, events)
        row = {
            "selection_coverage_known": True,
            "registry_error": None,
            "processes": [{"pid": 20, "starttime_ticks": 7, "label": "provider", "status": "ok", "state": "S"}],
        }
        self.assertEqual(plot.verified_vm_count(row, 0.5, transitions), 0)
        self.assertTrue(math.isnan(plot.verified_vm_count(row, 1.5, transitions)))
        self.assertEqual(plot.verified_vm_count(row, 3, transitions), 1)
        self.assertTrue(math.isnan(plot.verified_vm_count(row, 4.5, transitions)))
        self.assertEqual(plot.verified_vm_count(row, 6, transitions), 0)
        self.assertTrue(math.isnan(plot.verified_vm_count(row, 8.5, transitions)))
        missing = {**row, "processes": []}
        self.assertTrue(math.isnan(plot.verified_vm_count(missing, 3, transitions)))

    def test_raw_loader_aligns_clock_and_retains_long_sample_gap(self):
        with tempfile.TemporaryDirectory(prefix="offload-plot-raw-") as temporary:
            root = Path(temporary)
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "measurement_started_monotonic_seconds": 100,
                        "clock_calibration": {"client_minus_node_seconds": 10, "uncertainty_seconds": 0.02},
                    }
                )
            )
            (root / "events.jsonl").write_text("")
            rows = [
                {
                    "monotonic_seconds": moment,
                    "registry_error": None,
                    "selection_coverage_known": True,
                    "processes": [{"status": "ok", "memory": {"pss_bytes": 1024}}],
                    "node": {"memory_bytes": {"AnonPages": 2048, "Cached": 4096}},
                    "sample_duration_seconds": 0.01,
                }
                for moment in (89.5, 90.0, 90.5, 93.0)
            ]
            raw = "\n".join(json.dumps(row) for row in rows)
            (root / "metrics.jsonl").write_text(raw)
            observed = plot.resource_observations(root, 1.0)
            np.testing.assert_allclose(observed["times"], [-0.5, 0, 0.5, 1.75, 3])
            self.assertTrue(math.isnan(observed["series"]["pss"][3]))
            self.assertTrue(math.isnan(observed["series"]["anon"][3]))
            self.assertEqual(observed["clock_uncertainty"], 0.02)
            self.assertEqual((root / "metrics.jsonl").read_text(), raw)


if __name__ == "__main__":
    unittest.main()
