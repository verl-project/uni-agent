"""Analytical curves, failure accounting and process turnover regression tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agentenv_offload" / "analyze.py"
spec = importlib.util.spec_from_file_location("offload_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def process(pid, start, read, write, pss=100, label="sandbox"):
    return {
        "pid": pid,
        "starttime_ticks": start,
        "status": "ok",
        "label": label,
        "memory": {"pss_bytes": pss},
        "io": {"read_bytes": read, "write_bytes": write},
    }


def metric(elapsed, pss, *, processes=None):
    return {
        "monotonic_seconds": 90 + elapsed,
        "selection_coverage_known": True,
        "registry_error": None,
        "processes": processes
        if processes is not None
        else [process(100, 123, 1000 + 100 * elapsed, 2000 + 200 * elapsed, pss)],
        "node": {
            "memory_bytes": {
                "AnonPages": 10 * analysis.GIB,
                "Cached": 2 * analysis.GIB,
                "MemAvailable": 20 * analysis.GIB,
            },
            "disk_devices": [
                {
                    "device": "vda",
                    "major_minor": "254:0",
                    "sector_bytes": 512,
                    "slaves": [],
                    "counters": {"sectors_read": 100 + elapsed * 10, "sectors_written": 200 + elapsed * 20},
                }
            ],
        },
    }


class NumericalTests(unittest.TestCase):
    def test_trapezoids_and_interpolated_edges_are_hand_computable(self):
        result = analysis.curve_summary([(0, 0), (1, analysis.GIB), (2, 0)], 0, 2)
        self.assertEqual(result["status"], "observed")
        self.assertAlmostEqual(result["integral_gib_seconds"], 1)
        self.assertEqual(result["peak_bytes"], analysis.GIB)
        clipped = analysis.curve_summary([(0, 0), (2, 2 * analysis.GIB)], 0.5, 1.5, max_gap=2)
        self.assertAlmostEqual(clipped["integral_gib_seconds"], 1)

    def test_missing_values_gaps_and_short_windows_are_not_extrapolated(self):
        missing = analysis.curve_summary([(0, 0), (1, None), (2, 0)], 0, 2)
        short = analysis.curve_summary([(0, 1), (1, 1)], 0, 2)
        gap = analysis.curve_summary([(0, 1), (3, 1)], 0, 3, max_gap=1)
        self.assertEqual(missing["reason"], "missing_metric_inside_window")
        self.assertEqual(short["reason"], "window_not_covered")
        self.assertEqual(gap["reason"], "sample_gap_exceeded")
        for result in (missing, short, gap):
            self.assertIsNone(result["integral_gib_seconds"])

    def test_io_turnover_differences_each_identity_without_counting_initial_bytes(self):
        snapshots = [
            [process(10, 1, 100, 200)],
            [process(10, 1, 150, 300), process(20, 2, 20, 40)],
            [process(20, 2, 70, 90)],
            [process(20, 3, 5, 5)],  # PID 20 has been reused.
            [process(20, 3, 15, 25)],
        ]
        samples = [{"time": index, "row": metric(index, 100, processes=rows)} for index, rows in enumerate(snapshots)]
        result = analysis.counter_deltas(samples, 0, 4)
        self.assertEqual(len(result["process_identities"]), 3)
        self.assertEqual(result["process_totals"]["observed_read_bytes_lower_bound"], 110)
        self.assertEqual(result["process_totals"]["observed_write_bytes_lower_bound"], 170)
        self.assertEqual(result["process_identities"][1]["first_counter_read_bytes"], 20)
        self.assertEqual(result["disk_devices"][0]["observed_read_bytes"], 40 * 512)

    def test_counter_reset_is_missing_not_negative_savings(self):
        samples = [
            {"time": 0, "row": metric(0, 1, processes=[process(1, 1, 100, 100)])},
            {"time": 1, "row": metric(1, 1, processes=[process(1, 1, 5, None)])},
        ]
        result = analysis.counter_deltas(samples, 0, 1)
        row = result["process_identities"][0]
        self.assertEqual(row["read_bytes_status"], "counter_reset")
        self.assertIsNone(row["observed_read_bytes"])
        self.assertIsNone(row["observed_write_bytes"])
        self.assertIsNone(result["process_totals"]["observed_read_bytes_lower_bound"])


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offload-analysis-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "agentenv_sha": "fixed-test-revision",
            "workload": "external simulated inference wait",
            "parameters": {"pairs": 1, "mode": "both", "io_tail_seconds": 1},
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.results = []

    def write_trial(self, mode, *, pair=0, success=True, complete=True, last_sample=5):
        trial_id = f"pair-{pair:03d}-{mode}-test"
        directory = self.root / trial_id
        directory.mkdir()
        events = [
            {
                "event": "trial_started",
                "trial_id": trial_id,
                "mode": mode,
                "monotonic_seconds": 98,
                "wall_time_ns": 1000000000,
            },
            {"event": "measurement_started", "monotonic_seconds": 100},
            {"event": "cleanup_confirmed", "monotonic_seconds": 103},
        ]
        (directory / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
        pss = 100 * 1024**2 if mode == "resident" else 50 * 1024**2
        metrics = [metric(t, pss) for t in range(-1, last_sample + 1)]
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
        if complete:
            result = {
                "trial_id": trial_id,
                "pair_index": pair,
                "mode": mode,
                "success": success,
                "cleanup": {"confirmed": True},
                "task_seconds": 10 if mode == "resident" else 10.2,
                "measurement_started_monotonic_seconds": 100,
                "clock_calibration": {"client_minus_node_seconds": 10, "uncertainty_seconds": 0.1},
                "cycles": [
                    {
                        "cycle": 0,
                        "first_tool_seconds": 0.1 if mode == "resident" else 0.5,
                        "restore_oracle_seconds": 0.05 if mode == "resident" else 0.4,
                    }
                ],
            }
            self.results.append(result)
            (directory / "result.json").write_text(json.dumps(result))
        (self.root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in self.results) + "\n")

    def test_complete_pair_measures_pss_without_claiming_physical_node_savings(self):
        self.write_trial("resident")
        self.write_trial("offload")
        result = analysis.analyze_directory(self.root)
        self.assertEqual(result["pairs"][0]["common_window_seconds"], 4)
        self.assertAlmostEqual(result["paired_summary"]["assigned_process_pss_saving_fraction"], 0.5)
        self.assertAlmostEqual(result["paired_summary"]["task_p95_relative_increase"], 0.02)
        self.assertAlmostEqual(result["paired_summary"]["first_tool_extra_seconds_task_p95"], 0.4)
        self.assertEqual(result["gates"]["assigned_process_pss_saving"], "pass")
        self.assertEqual(result["gates"]["physical_node_saving_attribution"], "unverified")
        node_curve = next(iter(result["trial_measurements"].values()))["curves"]["node_anon"]
        self.assertEqual(node_curve["integral_gib_seconds"], 40)
        self.assertIn("outer145", analysis.report_markdown(result))

    def test_failures_and_abandoned_trials_stay_in_denominator(self):
        self.write_trial("resident")
        self.write_trial("offload")
        self.write_trial("resident", pair=1, success=False)
        self.write_trial("offload", pair=1, complete=False)
        result = analysis.analyze_directory(self.root)
        self.assertEqual(result["started_tasks"], 4)
        self.assertEqual(result["by_mode"]["resident"]["success_rate"], 0.5)
        self.assertEqual(result["by_mode"]["offload"]["success_rate"], 0.5)
        self.assertEqual(result["by_mode"]["offload"]["cleanup_rate"], 0.5)
        self.assertEqual(result["gates"]["task_correctness_and_cleanup"], "fail")
        self.assertEqual(result["gates"]["task_p95_increase"], "unverified")

    def test_shorter_arm_does_not_get_extrapolated_to_common_window(self):
        self.write_trial("resident", last_sample=2)
        self.write_trial("offload")
        result = analysis.analyze_directory(self.root)
        self.assertEqual(result["paired_summary"]["memory_comparable_pairs"], 0)
        self.assertIsNone(result["paired_summary"]["assigned_process_pss_saving_fraction"])
        self.assertEqual(result["gates"]["assigned_process_pss_saving"], "unverified")
        self.assertEqual(result["pairs"][0]["reason"], "common_window_or_metric_coverage_insufficient")

    def test_interrupted_plan_preserves_diagnostics_but_cannot_pass_acceptance(self):
        self.manifest["parameters"]["pairs"] = 10
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.write_trial("resident")
        self.write_trial("offload")
        result = analysis.analyze_directory(self.root)
        self.assertEqual(result["started_tasks"], 2)
        self.assertEqual(result["unstarted_tasks"], 18)
        self.assertEqual(result["by_mode"]["offload"]["success_rate"], 1)
        self.assertAlmostEqual(result["paired_summary"]["assigned_process_pss_saving_fraction"], 0.5)
        self.assertEqual(result["gates"]["planned_run_complete"], "unfinished")
        self.assertEqual(result["gates"]["assigned_process_pss_saving"], "unverified")
        self.assertEqual(result["acceptance_status"], "unfinished")
        self.assertIn("计划任务完整执行", analysis.report_markdown(result))
        self.assertIn("sandbox 标签 PSS 下界", analysis.report_markdown(result))
        del self.manifest["parameters"]["pairs"]
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        unknown = analysis.analyze_directory(self.root)
        self.assertEqual(unknown["gates"]["planned_run_complete"], "unverified")
        self.assertEqual(unknown["acceptance_status"], "unverified")

    def test_cli_outputs_are_valid_and_raw_sources_are_unchanged(self):
        self.write_trial("resident")
        self.write_trial("offload")
        original = (self.root / "results.jsonl").read_bytes()
        self.assertEqual(analysis.main([str(self.root)]), 0)
        self.assertEqual((self.root / "results.jsonl").read_bytes(), original)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["started_tasks"], 2)
        self.assertTrue((self.root / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
