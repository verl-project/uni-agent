# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path


ANALYZE_PATH = Path(__file__).resolve().parent / "analyze.py"
SEPARATED_SUBMITTER_PATH = (
    Path(__file__).resolve().parent / "separated_submitter.sh"
)
START_PATH = Path(__file__).resolve().parent / "start.sh"
spec = importlib.util.spec_from_file_location("cape_mimic_analyze", str(ANALYZE_PATH))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class MimicAnalyzeTests(unittest.TestCase):
    def test_retained_execution_states_release_only_confirmed_gpu_accounting(self):
        for state, released in (("SUSPENDING", False), ("SUSPENDED", True),
                                ("PURGING", True), ("PURGED", True)):
            with self.subTest(state=state):
                events = [{"event_type": "session_state", "entity_id": identity,
                           "state": value, "timestamp": index,
                           "details": {"execution_session_id": identity, "gpu_uuids": ["GPU-one"]}}
                          for index, (identity, value) in enumerate((
                              ("old", "RUNNING_COMMAND"), ("old", state), ("new", "RUNNING_COMMAND")))]
                self.assertEqual(analyze.analyze_journal(events)["lease_uniqueness_pass"], released)

    def test_separated_submitter_uses_the_canonical_busy_threshold(self):
        source = SEPARATED_SUBMITTER_PATH.read_text()

        self.assertNotIn('${BUSY_THRESHOLD:-0.5}', source)
        self.assertIn('${BUSY_THRESHOLD:-0.9}', source)
        self.assertIn("runtime_adapter,isolation_policy,command_id,log_path", source)
        self.assertIn("collect_runtime_logs", source)
        self.assertIn('${CAPE_REQUEST_RUNTIME_ADAPTER:-apptainer}', source)
        self.assertIn('${CAPE_ISOLATION_POLICY:-image-stateful}', source)
        self.assertIn('--expected-runtime-adapter "$EXPECTED_RUNTIME_ADAPTER"', source)
        self.assertIn('--expected-isolation-policy "$EXPECTED_ISOLATION_POLICY"', source)

    def test_runtime_evidence_requires_matching_exact_command_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "pool", "pool-0", "agent-logs")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "cmd-000001.log")
            with open(log_path, "w") as fh:
                fh.write(
                    "runtime-adapter: apptainer\n"
                    "isolation-policy: image-stateful\n"
                    "exec: /usr/bin/apptainer exec image.sif true\n"
                    "argv-json: [\"/usr/bin/apptainer\", \"exec\"]\n"
                    "exit-code: 0\n"
                )
            rows = [{
                "jobid": "req-000001",
                "state": "COMPLETED",
                "node_id": "pool-0",
                "command_id": "cmd-000001",
                "runtime_adapter": "apptainer",
                "isolation_policy": "image-stateful",
            }]

            result = analyze.analyze_runtime_evidence(
                tmp, rows, expected_adapter="apptainer", expected_policy="image-stateful"
            )
            self.assertTrue(result["pass"])
            self.assertEqual(result["checks"][0]["log_path"], "pool/pool-0/agent-logs/cmd-000001.log")

            rows[0]["runtime_adapter"] = "podman"
            result = analyze.analyze_runtime_evidence(
                tmp, rows, expected_adapter="apptainer", expected_policy="image-stateful"
            )
            self.assertFalse(result["pass"])

    def test_single_node_harness_captures_runtime_fields_and_log_layout(self):
        source = START_PATH.read_text()
        self.assertIn(
            "gpu_uuids,runtime_adapter,isolation_policy,command_id",
            source,
        )
        self.assertIn('data.get("runtime_adapter", "")', source)
        self.assertIn('data.get("isolation_policy", "")', source)
        self.assertIn('data.get("command_id", "")', source)

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "pool", "agent-logs", "node-0")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "cmd-000001.log")
            with open(log_path, "w") as fh:
                fh.write(
                    "runtime-adapter: local-process\n"
                    "isolation-policy: none\n"
                    "exec: bash workload.sh\n"
                    "argv-json: [\"bash\", \"workload.sh\"]\n"
                    "exit-code: 0\n"
                )
            result = analyze.analyze_runtime_evidence(
                tmp,
                [{
                    "jobid": "req-000001",
                    "node_id": "node-0",
                    "command_id": "cmd-000001",
                    "runtime_adapter": "local-process",
                    "isolation_policy": "none",
                }],
                expected_adapter="local-process",
                expected_policy="none",
            )
            self.assertTrue(result["pass"])
            self.assertEqual(
                result["checks"][0]["log_path"],
                "pool/agent-logs/node-0/cmd-000001.log",
            )

    def test_analyze_writes_summary_for_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_csv(
                os.path.join(tmp, "submits.csv"),
                ["uid", "request", "jobid", "t_start", "t_return", "latency", "session_id"],
                [
                    ["user-000", "A", "req-000001", "1.0", "1.1", "0.1", "user-000/task"],
                    ["user-000", "B", "req-000002", "1.2", "1.3", "0.1", "user-000/task"],
                ],
            )
            self._write_csv(
                os.path.join(tmp, "utilization_series.csv"),
                ["t", "gpus_total", "gpus_leased", "idle", "queued", "running"],
                [
                    ["1.0", "1", "1", "0", "1", "1"],
                    ["2.0", "1", "1", "0", "0", "1"],
                    ["3.0", "1", "0", "1", "0", "0"],
                ],
            )
            self._write_csv(
                os.path.join(tmp, "status_transitions.csv"),
                ["jobid", "state", "t", "exit_code"],
                [
                    ["req-000001", "COMPLETED", "2.0", "0"],
                    ["req-000002", "COMPLETED", "3.0", "0"],
                ],
            )
            out_dir = os.path.join(tmp, "users", "user-000", "output")
            os.makedirs(out_dir)
            self._write_result(os.path.join(out_dir, "gemm_silu_result.txt"), "gemm_silu", "GPU-a", checksum="12.0")
            self._write_result(os.path.join(out_dir, "cutedsl_add_result.txt"), "cutedsl_add", "GPU-a")
            self._write_journal(os.path.join(tmp, "journal.sqlite"))
            with open(os.path.join(tmp, "auth_probes.json"), "w") as fh:
                json.dump({
                    "status": "PASS",
                    "pass": True,
                    "cross_user_status": {"denied": True, "return_code": 1},
                    "node_agent_submit": {"denied": True, "return_code": 1},
                }, fh)

            rc = analyze.main([tmp, "--busy-threshold", "0.5"])
            self.assertEqual(rc, 0)
            with open(os.path.join(tmp, "summary.json")) as fh:
                summary = json.load(fh)
            self.assertTrue(summary["overall_pass"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary.md")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "isolation_checks.json")))

    def test_utilization_busy_fraction_excludes_startup_and_partial_demand_tail(self):
        rows = [
            {"t": "0.0", "gpus_total": "4", "gpus_leased": "0", "idle": "4", "queued": "8", "running": "0"},
            {"t": "1.0", "gpus_total": "4", "gpus_leased": "4", "idle": "0", "queued": "4", "running": "4"},
            {"t": "2.0", "gpus_total": "4", "gpus_leased": "4", "idle": "0", "queued": "0", "running": "4"},
            {"t": "3.0", "gpus_total": "4", "gpus_leased": "1", "idle": "3", "queued": "0", "running": "1"},
            {"t": "4.0", "gpus_total": "4", "gpus_leased": "0", "idle": "4", "queued": "0", "running": "0"},
        ]

        result = analyze.analyze_utilization(rows, grace_seconds=2.0, busy_threshold=0.9)

        self.assertEqual(result["steady_state_samples"], 2)
        self.assertEqual(result["startup_excluded_samples"], 1)
        self.assertEqual(result["tail_excluded_samples"], 2)
        self.assertAlmostEqual(result["busy_fraction"], 1.0)
        self.assertTrue(result["busy_pass"])
        self.assertTrue(result["work_conserving_pass"])

    def test_tail_exclusion_does_not_hide_work_conserving_violation(self):
        rows = [
            {"t": "0.0", "gpus_total": "4", "gpus_leased": "4", "idle": "0", "queued": "4", "running": "4"},
            {"t": "1.0", "gpus_total": "4", "gpus_leased": "1", "idle": "3", "queued": "1", "running": "1"},
            {"t": "4.0", "gpus_total": "4", "gpus_leased": "1", "idle": "3", "queued": "1", "running": "1"},
        ]

        result = analyze.analyze_utilization(rows, grace_seconds=2.0, busy_threshold=0.9)

        self.assertEqual(result["steady_state_samples"], 1)
        self.assertEqual(result["tail_excluded_samples"], 2)
        self.assertAlmostEqual(result["busy_fraction"], 1.0)
        self.assertTrue(result["busy_pass"])
        self.assertEqual(result["invariant_violations"], 1)
        self.assertFalse(result["work_conserving_pass"])

    def test_busy_fraction_fails_closed_without_saturated_steady_state(self):
        rows = [
            {"t": "0.0", "gpus_total": "4", "gpus_leased": "1", "idle": "3", "queued": "0", "running": "1"},
            {"t": "1.0", "gpus_total": "4", "gpus_leased": "0", "idle": "4", "queued": "0", "running": "0"},
        ]

        result = analyze.analyze_utilization(rows, grace_seconds=2.0, busy_threshold=0.9)

        self.assertEqual(result["steady_state_samples"], 0)
        self.assertEqual(result["startup_excluded_samples"], 2)
        self.assertEqual(result["tail_excluded_samples"], 0)
        self.assertIsNone(result["busy_fraction"])
        self.assertFalse(result["busy_pass"])

    def _write_csv(self, path, headers, rows):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            writer.writerows(rows)

    def _write_result(self, path, workload, cuda_visible, checksum=None):
        with open(path, "w") as fh:
            fh.write("workload=%s\n" % workload)
            fh.write("status=PASS\n")
            fh.write("run_tag=user-000@run\n")
            fh.write("cuda_visible_devices=%s\n" % cuda_visible)
            if checksum is not None:
                fh.write("checksum=%s\n" % checksum)

    def _write_journal(self, path):
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "CREATE TABLE events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp REAL NOT NULL, "
                "event_type TEXT NOT NULL, "
                "entity_id TEXT NOT NULL, "
                "state TEXT, "
                "reason TEXT, "
                "details TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO events(timestamp, event_type, entity_id, state, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1.0,
                    "session_state",
                    "sess-000001",
                    "RUNNING",
                    None,
                    json.dumps({"execution_session_id": "sess-000001", "gpu_uuids": ["GPU-a"]}),
                ),
            )
            conn.execute(
                "INSERT INTO events(timestamp, event_type, entity_id, state, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1.1,
                    "request_state",
                    "req-000001",
                    "RUNNING",
                    None,
                    json.dumps({"request_id": "req-000001", "execution_session_id": "sess-000001", "gpu_uuids": ["GPU-a"]}),
                ),
            )
            conn.execute(
                "INSERT INTO events(timestamp, event_type, entity_id, state, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1.2,
                    "session_state",
                    "sess-000001",
                    "CACHED_IDLE",
                    None,
                    json.dumps({"execution_session_id": "sess-000001", "gpu_uuids": [], "gpu_count": 1}),
                ),
            )
            conn.execute(
                "INSERT INTO events(timestamp, event_type, entity_id, state, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1.3,
                    "session_state",
                    "sess-000002",
                    "RUNNING_COMMAND",
                    None,
                    json.dumps({"execution_session_id": "sess-000002", "gpu_uuids": ["GPU-a"]}),
                ),
            )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
