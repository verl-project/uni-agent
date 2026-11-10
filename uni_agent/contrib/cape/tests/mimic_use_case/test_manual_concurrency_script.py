# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


CONCURRENCY_SCRIPT = (
    Path(__file__).resolve().parent / "manual" / "concurrency.sh"
)
CALIBRATED_SCRIPT = (
    Path(__file__).resolve().parent / "calibrated_workload.sh"
)
MANUAL_MAKEFILE = (
    Path(__file__).resolve().parent / "manual" / "Makefile"
)

LEDGER_COLUMNS = (
    "run_id,event,index,request_id,timestamp,delay_seconds,state,exit_code,"
    "label,class,scheduled_offset,submit_offset,node_id,isolation_policy,"
    "runtime_adapter,gpu_uuids,work_start,work_end,cpu_done,gpu_done,"
    "created_at,leased_at,started_at,gpu_mode,lock_wait_start,lock_acquired"
).split(",")


def ledger_rows(ledger_path):
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header == LEDGER_COLUMNS, header
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


class ManualConcurrencyScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.submit_state = self.tmp_path / "submit-count"
        self.status_state = self.tmp_path / "status-counts.json"
        self.submissions = self.tmp_path / "submissions.jsonl"
        self.status_calls = self.tmp_path / "status-calls.jsonl"
        self.events = self.tmp_path / "events.log"
        self.fake_submit = self.tmp_path / "submit.sh"
        self.fake_status = self.tmp_path / "status.sh"

        self.fake_submit.write_text(
            "#!/usr/bin/env python3\n"
            "import fcntl, json, os, pathlib, sys\n"
            "counter = pathlib.Path(os.environ['FAKE_SUBMIT_STATE'])\n"
            "with open(str(counter) + '.lock', 'w') as lock:\n"
            "    fcntl.flock(lock, fcntl.LOCK_EX)\n"
            "    number = int(counter.read_text() if counter.exists() else '0') + 1\n"
            "    counter.write_text(str(number))\n"
            "request_id = 'req-{:06d}'.format(number)\n"
            "record = {\n"
            "    'request_id': request_id,\n"
            "    'args': sys.argv[1:],\n"
            "    'run_tag': os.environ.get('CAPE_SUBMIT_RUN_TAG', ''),\n"
            "}\n"
            "with open(os.environ['FAKE_SUBMISSIONS'], 'a') as stream:\n"
            "    stream.write(json.dumps(record) + '\\n')\n"
            "with open(os.environ['FAKE_EVENTS'], 'a') as stream:\n"
            "    stream.write('submit {}\\n'.format(request_id))\n"
            "print(request_id)\n",
            encoding="utf-8",
        )
        self.fake_status.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "arguments = sys.argv[1:]\n"
            "position = arguments.index('status')\n"
            "request_id = arguments[position + 1]\n"
            "state_path = pathlib.Path(os.environ['FAKE_STATUS_STATE'])\n"
            "counts = json.loads(state_path.read_text()) if state_path.exists() else {}\n"
            "counts[request_id] = counts.get(request_id, 0) + 1\n"
            "state_path.write_text(json.dumps(counts))\n"
            "terminal_poll = int(os.environ.get('FAKE_TERMINAL_POLL', '1'))\n"
            "if counts[request_id] < terminal_poll:\n"
            "    state, exit_code = 'RUNNING', None\n"
            "elif request_id == os.environ.get('FAKE_FAILED_REQUEST'):\n"
            "    state, exit_code = 'FAILED', 7\n"
            "else:\n"
            "    state, exit_code = 'COMPLETED', 0\n"
            "with open(os.environ['FAKE_STATUS_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(arguments) + '\\n')\n"
            "with open(os.environ['FAKE_EVENTS'], 'a') as stream:\n"
            "    stream.write('status {} {}\\n'.format(request_id, state))\n"
            "print(json.dumps({\n"
            "    'request_id': request_id,\n"
            "    'state': state,\n"
            "    'exit_code': exit_code,\n"
            "}))\n",
            encoding="utf-8",
        )
        for script in (self.fake_submit, self.fake_status):
            script.chmod(script.stat().st_mode | stat.S_IXUSR)

        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_") and key not in ("REPO_ROOT", "BUNDLE", "RUN_ID")
        }
        self.env.update({
            "CAPE_CONCURRENCY_SUBMIT_SCRIPT": str(self.fake_submit),
            "CAPE_CONCURRENCY_STATUS_SCRIPT": str(self.fake_status),
            "CAPE_CONCURRENCY_PYTHON": sys.executable,
            "FAKE_SUBMIT_STATE": str(self.submit_state),
            "FAKE_STATUS_STATE": str(self.status_state),
            "FAKE_SUBMISSIONS": str(self.submissions),
            "FAKE_STATUS_CALLS": str(self.status_calls),
            "FAKE_EVENTS": str(self.events),
        })

    def run_concurrency(self, *arguments, env=None, check=True):
        return subprocess.run(
            [str(CONCURRENCY_SCRIPT), *arguments],
            env=env or self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def read_json_lines(self, path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def common_submit_arguments(self):
        return (
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "user-a",
            "--token", "user-token",
            "--gpus", "0",
            "--", "true",
        )

    def write_controlled_submitter(self):
        activity = self.tmp_path / "submit-activity.json"
        self.env["FAKE_SUBMIT_ACTIVITY"] = str(activity)
        self.fake_submit.write_text(
            "#!/usr/bin/env python3\n"
            "import fcntl, json, os, pathlib, sys, time\n"
            "arguments = sys.argv[1:]\n"
            "index = int(next(value.split('=', 1)[1] for value in arguments "
            "if value.startswith('CAPE_CONCURRENCY_INDEX=')))\n"
            "path = pathlib.Path(os.environ['FAKE_SUBMIT_ACTIVITY'])\n"
            "lock_path = pathlib.Path(str(path) + '.lock')\n"
            "def update(starting):\n"
            "    with open(lock_path, 'w') as lock:\n"
            "        fcntl.flock(lock, fcntl.LOCK_EX)\n"
            "        value = json.loads(path.read_text()) if path.exists() else "
            "{'active': 0, 'peak': 0, 'starts': {}}\n"
            "        value['active'] += 1 if starting else -1\n"
            "        value['peak'] = max(value['peak'], value['active'])\n"
            "        if starting:\n"
            "            value['starts'][str(index)] = time.time()\n"
            "        path.write_text(json.dumps(value))\n"
            "update(True)\n"
            "try:\n"
            "    time.sleep(float(os.environ.get('FAKE_SUBMIT_DELAY', '0')))\n"
            "    record = {'request_id': 'req-{:06d}'.format(index), "
            "'args': arguments, 'run_tag': os.environ.get('CAPE_SUBMIT_RUN_TAG', '')}\n"
            "    with open(os.environ['FAKE_SUBMISSIONS'], 'a') as stream:\n"
            "        stream.write(json.dumps(record) + '\\n')\n"
            "    if str(index) == os.environ.get('FAKE_SUBMIT_FAIL_INDEX'):\n"
            "        print('induced submit failure {}'.format(index), file=sys.stderr)\n"
            "        raise SystemExit(9)\n"
            "    print(record['request_id'])\n"
            "finally:\n"
            "    update(False)\n",
            encoding="utf-8",
        )
        self.fake_submit.chmod(self.fake_submit.stat().st_mode | stat.S_IXUSR)
        return activity

    def test_open_loop_schedule_dispatches_slow_submitters_without_serial_delay(self):
        activity = self.write_controlled_submitter()
        report = self.tmp_path / "async-report.txt"
        env = dict(self.env, FAKE_SUBMIT_DELAY="0.15")
        started = time.time()
        result = self.run_concurrency(
            "--workloads", "4",
            "--concurrency", "0",
            "--arrival-schedule", "uniform:0.06",
            "--seed", "4",
            "--submitter-cap", "4",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--report", str(report),
            "--", *self.common_submit_arguments(),
            env=env,
        )
        elapsed = time.time() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(elapsed, 1.5)
        activity_value = json.loads(activity.read_text())
        self.assertGreater(activity_value["peak"], 1)
        starts = [activity_value["starts"][str(index)] for index in range(1, 5)]
        self.assertLess(max(starts) - min(starts), 0.12)
        payload = json.loads((self.tmp_path / "async-report.txt.json").read_text())
        self.assertEqual(payload["schedule"]["fidelity"]["verdict"], "pass")
        self.assertEqual(payload["submission_dispatch"]["dispatched"], 4)
        self.assertEqual(payload["submission_dispatch"]["completed"], 4)
        self.assertEqual(payload["submission_dispatch"]["failed"], 0)
        self.assertEqual(
            [row["index"] for row in payload["requests"]], [1, 2, 3, 4]
        )

    def test_submitter_cap_bounds_workers_and_degrades_missed_schedule(self):
        activity = self.write_controlled_submitter()
        report = self.tmp_path / "cap-report.txt"
        env = dict(self.env, FAKE_SUBMIT_DELAY="0.12")
        result = self.run_concurrency(
            "--workloads", "6", "--concurrency", "0",
            "--arrival-schedule", "uniform:0.01", "--seed", "5",
            "--submitter-cap", "2", "--poll-interval", "0.001",
            "--timeout", "10", "--report", str(report),
            "--", *self.common_submit_arguments(), env=env, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(activity.read_text())["peak"], 2)
        payload = json.loads((self.tmp_path / "cap-report.txt.json").read_text())
        self.assertEqual(payload["run_status"], "degraded")
        self.assertEqual(payload["schedule"]["fidelity"]["verdict"], "fail")
        self.assertEqual(payload["submission_dispatch"]["peak_in_flight"], 2)
        self.assertEqual(payload["submission_dispatch"]["dispatched"], 6)
        self.assertEqual(payload["submission_dispatch"]["completed"], 6)

    def test_async_submission_failure_is_captured_without_losing_other_ids(self):
        self.write_controlled_submitter()
        report = self.tmp_path / "failure-report.txt"
        env = dict(
            self.env,
            FAKE_SUBMIT_DELAY="0.03",
            FAKE_SUBMIT_FAIL_INDEX="2",
        )
        result = self.run_concurrency(
            "--workloads", "3", "--concurrency", "0",
            "--arrival-schedule", "uniform:0.01", "--seed", "6",
            "--submitter-cap", "3",
            "--schedule-skew-threshold-ratio", "100",
            "--poll-interval", "0.001", "--timeout", "10",
            "--report", str(report),
            "--", *self.common_submit_arguments(), env=env, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["req-000001", "req-000003"])
        self.assertIn("submission failed for workload 2", result.stderr)
        payload = json.loads((self.tmp_path / "failure-report.txt.json").read_text())
        self.assertEqual(payload["submission_dispatch"]["dispatched"], 3)
        self.assertEqual(payload["submission_dispatch"]["completed"], 3)
        self.assertEqual(payload["submission_dispatch"]["failed"], 1)
        self.assertEqual(payload["workloads"]["tracked"], 2)

    def test_real_submit_wrapper_phases_are_aggregated(self):
        fake_cape = self.tmp_path / "fake-rscape"
        fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, time\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('rscape 0.1.0'); raise SystemExit(0)\n"
            "time.sleep(0.05)\n"
            "index = next(value.split('=', 1)[1] for value in sys.argv[1:] "
            "if value.startswith('CAPE_CONCURRENCY_INDEX='))\n"
            "print('req-{:06d}'.format(int(index)))\n",
            encoding="utf-8",
        )
        fake_cape.chmod(fake_cape.stat().st_mode | stat.S_IXUSR)
        report = self.tmp_path / "wrapper-report.txt"
        env = dict(
            self.env,
            CAPE_CONCURRENCY_SUBMIT_SCRIPT=str(
                CONCURRENCY_SCRIPT.parent / "submit.sh"
            ),
            CAPE_SUBMIT_BIN=str(fake_cape),
            CAPE_STATUS_BIN=str(fake_cape),
        )
        result = self.run_concurrency(
            "--workloads", "3", "--concurrency", "0",
            "--arrival-schedule", "uniform:0.03", "--seed", "7",
            "--submitter-cap", "3", "--poll-interval", "0.001",
            "--timeout", "10", "--report", str(report),
            "--", *self.common_submit_arguments(), env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads((self.tmp_path / "wrapper-report.txt.json").read_text())
        dispatch = payload["submission_dispatch"]
        self.assertEqual(dispatch["timing_files"], 3)
        self.assertEqual(dispatch["wrapper_phases"]["binary_round_trip"]["count"], 3)
        self.assertGreaterEqual(
            dispatch["wrapper_phases"]["binary_round_trip"]["p50"], 0.04
        )

    def test_caps_in_flight_requests_and_applies_variable_arrival_delays(self):
        ledger = self.tmp_path / "ledger.csv"
        env = dict(self.env, FAKE_TERMINAL_POLL="2")
        result = self.run_concurrency(
            "--workloads", "5",
            "--concurrency", "2",
            "--arrival-window", "0.001:0.004",
            "--arrival-distribution", "uniform",
            "--seed", "1234",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--ledger", str(ledger),
            "--",
            *self.common_submit_arguments(),
            env=env,
        )

        self.assertEqual(
            result.stdout.splitlines(),
            ["req-{:06d}".format(index) for index in range(1, 6)],
        )
        self.assertIn("submitted=5/5", result.stderr)
        self.assertIn("succeeded=5", result.stderr)
        self.assertIn("max_in_flight=2", result.stderr)

        rows = [line.split(",") for line in ledger.read_text(encoding="utf-8").splitlines()[1:]]
        delays = [float(row[5]) for row in rows if row[1] == "SUBMITTED"]
        self.assertEqual(len(delays), 5)
        self.assertTrue(all(0.001 <= delay <= 0.004 for delay in delays))
        self.assertGreater(len(set(delays)), 1)

        submissions = self.read_json_lines(self.submissions)
        for index, record in enumerate(submissions, 1):
            arguments = record["args"]
            self.assertIn("CAPE_CONCURRENCY_INDEX={}".format(index), arguments)
            self.assertIn("CAPE_CONCURRENCY_TOTAL=5", arguments)
            label_position = arguments.index("--label")
            self.assertEqual(arguments[label_position + 1], "workload-{:06d}".format(index))

        calls = self.read_json_lines(self.status_calls)
        self.assertTrue(calls)
        self.assertTrue(all("--controller-url" in call for call in calls))
        self.assertTrue(all("--token" in call for call in calls))

        events = self.events.read_text(encoding="utf-8").splitlines()
        third_submit = events.index("submit req-000003")
        self.assertTrue(any(
            event.endswith("COMPLETED") for event in events[:third_submit]
        ))

    def test_injects_unique_session_ids_unless_the_operator_supplies_one(self):
        env = dict(self.env, CAPE_CONCURRENCY_RUN_ID="execution-session-identity-run")
        result = self.run_concurrency(
            "--workloads", "3",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            *self.common_submit_arguments(),
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        observed = []
        for record in self.read_json_lines(self.submissions):
            arguments = record["args"]
            positions = [
                index
                for index, value in enumerate(arguments)
                if value == "--session-id"
            ]
            self.assertEqual(len(positions), 1)
            observed.append(arguments[positions[0] + 1])
        self.assertEqual(observed, [
            "execution-session-identity-run-000001",
            "execution-session-identity-run-000002",
            "execution-session-identity-run-000003",
        ])

        self.submissions.unlink()
        self.submit_state.unlink()
        self.status_state.unlink()
        explicit = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "user-a",
            "--token", "user-token",
            "--gpus", "0",
            "--session-id", "operator-shared-key",
            "--", "true",
            env=env,
        )
        self.assertEqual(explicit.returncode, 0)
        for record in self.read_json_lines(self.submissions):
            arguments = record["args"]
            positions = [
                index
                for index, value in enumerate(arguments)
                if value == "--session-id"
            ]
            self.assertEqual(len(positions), 1)
            self.assertEqual(
                arguments[positions[0] + 1], "operator-shared-key"
            )

    def test_repeated_mimic_requests_receive_unique_work_roots_and_tags(self):
        work_root = self.tmp_path / "shared-work"
        result = self.run_concurrency(
            "--workloads", "3",
            "--concurrency", "2",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--work-root", str(work_root),
            "--label-prefix", "gpu-job",
            "--",
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "user-a",
            "--token", "user-token",
            "--mimic", "A",
        )
        self.assertEqual(result.returncode, 0)

        submissions = self.read_json_lines(self.submissions)
        self.assertEqual(len(submissions), 3)
        roots = []
        tags = []
        for index, record in enumerate(submissions, 1):
            arguments = record["args"]
            root_position = len(arguments) - 1 - arguments[::-1].index("--user-root")
            roots.append(arguments[root_position + 1])
            tags.append(record["run_tag"])
            self.assertEqual(
                arguments[root_position + 1],
                str(work_root / "workload-{:06d}".format(index)),
            )
        self.assertEqual(len(set(roots)), 3)
        self.assertEqual(len(set(tags)), 3)

    def test_controller_bundle_is_not_reused_as_the_local_mimic_root(self):
        controller_env = self.tmp_path / "controller.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n"
            "export RUN_ID=controller-run\n"
            "export CASE=manual\n"
            "export BUNDLE=/foreign/controller/results/controller-run\n",
            encoding="utf-8",
        )
        local_bundle = self.tmp_path / "submitter-bundle"
        env = dict(
            self.env,
            CAPE_CONCURRENCY_BUNDLE=str(local_bundle),
            CAPE_CONCURRENCY_RUN_ID="portable-run",
        )

        result = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            "--controller-env", str(controller_env),
            "--user", "user-a",
            "--mimic", "A",
            env=env,
        )
        self.assertEqual(result.returncode, 0)

        roots = []
        for record in self.read_json_lines(self.submissions):
            arguments = record["args"]
            position = len(arguments) - 1 - arguments[::-1].index("--user-root")
            roots.append(arguments[position + 1])
        expected = local_bundle / "concurrency" / "portable-run"
        self.assertEqual(roots, [
            str(expected / "workload-000001"),
            str(expected / "workload-000002"),
        ])
        self.assertTrue(all("/foreign/controller" not in root for root in roots))

    def test_polls_with_inline_submitter_token_instead_of_admin_token(self):
        controller_env = self.tmp_path / "controller-inline.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n"
            "export CAPE_USER_TOKENS_INLINE=$'user-a,user-token\\nuser-b,other-token'\n",
            encoding="utf-8",
        )
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            "--controller-env", str(controller_env),
            "--user", "user-a",
            "--gpus", "0",
            "--", "true",
        )
        self.assertEqual(result.returncode, 0)
        calls = self.read_json_lines(self.status_calls)
        self.assertTrue(calls)
        self.assertTrue(all("--token" in call for call in calls))
        self.assertTrue(all(call[call.index("--token") + 1] == "user-token" for call in calls))
        self.assertTrue(all("admin-token" not in call for call in calls))

    def test_passed_handoff_replaces_a_stale_generic_controller_url(self):
        controller_env = self.tmp_path / "controller-fresh.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://fresh-controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_USER_TOKENS_INLINE=$'user-a,user-token'\n",
            encoding="utf-8",
        )
        env = dict(
            self.env,
            CAPE_URL="http://stale-controller.example:18892",
        )
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            "--controller-env", str(controller_env),
            "--user", "user-a",
            "--gpus", "0",
            "--", "true",
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        calls = self.read_json_lines(self.status_calls)
        self.assertTrue(calls)
        for call in calls:
            position = call.index("--controller-url")
            self.assertEqual(
                call[position + 1], "http://fresh-controller.example:18892"
            )

    def test_no_token_submission_keeps_status_polling_unauthenticated(self):
        controller_env = self.tmp_path / "controller-admin.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n",
            encoding="utf-8",
        )
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            "--controller-env", str(controller_env),
            "--user", "user-a",
            "--no-token",
            "--gpus", "0",
            "--", "true",
        )
        self.assertEqual(result.returncode, 0)
        calls = self.read_json_lines(self.status_calls)
        self.assertTrue(calls)
        self.assertTrue(all("--no-token" in call for call in calls))
        self.assertTrue(all("--token" not in call for call in calls))

    def test_waits_for_all_requests_but_returns_failure_for_failed_workload(self):
        env = dict(self.env, FAKE_FAILED_REQUEST="req-000002")
        result = self.run_concurrency(
            "--workloads", "3",
            "--concurrency", "2",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            *self.common_submit_arguments(),
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(result.stdout.splitlines()), 3)
        self.assertIn("terminal=3", result.stderr)
        self.assertIn("succeeded=2", result.stderr)
        self.assertIn("failed=1", result.stderr)
        self.assertNotIn("requests still active", result.stderr)

    def write_rich_fake_status(self, with_markers=True):
        """Fake status.sh handling status (with placement), logs, utilization."""
        self.fake_status.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "arguments = sys.argv[1:]\n"
            "if 'utilization' in arguments:\n"
            "    print(json.dumps({'schedulable_gpus': 4, 'active_nodes': 2}))\n"
            "    raise SystemExit(0)\n"
            "if 'logs' in arguments:\n"
            "    request_id = arguments[arguments.index('logs') + 1]\n"
            "    marker_mode = os.environ.get('FAKE_MARKER_MODE', 'full')\n"
            "    if marker_mode == 'none':\n"
            "        raise SystemExit(1)\n"
            "    import time\n"
            "    number = int(request_id.split('-')[1])\n"
            "    start = time.time() + number / 1000.0\n"
            "    finish_seconds = float(os.environ.get('FAKE_FINISH_SECONDS', '0'))\n"
            "    print('hello workload')\n"
            "    print('CAPE_CALIBRATED work_start {:.3f}'.format(start))\n"
            "    if marker_mode in ('full', 'locked', 'bad-lock'):\n"
            "        print('CAPE_CALIBRATED cpu_done {:.3f}'.format(start + 1.0))\n"
            "        print('CAPE_CALIBRATED gpu_done {:.3f}'.format(start + 2.0))\n"
            "    if marker_mode in ('locked', 'bad-lock'):\n"
            "        print('CAPE_CALIBRATED lock_wait_start {:.3f}'.format(start + 1.1))\n"
            "        print('CAPE_CALIBRATED lock_acquired {:.3f}'.format(start + (1.6 if marker_mode == 'locked' else 2.5)))\n"
            "    print('CAPE_CALIBRATED work_end {:.3f}'.format(\n"
            "        start + 2.0 + finish_seconds))\n"
            "    raise SystemExit(0)\n"
            "position = arguments.index('status')\n"
            "request_id = arguments[position + 1]\n"
            "state_path = pathlib.Path(os.environ['FAKE_STATUS_STATE'])\n"
            "counts = json.loads(state_path.read_text()) if state_path.exists() else {}\n"
            "counts[request_id] = counts.get(request_id, 0) + 1\n"
            "state_path.write_text(json.dumps(counts))\n"
            "terminal_poll = int(os.environ.get('FAKE_TERMINAL_POLL', '1'))\n"
            "if counts[request_id] < terminal_poll:\n"
            "    state, exit_code = 'RUNNING', None\n"
            "else:\n"
            "    state, exit_code = 'COMPLETED', 0\n"
            "with open(os.environ['FAKE_STATUS_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(arguments) + '\\n')\n"
            "number = int(request_id.split('-')[1])\n"
            "now = __import__('time').time()\n"
            "print(json.dumps({\n"
            "    'request_id': request_id,\n"
            "    'state': state,\n"
            "    'exit_code': exit_code,\n"
            "    'node_id': 'node-{}'.format(number % 2),\n"
            "    'isolation_policy': 'image-stateless' if number % 2 else 'host-stateless',\n"
            "    'runtime_adapter': 'apptainer',\n"
            "    'gpu_uuids': ['GPU-{}a'.format(number), 'GPU-{}b'.format(number)],\n"
            "    'created_at': now - 0.030,\n"
            "    'leased_at': now - 0.020,\n"
            "    'started_at': now - 0.010,\n"
            "}))\n",
            encoding="utf-8",
        )
        self.fake_status.chmod(self.fake_status.stat().st_mode | stat.S_IXUSR)
        self.env["FAKE_MARKER_MODE"] = "full" if with_markers else "none"

    def test_arrival_schedule_records_offsets_and_skew(self):
        ledger = self.tmp_path / "ledger.csv"
        result = self.run_concurrency(
            "--workloads", "4",
            "--concurrency", "0",
            "--arrival-schedule", "uniform:0.02",
            "--seed", "77",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--ledger", str(ledger),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(len(result.stdout.splitlines()), 4)
        rows = ledger_rows(ledger)
        submitted = [row for row in rows if row["event"] == "SUBMITTED"]
        self.assertEqual(len(submitted), 4)
        offsets = [float(row["scheduled_offset"]) for row in submitted]
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(0.0 <= value <= 0.02 for value in offsets))
        for row in submitted:
            self.assertTrue(row["submit_offset"])
            self.assertGreaterEqual(
                float(row["submit_offset"]), float(row["scheduled_offset"]) - 0.001
            )
        # Same seed must reproduce the same schedule.
        rerun_ledger = self.tmp_path / "ledger-rerun.csv"
        self.run_concurrency(
            "--workloads", "4",
            "--concurrency", "0",
            "--arrival-schedule", "uniform:0.02",
            "--seed", "77",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--ledger", str(rerun_ledger),
            "--",
            *self.common_submit_arguments(),
        )
        rerun_offsets = [
            float(row["scheduled_offset"])
            for row in ledger_rows(rerun_ledger)
            if row["event"] == "SUBMITTED"
        ]
        self.assertEqual(offsets, rerun_offsets)

    def test_open_loop_submits_everything_without_waiting_for_completions(self):
        env = dict(self.env, FAKE_TERMINAL_POLL="3")
        result = self.run_concurrency(
            "--workloads", "6",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            *self.common_submit_arguments(),
            env=env,
        )
        self.assertEqual(len(result.stdout.splitlines()), 6)
        self.assertIn("max_in_flight=6", result.stderr)
        events = self.events.read_text(encoding="utf-8").splitlines()
        last_submit = max(
            position for position, event in enumerate(events)
            if event.startswith("submit ")
        )
        completions_before_last_submit = [
            event for event in events[:last_submit] if event.endswith("COMPLETED")
        ]
        self.assertEqual(completions_before_last_submit, [])

    def test_classes_merge_args_env_label_and_ledger_column(self):
        ledger = self.tmp_path / "ledger.csv"
        result = self.run_concurrency(
            "--workloads", "4",
            "--concurrency", "2",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--ledger", str(ledger),
            "--class", "host:1:--isolation-policy host-stateless",
            "--class", "image:1:--isolation-policy image-stateful --gpus 2",
            "--mix-order", "round-robin",
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        submissions = self.read_json_lines(self.submissions)
        self.assertEqual(len(submissions), 4)
        expected_classes = ["host", "image", "host", "image"]
        for index, (record, expected) in enumerate(
            zip(submissions, expected_classes), 1
        ):
            arguments = record["args"]
            self.assertIn("CAPE_CONCURRENCY_CLASS={}".format(expected), arguments)
            policy_position = len(arguments) - 1 - arguments[::-1].index("--isolation-policy")
            expected_policy = (
                "host-stateless" if expected == "host" else "image-stateful"
            )
            self.assertEqual(arguments[policy_position + 1], expected_policy)
            label_position = arguments.index("--label")
            self.assertEqual(
                arguments[label_position + 1],
                "workload-{}-{:06d}".format(expected, index),
            )
            if expected == "image":
                gpus_position = len(arguments) - 1 - arguments[::-1].index("--gpus")
                self.assertEqual(arguments[gpus_position + 1], "2")
            command_position = arguments.index("--")
            self.assertEqual(arguments[command_position + 1 :], ["true"])
        classes = [
            row["class"]
            for row in ledger_rows(ledger)
            if row["event"] == "SUBMITTED"
        ]
        self.assertEqual(classes, expected_classes)

    def test_class_can_route_a_workload_to_a_shared_pool(self):
        result = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "2",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--class", "shared:1:--pool shared-pool",
            "--class", "exclusive:1:--pool exclusive-pool",
            "--mix-order", "round-robin",
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        submissions = self.read_json_lines(self.submissions)
        expected = ("shared-pool", "exclusive-pool")
        for record, pool in zip(submissions, expected):
            arguments = record["args"]
            positions = [
                index
                for index, value in enumerate(arguments)
                if value == "--pool"
            ]
            self.assertGreaterEqual(len(positions), 2)
            self.assertEqual(arguments[positions[-1] + 1], pool)

    def test_class_user_polls_with_that_users_private_token_file(self):
        own = self.tmp_path / "alice.json"
        other = self.tmp_path / "bob.json"
        for file, token in ((own, "token-a"), (other, "token-b")):
            file.write_text(json.dumps({"secret": token}))
            file.chmod(0o600)
        result = self.run_concurrency(
            "--workloads", "2", "--concurrency", "2", "--arrival-window", "0:0",
            "--poll-interval", "0.001", "--timeout", "10",
            "--class", "own:1:",
            "--class", "other:1:--user user-b --token-file " + str(other),
            "--", "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a", "--token-file", str(own),
            "--user", "user-a", "--gpus", "0", "--", "true",
        )
        self.assertEqual(result.returncode, 0)
        tokens = {}
        for call in self.read_json_lines(self.status_calls):
            tokens[call[call.index("status") + 1]] = call[call.index("--token-file") + 1]
            self.assertNotIn("--token", call)
        self.assertEqual(tokens["req-000001"], str(own))
        self.assertEqual(tokens["req-000002"], str(other))

    def test_mix_file_supports_values_with_spaces_and_command_override(self):
        mix = self.tmp_path / "mix.jsonl"
        mix.write_text(
            json.dumps({
                "name": "calibrated",
                "weight": 1,
                "args": ["--env", "NOTE=two words here"],
                "command": ["bash", "-c", "echo done"],
            }) + "\n",
            encoding="utf-8",
        )
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--mix-file", str(mix),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        record = self.read_json_lines(self.submissions)[0]
        arguments = record["args"]
        self.assertIn("NOTE=two words here", arguments)
        command_position = arguments.index("--")
        self.assertEqual(
            arguments[command_position + 1 :], ["bash", "-c", "echo done"]
        )

    def test_placement_fields_are_recorded_on_terminal_rows(self):
        self.write_rich_fake_status()
        ledger = self.tmp_path / "ledger.csv"
        result = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "2",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--ledger", str(ledger),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        terminal = {
            row["request_id"]: row
            for row in ledger_rows(ledger)
            if row["event"] == "TERMINAL"
        }
        self.assertEqual(len(terminal), 2)
        row = terminal["req-000001"]
        self.assertEqual(row["node_id"], "node-1")
        self.assertEqual(row["isolation_policy"], "image-stateless")
        self.assertEqual(row["runtime_adapter"], "apptainer")
        self.assertEqual(row["gpu_uuids"], "GPU-1a;GPU-1b")

    def test_lock_wait_trace_is_separate_and_contiguous(self):
        for mode in ('locked', 'bad-lock'):
            with self.subTest(mode=mode):
                self.write_rich_fake_status(with_markers=True)
                report = self.tmp_path / (mode + '.txt')
                self.run_concurrency('--workloads', '1', '--concurrency', '1',
                    '--arrival-window', '0:0', '--poll-interval', '0.001', '--timeout', '10',
                    '--report', str(report), '--', *self.common_submit_arguments(),
                    env=dict(self.env, FAKE_MARKER_MODE=mode))
                payload = json.loads(Path(str(report) + '.json').read_text())
                row = payload['requests'][0]
                events = json.loads(Path(str(report) + '.trace.json').read_text())['traceEvents']
                slices = [e for e in events if e['ph'] == 'X' and e['pid'] == 1
                          and e['name'].startswith('processing:')]
                names = [e['name'] for e in slices]
                self.assertAlmostEqual(row['gpu_time'], 1, places=3)
                if mode == 'locked':
                    self.assertEqual(names, ['processing:cpu', 'processing:prepare',
                                             'processing:lock-wait', 'processing:gpu'])
                    self.assertAlmostEqual(row['gpu_prepare_time'], .1, places=3)
                    self.assertAlmostEqual(row['lock_wait_time'], .5, places=3)
                    self.assertAlmostEqual(row['gpu_after_lock_time'], .4, places=3)
                    for left, right in zip(slices, slices[1:]):
                        self.assertEqual(left['ts'] + left['dur'], right['ts'])
                    self.assertEqual(sum(e['dur'] for e in slices), 2000000)
                else:
                    self.assertEqual(names, ['processing:cpu', 'processing:gpu'])
                    self.assertIsNone(row['lock_wait_time'])
                    self.assertTrue(any('lock markers' in a for a in payload['anomalies']))

    def test_previous_ledger_gets_optional_columns_without_losing_rows(self):
        ledger = self.tmp_path / 'old.csv'
        old_header = LEDGER_COLUMNS[:-2]
        row = ['old-run', 'SUBMITTED'] + [''] * (len(old_header) - 2)
        ledger.write_text(','.join(old_header) + '\n' + ','.join(row) + '\n')
        self.run_concurrency('--workloads', '1', '--concurrency', '1', '--arrival-window', '0:0',
            '--poll-interval', '0.001', '--timeout', '10', '--ledger', str(ledger),
            '--', *self.common_submit_arguments())
        rows = ledger_rows(ledger)
        self.assertEqual(rows[0]['run_id'], 'old-run')
        self.assertEqual(rows[0]['lock_wait_start'], '')
        self.assertEqual(rows[0]['lock_acquired'], '')

    def test_report_uses_markers_for_latency_decomposition(self):
        self.write_rich_fake_status(with_markers=True)
        report = self.tmp_path / "report.txt"
        result = self.run_concurrency(
            "--workloads", "3",
            "--concurrency", "0",
            "--arrival-schedule", "uniform:0.01",
            "--seed", "9",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--report", str(report),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(report.exists())
        payload = json.loads((self.tmp_path / "report.txt.json").read_text())
        self.assertEqual(payload["workloads"]["completed"], 3)
        self.assertEqual(payload["measurement"]["marker_requests"], 3)
        self.assertEqual(payload["measurement"]["fallback_requests"], 0)
        for entry in payload["requests"]:
            self.assertEqual(entry["measurement_path"], "marker")
            self.assertAlmostEqual(entry["processing_time"], 2.0, places=3)
            self.assertAlmostEqual(entry["cpu_time"], 1.0, places=3)
            self.assertAlmostEqual(entry["gpu_time"], 1.0, places=3)
            self.assertIsNotNone(entry["queuing_time"])
            self.assertIsNotNone(entry["client_submit_overhead"])
            self.assertAlmostEqual(entry["true_queuing_time"], 0.010, places=2)
            self.assertAlmostEqual(entry["session_assembly_time"], 0.010, places=2)
            self.assertIsNotNone(entry["agent_handoff_time"])
        self.assertAlmostEqual(payload["latency"]["cpu_time"]["mean"], 1.0)
        self.assertAlmostEqual(payload["latency"]["gpu_time"]["mean"], 1.0)
        self.assertEqual(
            payload["measurement"]["warm_reuse"],
            "disabled_unique_session_id",
        )
        self.assertAlmostEqual(
            payload["by_class"]["(none)"]["cpu_time"]["mean"], 1.0
        )
        self.assertAlmostEqual(
            payload["by_class"]["(none)"]["gpu_time"]["mean"], 1.0
        )
        self.assertIn("by_isolation_policy", payload)
        self.assertIn("host-stateless", payload["by_isolation_policy"])
        meta = json.loads((self.tmp_path / "report.txt.meta.json").read_text())
        self.assertEqual(meta["schedule_mode"], "uniform")
        snapshot = json.loads(
            (self.tmp_path / "report.txt.utilization-start.json").read_text()
        )
        self.assertEqual(snapshot["schedulable_gpus"], 4)
        text = report.read_text(encoding="utf-8")
        self.assertIn("throughput:", text)
        self.assertIn("latency decomposition", text)

        trace = json.loads((self.tmp_path / "report.txt.trace.json").read_text())
        events = trace["traceEvents"]
        process_names = {
            event["args"]["name"]
            for event in events
            if event["ph"] == "M" and event["name"] == "process_name"
        }
        # Fake status reports two GPU UUIDs per request (GPU-Na, GPU-Nb).
        self.assertTrue(any(name.startswith("GPU GPU-") for name in process_names))
        slice_names = [event["name"] for event in events if event["ph"] == "X"]
        self.assertIn("queuing", slice_names)
        self.assertIn("processing:cpu", slice_names)
        self.assertIn("processing:gpu", slice_names)
        self.assertNotIn("processing", slice_names)
        self.assertNotIn("processing:finish", slice_names)
        cpu_slices = [
            event for event in events
            if event["ph"] == "X" and event["name"] == "processing:cpu"
        ]
        gpu_slices = [
            event for event in events
            if event["ph"] == "X" and event["name"] == "processing:gpu"
        ]
        self.assertEqual(len(cpu_slices), len(gpu_slices))
        for event in cpu_slices + gpu_slices:
            self.assertAlmostEqual(event["dur"], 1.0e6, delta=1e3)
            self.assertIn("request_id", event["args"])
            self.assertEqual(event["args"]["phase"], event["name"])
        cpu_by_track = {
            (event["pid"], event["tid"]): event for event in cpu_slices
        }
        gpu_by_track = {
            (event["pid"], event["tid"]): event for event in gpu_slices
        }
        self.assertEqual(cpu_by_track.keys(), gpu_by_track.keys())
        for track, cpu_event in cpu_by_track.items():
            gpu_event = gpu_by_track[track]
            # Simulate Perfetto's independent microsecond-to-nanosecond
            # rounding: the shared integer-us boundary must remain adjacent.
            cpu_end_ns = (
                round(cpu_event["ts"] * 1000)
                + round(cpu_event["dur"] * 1000)
            )
            self.assertEqual(cpu_end_ns, round(gpu_event["ts"] * 1000))
        # Each request occupies its own row (thread) under each of its GPUs:
        # the fake status reports two request-unique UUIDs per request, so
        # 3 workloads produce 6 GPU track groups with one request row each.
        thread_rows = {
            (event["pid"], event["tid"])
            for event in events
            if event["ph"] == "M" and event["name"] == "thread_name"
        }
        self.assertEqual(len(process_names), 6)
        self.assertEqual(len(thread_rows), 6)

    def test_trace_keeps_single_processing_slice_for_two_marker_workload(self):
        self.write_rich_fake_status(with_markers=True)
        self.env["FAKE_MARKER_MODE"] = "bounds-only"
        report = self.tmp_path / "report.txt"
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--report", str(report),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads((self.tmp_path / "report.txt.json").read_text())
        entry = payload["requests"][0]
        self.assertEqual(entry["measurement_path"], "marker")
        self.assertAlmostEqual(entry["processing_time"], 2.0, places=3)
        self.assertIsNone(entry["cpu_time"])
        self.assertIsNone(entry["gpu_time"])
        terminal = next(
            row
            for row in ledger_rows(self.tmp_path / "report.txt.ledger.csv")
            if row["event"] == "TERMINAL"
        )
        self.assertEqual(terminal["cpu_done"], "")
        self.assertEqual(terminal["gpu_done"], "")
        trace = json.loads((self.tmp_path / "report.txt.trace.json").read_text())
        names = [
            event["name"] for event in trace["traceEvents"] if event["ph"] == "X"
        ]
        self.assertIn("processing", names)
        self.assertNotIn("processing:cpu", names)
        self.assertNotIn("processing:gpu", names)

    def test_trace_emits_only_positive_processing_finish_slice(self):
        self.write_rich_fake_status(with_markers=True)
        self.env["FAKE_FINISH_SECONDS"] = "0.125"
        report = self.tmp_path / "report.txt"
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--report", str(report),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        trace = json.loads((self.tmp_path / "report.txt.trace.json").read_text())
        finishes = [
            event
            for event in trace["traceEvents"]
            if event["ph"] == "X" and event["name"] == "processing:finish"
        ]
        self.assertTrue(finishes)
        for event in finishes:
            self.assertAlmostEqual(event["dur"], 125000, delta=1000)
            self.assertEqual(event["args"]["phase"], "processing:finish")

    def test_report_falls_back_to_processing_constant_without_markers(self):
        self.write_rich_fake_status(with_markers=False)
        report = self.tmp_path / "report.txt"
        result = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--processing-constant", "0.001",
            "--report", str(report),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads((self.tmp_path / "report.txt.json").read_text())
        self.assertEqual(payload["measurement"]["marker_requests"], 0)
        self.assertEqual(payload["measurement"]["fallback_requests"], 2)
        for entry in payload["requests"]:
            self.assertEqual(entry["measurement_path"], "fallback")
            self.assertAlmostEqual(entry["processing_time"], 0.001)
            self.assertIsNone(entry["cpu_time"])
            self.assertIsNone(entry["gpu_time"])
            self.assertAlmostEqual(
                entry["queuing_time"],
                entry["elapsed"] - entry["arrival_delay"] - 0.001,
                places=6,
            )
        self.assertIsNone(payload["latency"]["cpu_time"])
        self.assertIsNone(payload["latency"]["gpu_time"])
        trace = json.loads((self.tmp_path / "report.txt.trace.json").read_text())
        names = [
            event["name"] for event in trace["traceEvents"] if event["ph"] == "X"
        ]
        self.assertIn("request (fallback: submit->terminal)", names)
        self.assertNotIn("processing:cpu", names)
        self.assertNotIn("processing:gpu", names)

    def test_shared_client_is_resolved_once_and_exported_to_children(self):
        client = self.tmp_path / "fake-cape"
        client.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        client.chmod(client.stat().st_mode | stat.S_IXUSR)
        # A submit fake that advertises --resolve-client support and records
        # the CAPE_SUBMIT_BIN each real submission observed.
        self.fake_submit.write_text(
            "#!/usr/bin/env python3\n"
            "# supports --resolve-client\n"
            "import json, os, pathlib, sys\n"
            "if '--resolve-client' in sys.argv:\n"
            "    with open(os.environ['FAKE_EVENTS'], 'a') as stream:\n"
            "        stream.write('resolve-client\\n')\n"
            "    print(os.environ['FAKE_CLIENT_PATH'])\n"
            "    raise SystemExit(0)\n"
            "counter = pathlib.Path(os.environ['FAKE_SUBMIT_STATE'])\n"
            "number = int(counter.read_text() if counter.exists() else '0') + 1\n"
            "counter.write_text(str(number))\n"
            "request_id = 'req-{:06d}'.format(number)\n"
            "record = {\n"
            "    'request_id': request_id,\n"
            "    'args': sys.argv[1:],\n"
            "    'submit_bin': os.environ.get('CAPE_SUBMIT_BIN', ''),\n"
            "}\n"
            "with open(os.environ['FAKE_SUBMISSIONS'], 'a') as stream:\n"
            "    stream.write(json.dumps(record) + '\\n')\n"
            "with open(os.environ['FAKE_EVENTS'], 'a') as stream:\n"
            "    stream.write('submit {}\\n'.format(request_id))\n"
            "print(request_id)\n",
            encoding="utf-8",
        )
        self.fake_submit.chmod(self.fake_submit.stat().st_mode | stat.S_IXUSR)
        # A status fake that records the CAPE_STATUS_BIN it observed.
        self.fake_status.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "arguments = sys.argv[1:]\n"
            "request_id = arguments[arguments.index('status') + 1]\n"
            "with open(os.environ['FAKE_STATUS_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps({\n"
            "        'args': arguments,\n"
            "        'status_bin': os.environ.get('CAPE_STATUS_BIN', ''),\n"
            "    }) + '\\n')\n"
            "print(json.dumps({'request_id': request_id, 'state': 'COMPLETED',\n"
            "                  'exit_code': 0}))\n",
            encoding="utf-8",
        )
        self.fake_status.chmod(self.fake_status.stat().st_mode | stat.S_IXUSR)

        env = dict(self.env, FAKE_CLIENT_PATH=str(client))
        result = self.run_concurrency(
            "--workloads", "3",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            *self.common_submit_arguments(),
            env=env,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["req-{:06d}".format(index) for index in range(1, 4)],
            "the resolve call must not consume request ids",
        )
        events = self.events.read_text(encoding="utf-8").splitlines()
        self.assertEqual(events.count("resolve-client"), 1)
        submissions = self.read_json_lines(self.submissions)
        self.assertEqual(len(submissions), 3)
        for record in submissions:
            self.assertEqual(record["submit_bin"], str(client))
        for call in self.read_json_lines(self.status_calls):
            self.assertEqual(call["status_bin"], str(client))

    def test_preset_submit_bin_skips_resolution_and_is_shared(self):
        client = self.tmp_path / "preset-cape"
        client.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        client.chmod(client.stat().st_mode | stat.S_IXUSR)
        self.fake_status.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "arguments = sys.argv[1:]\n"
            "request_id = arguments[arguments.index('status') + 1]\n"
            "with open(os.environ['FAKE_STATUS_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps({\n"
            "        'status_bin': os.environ.get('CAPE_STATUS_BIN', ''),\n"
            "    }) + '\\n')\n"
            "print(json.dumps({'request_id': request_id, 'state': 'COMPLETED',\n"
            "                  'exit_code': 0}))\n",
            encoding="utf-8",
        )
        self.fake_status.chmod(self.fake_status.stat().st_mode | stat.S_IXUSR)
        env = dict(self.env, CAPE_SUBMIT_BIN=str(client))
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--",
            *self.common_submit_arguments(),
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("resolve-client", self.events.read_text(encoding="utf-8"))
        for call in self.read_json_lines(self.status_calls):
            self.assertEqual(call["status_bin"], str(client))

    def test_report_without_constant_reports_elapsed_only_for_marker_less_requests(self):
        self.write_rich_fake_status(with_markers=False)
        report = self.tmp_path / "report.txt"
        result = self.run_concurrency(
            "--workloads", "2",
            "--concurrency", "0",
            "--arrival-window", "0:0",
            "--poll-interval", "0.001",
            "--timeout", "10",
            "--report", str(report),
            "--",
            *self.common_submit_arguments(),
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads((self.tmp_path / "report.txt.json").read_text())
        self.assertIsNone(payload["measurement"]["processing_constant"])
        self.assertEqual(payload["measurement"]["elapsed_only_requests"], 2)
        self.assertEqual(payload["measurement"]["fallback_requests"], 0)
        for entry in payload["requests"]:
            self.assertEqual(entry["measurement_path"], "elapsed-only")
            self.assertIsNone(entry["queuing_time"])
            self.assertIsNone(entry["processing_time"])
            self.assertIsNotNone(entry["elapsed"])
        deviation_anomalies = [
            anomaly for anomaly in payload["anomalies"]
            if "deviates" in anomaly
        ]
        self.assertEqual(deviation_anomalies, [])

    def test_whole_run_timeout_bounds_the_pre_submission_delay(self):
        started = time.monotonic()
        result = self.run_concurrency(
            "--workloads", "1",
            "--concurrency", "1",
            "--arrival-window", "5:5",
            "--timeout", "1",
            "--",
            *self.common_submit_arguments(),
            check=False,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 1)
        self.assertLess(elapsed, 3)
        self.assertEqual(result.stdout, "")
        self.assertIn("timeout reached after 1s", result.stderr)
        self.assertIn("submitted=0/1", result.stderr)


class CalibratedWorkloadScriptTests(unittest.TestCase):
    def test_emits_ordered_markers_with_gpu_override(self):
        result = subprocess.run(
            [str(CALIBRATED_SCRIPT), "--cpu-seconds", "0.05", "--gpu-seconds", "0.05"],
            env=dict(os.environ, CAPE_CALIBRATED_GPU_CMD='sleep "$GPU_SLEEP_SECONDS"'),
            check=True,
            text=True,
            capture_output=True,
        )
        markers = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts[0] == "CAPE_CALIBRATED" and len(parts) >= 3:
                markers.setdefault(parts[1], parts[2])
        for name in ("work_start", "cpu_done", "gpu_done", "work_end"):
            self.assertIn(name, markers)
        self.assertLessEqual(float(markers["work_start"]), float(markers["cpu_done"]))
        self.assertLessEqual(float(markers["cpu_done"]), float(markers["gpu_done"]))
        self.assertLessEqual(float(markers["gpu_done"]), float(markers["work_end"]))
        self.assertGreaterEqual(
            float(markers["work_end"]) - float(markers["work_start"]), 0.1
        )
        self.assertIn("result ok", result.stdout)

    def test_fails_without_gpu_toolchain_instead_of_passing_silently(self):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_CALIBRATED")
        }
        environment["PATH"] = "/usr/bin:/bin"
        result = subprocess.run(
            [str(CALIBRATED_SCRIPT), "--cpu-seconds", "0",
             "--cache-dir", tempfile.mkdtemp()],
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            self.skipTest("host unexpectedly provides a CUDA toolchain")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nvcc", result.stderr)


class ManualConcurrencyMakefileTests(unittest.TestCase):
    def test_targets_use_explicit_url_and_individual_credential(self):
        for target in ("controller", "user", "register_exclusive", "register_shared",
                       "exclusive_host", "shared_host", "exclusive_mix", "shared_mix"):
            result = self.make("-n", target, "CAPE_URL=http://scheduler:18892",
                               "CAPE_USER=alice", "CAPE_USER_TOKEN_FILE=/private/alice.json",
                               check=True)
            self.assertNotIn("--controller-env", result.stdout)
            self.assertNotIn("handoff", result.stdout.lower())
            if target not in ("controller", "user"):
                self.assertIn("/private/alice.json", result.stdout)
            if target.endswith(("_host", "_mix")):
                self.assertIn("--controller-url", result.stdout)
                self.assertIn('--user "alice"', result.stdout)
                self.assertIn("--workspace-target /cape-workspace", result.stdout)

    def test_clean_preserves_persistent_instance_and_credentials(self):
        result = self.make("-n", "clean", check=True)
        self.assertNotIn("evidence/", result.stdout)

    def make(self, *arguments, check=False):
        return subprocess.run(
            ["make", "-C", str(MANUAL_MAKEFILE.parent), *arguments],
            check=check,
            text=True,
            capture_output=True,
        )

    def test_processing_constant_is_optional_and_forwarded_only_when_set(self):
        source = MANUAL_MAKEFILE.read_text(encoding="utf-8")
        self.assertIn("PROCESSING_CONSTANT ?=", source)
        self.assertNotIn("--processing-constant 4.1675", source)

        # Unset: the flag is omitted entirely (marker-only measurement).
        dry_run = self.make(
            "-n", "concurrency", "HANDOFF=/protected/controller.env",
            check=True,
        )
        self.assertNotIn("--processing-constant", dry_run.stdout)

        # Set: the calibrated value is forwarded verbatim.
        dry_run = self.make(
            "-n", "concurrency", "HANDOFF=/protected/controller.env",
            "PROCESSING_CONSTANT=2.55",
            check=True,
        )
        self.assertIn("--processing-constant 2.55", dry_run.stdout)

        # A malformed value fails fast with a legible message.
        result = self.make(
            "concurrency", "HANDOFF=/protected/controller.env",
            "PROCESSING_CONSTANT=fast",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PROCESSING_CONSTANT", result.stderr)


if __name__ == "__main__":
    unittest.main()
