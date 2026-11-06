"""Local workflow tests; these do not substitute for AgentEnv VM validation."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("offload_demo", ROOT / "scripts/agentenv_offload/demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class LocalNode:
    """Real guest probe process, with explicit simulated lifecycle ownership."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.running = True
        self.generation = 1
        self.process = None
        self.url = None

    def prepare(self, trial):
        return {"folder": str(self.folder), "sampler": {}, "clock_calibration": None}

    def discover(self, sid):
        return (
            [{"pid": 1000 + self.generation, "starttime_ticks": self.generation, "label": "sandbox"}]
            if self.running
            else []
        )

    def discover_and_register(self, context, trial, sid):
        processes = self.discover(sid)
        self.register(context, trial, processes)
        return processes

    def alive(self, processes):
        return [item for item in processes if item in self.discover("local")]

    def register(self, context, trial, processes):
        pass

    def audit(self, sid=None):
        return {"ublk_devices": [], "netns": [], "runtime_entries": [], "owned_artifacts": [], "errors": []}

    def launch_probe(self, sid):
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/agentenv_offload/guest_probe.py"),
                "--memory-mib",
                "1",
                "--port",
                "0",
                "--host",
                "127.0.0.1",
                "--workdir",
                str(self.folder / "guest"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        output = queue.Queue()
        threading.Thread(target=lambda: output.put(self.process.stdout.readline()), daemon=True).start()
        ready = json.loads(output.get(timeout=10))
        self.url = f"http://127.0.0.1:{ready['port']}"
        return 0.01

    def finish(self, context, output):
        pass

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.communicate(timeout=5)


class LocalAPI:
    def __init__(self, node: LocalNode, *, fail_tool=False, fail_delete=False):
        self.node = node
        self.secrets = ["test-platform-key", "test-private-key"]
        self.fail_tool, self.fail_delete = fail_tool, fail_delete
        self.tool_calls = 0
        self.deleted = False
        self.created_body = None

    def request(self, method, path, body=None, *, headers=None, expected=(200,)):
        if path == "/sandboxes-cold":
            self.created_body = body
            return 201, {"sandboxID": "local", "trafficAccessToken": "test-private-key"}
        if path == "/nodes":
            return 200, []
        if path.endswith("/refreshes"):
            return 204, {}
        if path.endswith("/pause"):
            self.node.running = False
            return 204, {}
        if path.endswith("/connect"):
            self.node.running = True
            self.node.generation += 1
            return 201, {"sandboxID": "local", "trafficAccessToken": "test-private-key"}
        if method == "DELETE":
            if self.fail_delete:
                raise demo.ExperimentError("delete unavailable")
            self.deleted = True
            self.node.running = False
            return 204, {}
        if path == "/sandboxes/local":
            return (404, {}) if self.deleted else (200, {"state": "running"})
        if path.startswith("/proxy"):
            if not self.node.running:
                raise AssertionError("orchestrator requested a paused guest")
            if path == "/proxy/tool":
                self.tool_calls += 1
                if self.fail_tool:
                    raise demo.ExperimentError("tool response lost")
            assert headers["e2b-traffic-access-token"] == "test-private-key"
            request = Request(
                self.node.url + path.removeprefix("/proxy"),
                method=method,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        raise AssertionError((method, path))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offload-demo-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            output=self.root,
            operation_timeout=0.1,
            cycles=2,
            wait_seconds=0.01,
            guest_image="python@sha256:" + "1" * 64,
            cpus=1,
            sandbox_memory_mib=512,
            io_tail_seconds=0,
            observation_seconds=0,
            sample_interval=0.001,
            pause_before_cleanup=False,
        )
        self.node = LocalNode(self.root)
        self.addCleanup(self.node.close)

    def test_real_probe_oracle_survives_workflow_and_secrets_are_filtered(self):
        api = LocalAPI(self.node)
        result = demo.Trial(self.args, api, self.node, "offload", "pair-000-offload-test").run()
        self.assertTrue(result["success"], result)
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertEqual(len(result["cycles"]), 2)
        self.assertEqual(api.tool_calls, 2)
        self.assertEqual(api.created_body["autoResume"], {"enabled": False})
        self.assertFalse(api.created_body["network"]["allowPublicTraffic"])
        events = (self.root / result["trial_id"] / "events.jsonl").read_text()
        self.assertNotIn("test-private-key", events)
        self.assertIn("[redacted]", events)
        names = [json.loads(line)["event"] for line in events.splitlines()]
        self.assertLess(names.index("resources_released"), names.index("resume_requested"))
        self.assertLess(names.index("resume_api_done"), names.index("tool_dispatched"))

    def test_lost_command_response_is_not_retried_and_cleanup_still_runs(self):
        api = LocalAPI(self.node, fail_tool=True)
        result = demo.Trial(self.args, api, self.node, "resident", "pair-000-resident-test").run()
        self.assertFalse(result["success"])
        self.assertTrue(result["unknown_command"])
        self.assertEqual(api.tool_calls, 1)
        self.assertTrue(api.deleted)
        report = demo.summary([result])
        self.assertEqual(report["started_trials"], 1)
        self.assertEqual(report["by_mode"]["resident"]["success_rate"], 0)

    def test_resume_200_is_an_unexpected_wakeup_and_still_cleans_up(self):
        class AlreadyRunningAPI(LocalAPI):
            def request(self, method, path, *args, **kwargs):
                status, body = super().request(method, path, *args, **kwargs)
                return (200, body) if path.endswith("/connect") else (status, body)

        api = AlreadyRunningAPI(self.node)
        result = demo.Trial(self.args, api, self.node, "offload", "pair-000-offload-test").run()
        self.assertFalse(result["success"])
        self.assertIn("unexpected wakeup", result["error"])
        self.assertTrue(api.deleted)
        self.assertTrue(result["cleanup"]["confirmed"])
        self.assertEqual(api.tool_calls, 0)

    def test_cleanup_failure_is_retained_even_after_correct_task(self):
        api = LocalAPI(self.node, fail_delete=True)
        result = demo.Trial(self.args, api, self.node, "resident", "pair-000-resident-test").run()
        self.assertTrue(result["success"])
        self.assertFalse(result["cleanup"]["confirmed"])
        report = demo.summary([result])["by_mode"]["resident"]
        self.assertEqual(report["started"], 1)
        self.assertEqual(report["cleanup_confirmed"], 0)

    def test_pause_refuses_busy_or_unknown_command_and_paused_guest_reads(self):
        trial = demo.Trial(self.args, LocalAPI(self.node), self.node, "offload", "pair-000-offload-test")
        trial.lifecycle = "running"
        trial.current_processes = self.node.discover("local")
        for field in ("busy", "unknown_command"):
            setattr(trial, field, True)
            with self.assertRaises(demo.ExperimentError):
                trial.pause()
            setattr(trial, field, False)
        trial.lifecycle = "paused"
        with self.assertRaises(demo.ExperimentError):
            trial.proxy("GET", "/state")

    def test_pause_api_success_does_not_override_live_firecracker_evidence(self):
        class StillRunningAPI(LocalAPI):
            def request(self, method, path, *args, **kwargs):
                if path.endswith("/pause"):
                    return 204, {}
                return super().request(method, path, *args, **kwargs)

        trial = demo.Trial(self.args, StillRunningAPI(self.node), self.node, "offload", "pair-000-offload-test")
        trial.sid = "local"
        trial.lifecycle = "running"
        trial.current_processes = self.node.discover("local")
        with self.assertRaisesRegex(demo.ExperimentError, "still alive"):
            trial.pause()
        self.assertEqual(trial.lifecycle, "pausing")
        events = (trial.output / "events.jsonl").read_text()
        self.assertNotIn('"event": "resources_released"', events)

    def test_api_and_pid_cleanup_are_insufficient_with_owned_disk_residue(self):
        baseline_audit = self.node.audit

        def residue(sid=None):
            result = baseline_audit(sid)
            if sid:
                result["owned_artifacts"] = [{"path": "owned-artifact", "exists": True, "logical_bytes": 1}]
            return result

        self.node.audit = residue
        result = demo.Trial(self.args, LocalAPI(self.node), self.node, "resident", "pair-000-resident-test").run()
        self.assertTrue(result["cleanup"]["api_pid_confirmed"])
        self.assertFalse(result["cleanup"]["confirmed"])
        self.assertEqual(result["cleanup"]["resource_audit"]["status"], "pending")

    def test_private_namespace_is_observed_and_blocks_cleanup(self):
        state, runtime = self.root / "state", self.root / "runtime"
        state.mkdir()
        runtime.mkdir()

        def read_audit(sid=None):
            response = subprocess.run(
                [sys.executable, "-c", demo.NODE_HELPER],
                input=json.dumps(
                    {"action": "audit", "state_dir": str(state), "runtime_dir": str(runtime), "sandbox_id": sid}
                ),
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(response.returncode, 0, response.stderr)
            return json.loads(response.stdout)

        self.node.audit = read_audit
        self.node.running = False
        trial = demo.Trial(self.args, LocalAPI(self.node), self.node, "offload", "pair-000-offload-test")
        trial.sid, trial.lifecycle = "local", "paused"
        trial.baseline_audit = read_audit()
        private_directory = runtime / "netns"
        private_directory.mkdir()
        namespace = private_directory / "owned-private-namespace"
        namespace.touch()
        observed = read_audit("local")
        self.assertIn(str(namespace), observed["netns"])
        self.assertEqual(observed["netns_by_directory"][str(private_directory)], [str(namespace)])
        trial.delete()
        self.assertTrue(trial.cleanup["api_pid_confirmed"])
        self.assertFalse(trial.cleanup["confirmed"])
        self.assertEqual(trial.cleanup["resource_audit"]["status"], "pending")
        self.assertIn(str(namespace), trial.cleanup["resource_audit"]["new_netns"])

    def test_validation_deletes_paused_state_and_requires_its_artifacts_to_be_removed(self):
        self.args.pause_before_cleanup = True
        self.args.cycles = 1
        for index, retain_artifact in enumerate((False, True)):
            with self.subTest(retain_artifact=retain_artifact):
                node = LocalNode(self.root / f"node-{index}")
                self.addCleanup(node.close)
                artifact = self.root / f"paused-artifact-{index}" / "memory.state"

                class ArtifactAPI(LocalAPI):
                    deleted_from = None

                    def request(self, method, path, *args, **kwargs):
                        if path.endswith("/pause"):
                            self.artifact.parent.mkdir(exist_ok=True)
                            self.artifact.write_bytes(b"test runtime artifact")
                        if method == "DELETE":
                            self.deleted_from = "running" if self.node.running else "paused"
                            if not self.retain_artifact and self.artifact.exists():
                                self.artifact.unlink()
                                self.artifact.parent.rmdir()
                        return super().request(method, path, *args, **kwargs)

                original_audit = node.audit

                def artifact_audit(sid=None, original_audit=original_audit, artifact=artifact):
                    audit = original_audit(sid)
                    if sid:
                        audit["owned_artifacts"] = [{"path": str(artifact.parent), "exists": artifact.parent.exists()}]
                    return audit

                node.audit = artifact_audit
                api = ArtifactAPI(node)
                api.artifact = artifact
                api.retain_artifact = retain_artifact
                result = demo.Trial(self.args, api, node, "offload", f"pair-{index:03d}-offload-test").run()
                self.assertTrue(result["success"], result)
                self.assertTrue(result["validation_only"])
                self.assertEqual(result["final_cleanup_state"], "paused")
                self.assertEqual(result["cleanup"]["final_cleanup_state"], "paused")
                self.assertEqual(api.deleted_from, "paused")
                self.assertEqual(result["cleanup"]["confirmed"], not retain_artifact)
                events = [
                    json.loads(line)
                    for line in (self.root / result["trial_id"] / "events.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    next(row for row in events if row["event"] == "delete_requested")["from_state"], "paused"
                )

    def test_oracle_checks_ram_and_fd_not_only_files(self):
        before = dict.fromkeys(demo.ORACLE_FIELDS, "same")
        for field in ("nonce", "memory_sha256", "fd_offset", "counter"):
            with self.assertRaises(demo.ExperimentError):
                demo.assert_continuity(before, {**before, field: "changed"})

    def test_main_stops_after_first_failure_and_keeps_started_denominator(self):
        self.args.output = self.root / "run"
        self.args.credentials_file = self.root / "private.toml"
        self.args.api_url = "http://127.0.0.1:8000"
        self.args.mode, self.args.pairs, self.args.seed = "both", 3, 1

        def failed_trial(args, api, node, mode, trial_id):
            return SimpleNamespace(
                run=lambda: {
                    "trial_id": trial_id,
                    "mode": mode,
                    "success": False,
                    "cleanup": {"confirmed": True},
                    "error": "ordinary setup failure",
                }
            )

        with (
            patch.object(demo, "parse_args", return_value=self.args),
            patch.object(demo, "read_credentials", return_value={"api_key": "test-secret"}),
            patch.object(demo, "Node", return_value=self.node),
            patch.object(demo, "Trial", side_effect=failed_trial) as trials,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(demo.main([]), 1)
        self.assertEqual(trials.call_count, 1)
        results = (self.args.output / "results.jsonl").read_text().splitlines()
        self.assertEqual(len(results), 1)
        report = json.loads((self.args.output / "run_summary.json").read_text())
        self.assertEqual(report["started_trials"], 1)
        self.assertEqual(next(iter(report["by_mode"].values()))["success_rate"], 0)


class ForecastTimingTests(unittest.TestCase):
    """Deterministic clock checks of scheduling, without a simulated speedup."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="offload-forecast-timing-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trial_count = 0

    def run_timed_trial(
        self,
        mode="offload",
        lead=None,
        *,
        pause_seconds=0.5,
        resume_seconds=1,
        oracle_seconds=0.5,
        tool_seconds=0.2,
        fail_resume=False,
    ):
        class Clock:
            value = 100.0

            def monotonic(self):
                return self.value

            def time_ns(self):
                return int(self.value * 1e9)

            def sleep(self, seconds):
                if seconds < 0:
                    raise AssertionError("negative sleep")
                self.value += seconds

        clock = Clock()
        node = LocalNode(self.root)
        node.launch_probe = lambda _sid: 0

        class TimedAPI(LocalAPI):
            def __init__(self):
                super().__init__(node)
                self.state = {
                    **dict.fromkeys(demo.ORACLE_FIELDS, "preserved"),
                    "counter": 0,
                    "fd_offset": 0,
                    "fd_size": 4096,
                }
                self.state_calls, self.dispatched, self.connect_calls = [], [], []

            def request(self, method, path, *args, **kwargs):
                if path.endswith("/pause"):
                    clock.sleep(pause_seconds)
                elif path.endswith("/connect"):
                    self.connect_calls.append(clock.monotonic())
                    clock.sleep(resume_seconds)
                    if fail_resume:
                        raise demo.ExperimentError("restore response unavailable")
                elif path == "/proxy/health":
                    return 200, {"ok": True}
                elif path == "/proxy/state":
                    self.state_calls.append(clock.monotonic())
                    clock.sleep(oracle_seconds)
                    return 200, dict(self.state)
                elif path == "/proxy/tool":
                    self.dispatched.append(clock.monotonic())
                    clock.sleep(tool_seconds)
                    self.state["counter"] += 1
                    self.state["fd_offset"] += 17
                    return 200, {"ok": True, "duplicate": False, "state": dict(self.state)}
                return super().request(method, path, *args, **kwargs)

        self.trial_count += 1
        args = SimpleNamespace(
            output=self.root,
            operation_timeout=10,
            cycles=1,
            wait_seconds=10,
            guest_image="python@sha256:" + "1" * 64,
            cpus=1,
            sandbox_memory_mib=512,
            io_tail_seconds=0,
            observation_seconds=0,
            sample_interval=0.01,
            pause_before_cleanup=False,
        )
        if lead is not None:
            args.resume_lead_seconds = lead
        api = TimedAPI()
        trial = demo.Trial(args, api, node, mode, f"pair-{self.trial_count:03d}-{mode}-clock")
        with patch.object(demo, "time", clock):
            result = trial.run()
        events = [json.loads(line) for line in (trial.output / "events.jsonl").read_text().splitlines()]
        deadline = next(event["ready_monotonic_seconds"] for event in events if event["event"] == "tool_deadline_known")
        return result, api, events, deadline

    def test_forecast_restores_and_validates_early_but_defers_tool_effects_to_deadline(self):
        result, api, events, deadline = self.run_timed_trial(lead=2)
        self.assertTrue(result["success"], result)
        cycle = result["cycles"][0]
        self.assertAlmostEqual(api.connect_calls[0], deadline - 2)
        self.assertLess(api.state_calls[-1], deadline)
        self.assertAlmostEqual(api.dispatched[0], deadline)
        self.assertAlmostEqual(cycle["first_tool_seconds"], 0.2)
        self.assertAlmostEqual(cycle["recovery_to_oracle_seconds"], 1.5)
        self.assertAlmostEqual(cycle["oracle_ready_before_tool_seconds"], 0.5)
        self.assertAlmostEqual(cycle["restore_oracle_seconds"], 0)
        self.assertAlmostEqual(cycle["idle_actual_seconds"], 10)
        measurement_start = next(
            event["monotonic_seconds"] for event in events if event["event"] == "measurement_started"
        )
        measurement_end = next(
            event["monotonic_seconds"] for event in events if event["event"] == "measurement_finished"
        )
        self.assertLess(measurement_start, cycle["recovery_started_monotonic_seconds"])
        self.assertGreater(measurement_end, api.dispatched[0])
        self.assertAlmostEqual(result["task_seconds"], 10.7)

    def test_default_is_on_demand_and_resident_control_ignores_the_forecast(self):
        on_demand, offload_api, _, deadline = self.run_timed_trial()
        self.assertAlmostEqual(offload_api.connect_calls[0], deadline)
        self.assertAlmostEqual(on_demand["cycles"][0]["first_tool_seconds"], 1.7)
        self.assertAlmostEqual(on_demand["cycles"][0]["recovery_to_oracle_seconds"], 1.5)
        self.assertAlmostEqual(on_demand["task_seconds"], 12.2)
        resident, resident_api, _, resident_deadline = self.run_timed_trial(mode="resident", lead=2)
        self.assertTrue(resident["success"], resident)
        self.assertEqual(resident_api.connect_calls, [])
        self.assertAlmostEqual(resident_api.state_calls[-1], resident_deadline)
        self.assertAlmostEqual(resident["cycles"][0]["resume_lead_seconds"], 0)
        self.assertAlmostEqual(resident["cycles"][0]["first_tool_seconds"], 0.7)
        self.assertAlmostEqual(resident["task_seconds"], 11.2)

    def test_late_recovery_and_pause_overrun_remain_in_first_tool_and_task_latency(self):
        for pause_seconds, resume_seconds, expected_extra in ((0.5, 5, 3.7), (12, 1, 3.7)):
            with self.subTest(pause_seconds=pause_seconds, resume_seconds=resume_seconds):
                result, api, _, deadline = self.run_timed_trial(
                    lead=2, pause_seconds=pause_seconds, resume_seconds=resume_seconds
                )
                self.assertTrue(result["success"], result)
                cycle = result["cycles"][0]
                self.assertAlmostEqual(cycle["first_tool_seconds"], expected_extra)
                self.assertAlmostEqual(cycle["restore_oracle_seconds"], expected_extra - 0.2)
                self.assertAlmostEqual(cycle["oracle_ready_before_tool_seconds"], 0)
                self.assertGreater(api.dispatched[0], deadline)
                self.assertAlmostEqual(result["task_seconds"], 10.5 + expected_extra)

    def test_failed_forecast_recovery_sends_no_tool_and_still_cleans_up(self):
        result, api, events, _ = self.run_timed_trial(lead=2, fail_resume=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["cycles"], [])
        self.assertEqual(api.dispatched, [])
        self.assertFalse(result["unknown_command"])
        self.assertTrue(api.deleted)
        self.assertTrue(result["cleanup"]["confirmed"])
        names = [event["event"] for event in events]
        self.assertIn("trial_failed", names)
        self.assertNotIn("tool_dispatched", names)
        self.assertGreater(result["task_seconds"], 0)


class HTTPTests(unittest.TestCase):
    def test_platform_credential_is_not_sent_to_application_and_redirect_is_rejected(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, dict(self.headers)))
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/credential-leak")
                    self.end_headers()
                else:
                    body = b'{"ok":true}'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Neither uppercase nor lowercase proxy settings may divert the
            # loopback tunnel, even when the environment disables no-proxy.
            with patch.dict(
                os.environ,
                {
                    "HTTP_PROXY": "http://127.0.0.1:1",
                    "http_proxy": "http://127.0.0.1:1",
                    "HTTPS_PROXY": "http://127.0.0.1:1",
                    "https_proxy": "http://127.0.0.1:1",
                    "NO_PROXY": "",
                    "no_proxy": "",
                },
            ):
                api = demo.API(f"http://127.0.0.1:{server.server_port}", "platform-secret")
                api.request("GET", "/nodes")
                api.request("GET", "/proxy/health", headers={"e2b-traffic-access-token": "app-secret"})
                with self.assertRaises(demo.ExperimentError):
                    api.request("GET", "/redirect")
            self.assertEqual(requests[0][1]["X-Api-Key"], "platform-secret")
            self.assertNotIn("X-Api-Key", requests[1][1])
            self.assertNotIn("/credential-leak", [path for path, _ in requests])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class SourceArchiveTests(unittest.TestCase):
    def test_archive_pins_original_bytes_and_omits_adjacent_credentials(self):
        with tempfile.TemporaryDirectory(prefix="offload-source-archive-") as temporary:
            root = Path(temporary)
            source, output = root / "input", root / "output"
            source.mkdir()
            output.mkdir()
            originals = {name: ("# original " + name + "\n").encode() for name in demo.SOURCE_FILES}
            for name, content in originals.items():
                (source / name).write_bytes(content)
            (source / "credentials.toml").write_text('api_key = "must-not-be-copied"\n')
            (source / "unrelated.py").write_text("# unrelated\n")
            hashes = demo.archive_sources(output, source)
            (source / "guest_probe.py").write_text("# later revision\n")
            self.assertEqual({path.name for path in (output / "source").iterdir()}, set(demo.SOURCE_FILES))
            for name, original in originals.items():
                archived = (output / "source" / name).read_bytes()
                self.assertEqual(archived, original)
                self.assertEqual(hashes[name], hashlib.sha256(archived).hexdigest())
                self.assertNotIn(b"must-not-be-copied", archived)


class DiscoveryRegistrationTests(unittest.TestCase):
    def test_combined_node_protocol_uses_one_control_call(self):
        with tempfile.TemporaryDirectory(prefix="offload-discovery-protocol-") as temporary:
            args = SimpleNamespace(
                output=Path(temporary), ssh_config=None, node_ssh="unused-test-node", serial_dir="/owned/serial"
            )
            node = demo.Node(args)
            identity = {"pid": 1234, "starttime_ticks": 777, "label": "sandbox"}
            with patch.object(node, "call", return_value={"processes": [identity]}) as control:
                found = node.discover_and_register({"folder": "/owned/trial"}, "trial-1", "sandbox-1")
            self.assertEqual(found, [identity])
            control.assert_called_once_with(
                "discover_and_register",
                sandbox_id="sandbox-1",
                serial_dir="/owned/serial",
                folder="/owned/trial",
                trial="trial-1",
            )

    def test_helper_registers_only_unique_owned_firecracker_and_keeps_base_roots(self):
        with tempfile.TemporaryDirectory(prefix="offload-discovery-helper-") as temporary:
            root = Path(temporary)
            path_type = type(root)
            proc, folder, serial = root / "proc", root / "trial", root / "serial"
            proc.mkdir()
            folder.mkdir()
            base = {"pid": 9000, "starttime_ticks": 100, "label": "provider", "include_descendants": True}
            (folder / "base-pids.json").write_text(json.dumps({"processes": [base]}))
            links = {}

            def add_process(pid):
                process = proc / str(pid)
                process.mkdir()
                fields = ["R"] + ["0"] * 18 + ["777"] + ["0"] * 5
                (process / "stat").write_text(f"{pid} (firecracker) " + " ".join(fields))
                links[str(process / "exe")] = "/owned/bin/firecracker"
                links[str(process / "fd" / "1")] = str(serial / "sandbox-1" / "firecracker-stdout.log")

            def fixture_path(*parts):
                if parts and parts[0] == "/proc":
                    return path_type(proc, *parts[1:])
                return path_type(*parts)

            def run_helper():
                payload = {
                    "action": "discover_and_register",
                    "sandbox_id": "sandbox-1",
                    "serial_dir": str(serial),
                    "folder": str(folder),
                    "trial": "trial-1",
                }
                output = io.StringIO()
                with (
                    patch("pathlib.Path", side_effect=fixture_path),
                    patch("os.readlink", side_effect=lambda path: links[str(path)]),
                    patch("sys.stdin", io.StringIO(json.dumps(payload))),
                    redirect_stdout(output),
                ):
                    exec(demo.NODE_HELPER, {})
                return json.loads(output.getvalue())

            add_process(1234)
            response = run_helper()
            recorded = json.loads((folder / "pids.json").read_text())
            self.assertEqual(recorded["run_id"], "trial-1")
            self.assertEqual(recorded["processes"], [base, response["processes"][0]])
            self.assertEqual(response["processes"][0]["pid"], 1234)
            self.assertEqual(response["processes"][0]["starttime_ticks"], 777)
            before = (folder / "pids.json").read_bytes()
            add_process(2345)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                run_helper()
            self.assertEqual((folder / "pids.json").read_bytes(), before)


@unittest.skipUnless(sys.platform.startswith("linux"), "node helper uses Linux /proc")
class NodeHelperTests(unittest.TestCase):
    def test_prepare_tracks_a_new_provider_child_before_sandbox_registration(self):
        with tempfile.TemporaryDirectory(prefix="offload-node-helper-") as temporary:
            root = Path(temporary)
            provider = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys,time;print('ready',flush=True);sys.stdin.readline();"
                    "child=subprocess.Popen([sys.executable,'-c',"
                    "'import time;data=bytearray(1048576);time.sleep(60)']);"
                    "print(child.pid,flush=True);time.sleep(60)",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            context = None

            def helper(payload):
                response = subprocess.run(
                    [sys.executable, "-c", demo.NODE_HELPER],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=20,
                    env={**os.environ, "PATH": str(root / "no-python-on-path")},
                )
                self.assertEqual(response.returncode, 0, response.stderr)
                return json.loads(response.stdout)

            try:
                self.assertEqual(provider.stdout.readline().strip(), "ready")
                fields = Path(f"/proc/{provider.pid}/stat").read_text().rsplit(") ", 1)[1].split()
                registry = root / "provider.json"
                registry.write_text(
                    json.dumps(
                        {
                            "processes": [
                                {
                                    "pid": provider.pid,
                                    "starttime_ticks": int(fields[19]),
                                    "label": "provider",
                                    "include_descendants": False,
                                }
                            ]
                        }
                    )
                )
                context = helper(
                    {
                        "action": "prepare",
                        "root": str(root),
                        "trial": "owned-test",
                        "provider_pid_file": str(registry),
                        "provider_pids": [],
                        "interval": 0.03,
                        "sampler": base64.b64encode((demo.HERE / "host_sampler.py").read_bytes()).decode(),
                    }
                )
                self.assertTrue(context["provider_roots"][0]["include_descendants"])
                provider.stdin.write("spawn\n")
                provider.stdin.flush()
                child_pid = int(provider.stdout.readline().strip())
                metrics_path = Path(context["folder"]) / "metrics.jsonl"
                deadline = time.monotonic() + 5
                found = []
                while time.monotonic() < deadline:
                    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.endswith("}")]
                    found = [process for row in rows for process in row["processes"] if process["pid"] == child_pid]
                    if found:
                        break
                    time.sleep(0.05)
                self.assertTrue(found, "new provider child must be sampled without explicit sandbox registration")
                self.assertTrue(all(item["label"] == "provider" for item in found))
                self.assertTrue(any(item.get("memory", {}).get("pss_bytes", 0) > 0 for item in found))
            finally:
                if context is not None:
                    helper({"action": "finish", "folder": context["folder"], "sampler": context["sampler"]})
                if provider.poll() is None:
                    os.killpg(provider.pid, signal.SIGTERM)
                provider.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
