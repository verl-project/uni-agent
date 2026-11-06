"""Ownership and missing-data tests for the Linux experiment sampler."""

import errno
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SAMPLER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agentenv_offload" / "host_sampler.py"
spec = importlib.util.spec_from_file_location("host_sampler", SAMPLER_PATH)
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


def stat_line(pid, parent=1, start=100, user=12, system=3, virtual=65536, rss=4):
    fields = ["S"] + ["0"] * 21
    fields[1], fields[11], fields[12], fields[17] = str(parent), str(user), str(system), "1"
    fields[19], fields[20], fields[21] = str(start), str(virtual), str(rss)
    return f"{pid} (name with ) parentheses) " + " ".join(fields) + "\n"


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="offload-sampler-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proc, self.sys = self.root / "proc", self.root / "sys"
        self.proc.mkdir()
        self.sys.mkdir()
        (self.proc / "meminfo").write_text(
            "MemTotal: 1000 kB\nMemAvailable: 700 kB\nCached: 200 kB\nAnonPages: 100 kB\n"
        )
        (self.proc / "stat").write_text("cpu 10 0 5 80 3 1 1 0 0 0\n")
        (self.proc / "diskstats").write_text(
            "8 0 sda 1 0 8 2 3 0 16 4 0 6 7\n8 1 sda1 1 0 8 2 3 0 16 4 0 6 7\n253 0 dm-0 1 0 8 2 3 0 16 4 0 6 7\n"
        )
        for name in ("sda", "sda1", "dm-0"):
            (self.sys / "class" / "block" / name / "slaves").mkdir(parents=True)
        (self.sys / "class" / "block" / "sda1" / "partition").write_text("1")
        (self.sys / "class" / "block" / "dm-0" / "slaves" / "sda").touch()
        self.pid_file = self.root / "pids.json"
        self.write_registry([])

    def write_registry(self, entries):
        self.pid_file.write_text(json.dumps({"run_id": "test-run", "processes": entries}))

    def process(self, pid, parent=1, start=100):
        directory = self.proc / str(pid)
        directory.mkdir(exist_ok=True)
        (directory / "stat").write_text(stat_line(pid, parent, start))
        (directory / "smaps_rollup").write_text(
            "Rss: 16 kB\nPss: 12 kB\nAnonymous: 8 kB\nPss_Anon: 8 kB\nPss_File: 3 kB\nPss_Shmem: 1 kB\nSwap: 0 kB\n"
        )
        (directory / "io").write_text("rchar: 100\nwchar: 200\nread_bytes: 4096\nwrite_bytes: 8192\n")

    def make_sampler(self):
        return sampler.Sampler(self.pid_file, proc=self.proc, sys_root=self.sys, clock_ticks=100, page_bytes=4096)

    def test_ownership_survives_reparenting_but_not_pid_reuse(self):
        self.process(101, start=100)
        self.process(102, parent=101, start=101)
        self.process(999, start=90)  # An unrelated workload must stay excluded.
        self.write_registry([{"pid": 101, "starttime_ticks": 100, "label": "provider"}])
        monitor = self.make_sampler()
        self.assertEqual([x["pid"] for x in monitor.sample()["processes"]], [101, 102])
        (self.proc / "101" / "stat").unlink()
        self.process(102, parent=1, start=101)
        self.assertEqual([x["pid"] for x in monitor.sample()["processes"]], [102])
        self.process(102, parent=1, start=900)  # Reused PID is a new, unowned process.
        self.assertEqual(monitor.sample()["processes"], [])
        self.process(201, start=1000)
        self.write_registry([{"pid": 201, "starttime_ticks": 1000, "label": "restored"}])
        row = monitor.sample()
        self.assertEqual([x["pid"] for x in row["processes"]], [201])
        self.assertEqual(row["processes"][0]["label"], "restored")
        self.write_registry([])
        self.assertEqual(monitor.sample()["processes"], [])

    def test_missing_metrics_are_null_and_node_disks_are_not_double_counted(self):
        self.process(101)
        self.write_registry([{"pid": 101, "starttime_ticks": 100}])
        (self.proc / "101" / "smaps_rollup").unlink()
        (self.proc / "101" / "io").unlink()
        row = self.make_sampler().sample()
        process = row["processes"][0]
        self.assertIsNone(process["memory"]["pss_bytes"])
        self.assertIsNone(process["io"]["read_bytes"])
        self.assertEqual(process["errors"]["smaps_rollup"], "not_found")
        self.assertEqual(row["node"]["memory_bytes"]["Cached"], 200 * 1024)
        self.assertIsNone(row["node"]["memory_bytes"]["SwapFree"])
        devices = row["node"]["disk_devices"]
        self.assertEqual([x["device"] for x in devices], ["sda", "dm-0"])
        self.assertEqual(devices[1]["slaves"], ["sda"])
        self.assertNotIn("disk_total", row["node"])
        self.assertIsNone(row["memory_by_label"]["experiment"]["memory"]["pss_bytes"])
        self.assertFalse(row["memory_by_label"]["experiment"]["complete"])

    def test_registry_error_preserves_identity_but_marks_stale_coverage(self):
        self.process(101)
        self.write_registry([{"pid": 101, "starttime_ticks": 100}])
        monitor = self.make_sampler()
        monitor.sample()
        self.pid_file.write_text("{")
        row = monitor.sample()
        self.assertTrue(row["registry_stale"])
        self.assertEqual(row["registry_error"], "invalid_registry")
        self.assertEqual(row["processes"][0]["pid"], 101)
        self.assertIsNone(row["memory_by_label"]["experiment"]["memory"]["pss_bytes"])

    def test_permission_denial_and_identity_race_do_not_leak_bad_metrics(self):
        self.process(101)
        read_original = sampler.read_text_details

        def denied(path):
            return (None, "permission_denied", errno.EACCES) if path.name == "smaps_rollup" else read_original(path)

        with mock.patch.object(sampler, "read_text_details", side_effect=denied):
            result = sampler.sample_process(self.proc, (101, 100), {"label": "provider"}, 100, 4096)
        self.assertIsNone(result["memory"]["rss_bytes"])
        self.assertEqual(result["errors"]["smaps_rollup"], "permission_denied")
        old = sampler.parse_proc_stat(stat_line(101, start=100))
        replacement = sampler.parse_proc_stat(stat_line(101, start=200))
        with mock.patch.object(
            sampler, "read_proc_stat_details", side_effect=[(old, None, None), (replacement, None, None)]
        ):
            result = sampler.sample_process(self.proc, (101, 100), {"label": "provider"}, 100, 4096)
        self.assertEqual(result["status"], "identity_changed_during_sample")
        self.assertNotIn("memory", result)

    def test_address_space_release_requires_independent_maps_and_stat_evidence(self):
        self.process(101)
        read_original = sampler.read_text_details

        def exit_read(path):
            return (None, "read_error", errno.ESRCH) if path.name == "smaps_rollup" else read_original(path)

        maps_path = self.proc / "101" / "maps"
        maps_path.write_text("")
        with mock.patch.object(sampler, "read_text_details", side_effect=exit_read):
            nonzero_stat = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
            self.assertIsNone(nonzero_stat["memory"]["pss_bytes"])
            self.assertFalse(nonzero_stat["address_space_released"])
            (self.proc / "101" / "stat").write_text(stat_line(101, virtual=0, rss=1))
            nonzero_rss = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
            self.assertFalse(nonzero_rss["address_space_released"])
            (self.proc / "101" / "stat").write_text(stat_line(101, virtual=0, rss=0))
            maps_path.unlink()
            no_maps = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
            self.assertIsNone(no_maps["memory"]["pss_bytes"])
            self.assertFalse(no_maps["release_evidence"]["maps_read_succeeded"])
            maps_path.write_text("")
            released = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
            self.assertEqual(released["status"], "ok")
            self.assertTrue(released["address_space_released"])
            self.assertEqual(released["memory_observation"], "address_space_released")
            self.assertTrue(all(released["memory"][key] == 0 for key in sampler.SMAPS_FIELDS))
            self.assertEqual(released["errors"]["smaps_rollup"], "read_error")
            self.assertEqual(released["read_errno"]["smaps_rollup"], errno.ESRCH)
            self.assertTrue(released["release_evidence"]["final_identity_matched"])
            self.assertEqual(released["release_evidence"]["maps_characters"], 0)
            self.assertEqual(released["release_evidence"]["confirmation_stat"]["virtual_bytes"], 0)
            original = sampler.parse_proc_stat(stat_line(101, start=100, virtual=0, rss=0))
            reused = sampler.parse_proc_stat(stat_line(101, start=999, virtual=0, rss=0))
            with mock.patch.object(
                sampler, "read_proc_stat_details", side_effect=[(original, None, None), (reused, None, None)]
            ):
                replaced = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
            self.assertFalse(replaced["address_space_released"])
            self.assertEqual(replaced["status"], "identity_changed_during_sample")
            self.assertNotIn("memory", replaced)

    def test_permission_denial_cannot_be_converted_into_release(self):
        self.process(101)
        (self.proc / "101" / "stat").write_text(stat_line(101, virtual=0, rss=0))
        (self.proc / "101" / "maps").write_text("")
        read_original = sampler.read_text_details

        for denied_file in ("smaps_rollup", "maps"):
            with self.subTest(denied_file=denied_file):

                def denied(path, denied_file=denied_file):
                    if path.name == denied_file:
                        return None, "permission_denied", errno.EACCES
                    if path.name == "smaps_rollup":
                        return None, "read_error", errno.ESRCH
                    return read_original(path)

                with mock.patch.object(sampler, "read_text_details", side_effect=denied):
                    result = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096)
                self.assertIsNone(result["memory"]["pss_bytes"])
                self.assertFalse(result["address_space_released"])

    def remove_fixture_process(self, pid):
        directory = self.proc / str(pid)
        for name in ("stat", "smaps_rollup", "io", "maps"):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()

    def test_inventory_exit_moves_to_evidence_without_synthetic_counters(self):
        self.process(101, start=100)
        self.process(201, start=200)
        self.write_registry(
            [
                {"pid": 101, "starttime_ticks": 100, "label": "sandbox"},
                {"pid": 201, "starttime_ticks": 200, "label": "provider"},
            ]
        )
        original_scan = sampler.scan_processes

        def exit_after_inventory(proc):
            inventory = original_scan(proc)
            self.remove_fixture_process(101)
            return inventory

        with mock.patch.object(sampler, "scan_processes", side_effect=exit_after_inventory):
            row = self.make_sampler().sample()
        self.assertTrue(row["selection_coverage_known"])
        self.assertEqual([item["pid"] for item in row["processes"]], [201])
        self.assertEqual(row["memory_by_label"]["provider"]["memory"]["pss_bytes"], 12 * 1024)
        self.assertEqual(len(row["exited_processes"]), 1)
        exited = row["exited_processes"][0]
        self.assertNotIn("memory", exited)
        self.assertNotIn("io", exited)
        self.assertEqual(exited["read_errno"]["stat"], errno.ENOENT)
        evidence = exited["exit_evidence"]
        self.assertTrue(evidence["confirmed"])
        self.assertEqual(len(evidence["checks"]), 2)
        for check in evidence["checks"]:
            self.assertEqual(check["stat_errno"], errno.ENOENT)
            self.assertEqual(check["pid_directory"]["errno"], errno.ENOENT)
            self.assertFalse(check["pid_directory"]["present"])
        self.assertGreaterEqual(evidence["confirmed_monotonic_seconds"], row["monotonic_seconds"])
        self.assertLessEqual(row["exit_confirmation_delay_seconds"], row["sample_duration_seconds"])

    def test_missing_stat_with_present_directory_is_still_unknown(self):
        self.process(101)
        seen = sampler.read_proc_stat(self.proc, 101)
        (self.proc / "101" / "stat").unlink()
        result = sampler.sample_process(self.proc, (101, 100), {"label": "sandbox"}, 100, 4096, inventory_stat=seen)
        self.assertEqual(result["status"], "identity_unavailable")
        self.assertFalse(result["exit_evidence"]["confirmed"])
        self.assertTrue(result["exit_evidence"]["checks"][0]["pid_directory"]["present"])
        self.assertNotIn("memory", result)

    def test_absence_rejects_permission_io_root_errors_and_pid_reappearance(self):
        self.process(101)
        seen = sampler.read_proc_stat(self.proc, 101)
        root_identity = sampler.directory_identity(self.proc)
        self.remove_fixture_process(101)
        missing = (None, "not_found", errno.ENOENT)
        for category, number in (("permission_denied", errno.EACCES), ("read_error", errno.EIO), ("malformed", None)):
            with (
                self.subTest(stat_error=category),
                mock.patch.object(sampler, "read_proc_stat_details", return_value=(None, category, number)),
            ):
                result = sampler.sample_process(
                    self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
                )
                self.assertEqual(result["status"], "identity_unavailable")
                self.assertFalse(result["exit_evidence"]["confirmed"])
                self.assertNotIn("memory", result)
        for root_error, number in (("not_found", errno.ENOENT), ("permission_denied", errno.EACCES)):
            with (
                self.subTest(root_error=root_error),
                mock.patch.object(
                    sampler,
                    "directory_identity",
                    return_value={"device": None, "inode": None, "error": root_error, "errno": number},
                ),
            ):
                result = sampler.sample_process(
                    self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
                )
                self.assertEqual(result["exit_evidence"]["reason"], "proc_root_changed_or_unavailable")
                self.assertFalse(result["exit_evidence"]["confirmed"])
        with mock.patch.object(
            sampler,
            "directory_presence",
            return_value={"present": None, "error": "permission_denied", "errno": errno.EACCES},
        ):
            result = sampler.sample_process(
                self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
            )
            self.assertFalse(result["exit_evidence"]["confirmed"])
        reused = sampler.parse_proc_stat(stat_line(101, start=999))
        with mock.patch.object(sampler, "read_proc_stat_details", side_effect=[missing, missing, (reused, None, None)]):
            result = sampler.sample_process(
                self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
            )
        self.assertFalse(result["exit_evidence"]["confirmed"])
        self.assertEqual(result["exit_evidence"]["reason"], "identity_reappeared_or_reused")
        self.assertNotIn("memory", result)

    def test_exit_at_final_stat_is_confirmed_but_reuse_remains_unknown(self):
        self.process(101)
        seen = sampler.read_proc_stat(self.proc, 101)
        root_identity = sampler.directory_identity(self.proc)
        original_reader = sampler.read_proc_stat_details
        calls = 0

        def exit_before_final(proc, pid):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.remove_fixture_process(pid)
            return original_reader(proc, pid)

        with mock.patch.object(sampler, "read_proc_stat_details", side_effect=exit_before_final):
            result = sampler.sample_process(
                self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
            )
        self.assertEqual(result["status"], "confirmed_exited")
        self.assertNotIn("memory", result)
        self.assertNotIn("io", result)
        self.process(101, start=999)
        reused = sampler.sample_process(
            self.proc, (101, 100), {}, 100, 4096, inventory_stat=seen, proc_identity=root_identity
        )
        self.assertEqual(reused["status"], "identity_unavailable")
        self.assertFalse(reused["exit_evidence"]["confirmed"])

    def test_parse_stat_and_registration_validation(self):
        result = sampler.parse_proc_stat(stat_line(123, parent=99, start=765, user=20, system=4))
        self.assertEqual((result["ppid"], result["starttime_ticks"], result["user_ticks"]), (99, 765, 20))
        with self.assertRaises(ValueError):
            sampler.validate_registry({"processes": [{"pid": 123}]})
        self.assertFalse(sampler.path_is_within("/home/zxh/run-other/file", "/home/zxh/run"))
        self.assertFalse(sampler.path_is_within("/home/zxh/run/../other/file", "/home/zxh/run"))
        self.assertTrue(sampler.path_is_within("/home/zxh/run/socket", "/home/zxh/run"))

    def test_discovery_requires_exact_allowlist_and_owned_path(self):
        self.process(101)
        self.process(102)
        (self.proc / "101" / "cmdline").write_text("firecracker\0--api-sock=/home/zxh/run/vm.sock\0")
        (self.proc / "102" / "cmdline").write_text("firecracker\0--api-sock=/home/zxh/other/vm.sock\0")

        def link(path):
            return "/opt/firecracker" if path.name == "exe" else "/"

        inventory = sampler.scan_processes(self.proc)
        with mock.patch.object(sampler.os, "readlink", side_effect=link):
            found = sampler.discover_processes(self.proc, inventory, "/home/zxh/run", {"firecracker"})
            excluded = sampler.discover_processes(self.proc, inventory, "/home/zxh/run", {"agentenv"})
        self.assertEqual([x["pid"] for x in found], [101])
        self.assertEqual(excluded, [])


@unittest.skipUnless(sys.platform.startswith("linux"), "real proc accounting requires Linux")
class RealProcessTests(unittest.TestCase):
    def test_controlled_child_is_measured_without_selecting_parent(self):
        with tempfile.TemporaryDirectory(prefix="offload-live-sampler-") as directory:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; b=bytearray(8*1024*1024); print('ready',flush=True); time.sleep(30)",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                stat = sampler.read_proc_stat(Path("/proc"), process.pid)
                registry = Path(directory) / "owned.json"
                registry.write_text(
                    json.dumps({"processes": [{"pid": process.pid, "starttime_ticks": stat["starttime_ticks"]}]})
                )
                row = sampler.Sampler(registry).sample()
                self.assertEqual([x["pid"] for x in row["processes"]], [process.pid])
                self.assertNotIn(os.getpid(), [x["pid"] for x in row["processes"]])
                self.assertGreater(row["processes"][0]["memory"]["pss_bytes"], 8 * 1024 * 1024)
                self.assertGreater(row["node"]["memory_bytes"]["MemTotal"], 0)
                self.assertNotIn("cmdline", json.dumps(row))
                proc_identity = sampler.directory_identity(Path("/proc"))
                process.terminate()
                process.wait(timeout=5)
                exited = sampler.sample_process(
                    Path("/proc"),
                    (process.pid, stat["starttime_ticks"]),
                    {},
                    os.sysconf("SC_CLK_TCK"),
                    os.sysconf("SC_PAGE_SIZE"),
                    inventory_stat=stat,
                    proc_identity=proc_identity,
                )
                self.assertEqual(exited["status"], "confirmed_exited")
                self.assertNotIn("memory", exited)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
