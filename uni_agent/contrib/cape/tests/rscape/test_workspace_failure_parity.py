# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
import urllib.error

from pycape.workspace import WorkspaceConfig
from pycape.workspace_fs import WorkspaceManager, WorkspaceError, mount_entries, new_workspace_id
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_LIVE_TEST") == "1", "requires actual disposable fuse2fs images")
class WorkspaceFailureParityTests(unittest.TestCase):
    def test_bounded_failures_are_safe_in_both_native_agents(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                self.exercise(binary)

    def exercise(self, binary):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-fault-"))
        controller = node = busy = foreign = None
        foreign_mount = None
        passed = False
        try:
            environment = dict(os.environ)
            for tool, variable, actual in (("mkfs", "CAPE_WORKSPACE_MKFS_BIN", "/usr/sbin/mkfs.ext4"),
                                           ("fuse", "CAPE_WORKSPACE_FUSE2FS_BIN", "/usr/bin/fuse2fs")):
                wrapper = root / (tool + "-wrapper")
                wrapper.write_text("#!/bin/sh\nif test -e " + shlex.quote(str(root / (tool + "-fail"))) +
                                   "; then echo 'injected filesystem tool failure: No space left on device' >&2; exit 28; fi\nexec " + actual + ' "$@"\n')
                wrapper.chmod(0o700)
                environment[variable] = str(wrapper)
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "300")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "fault", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
            with (root / "node.log").open("w") as stream:
                node = subprocess.Popen([binary, "node", "enroll", *helper.node_credentials(url, "fault-node", "fault"), "--controller-url", url, "--pool", "fault", "--node-id", "fault-node",
                    "--cpu-cores", "4", "--memory-gb", "8", "--runtime-adapter", "local-process", "--heartbeat-interval", "0.05",
                    "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")], stdout=stream, stderr=stream, env=environment)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                self.assertIsNone(node.poll(), (root / "node.log").read_text())
                config = helper.request("GET", url, "/v1/admin/workspace-config?pool=fault")
                if config["nodes"] and config["nodes"][0]["state"] == "applied":
                    break
                time.sleep(0.05)
            else:
                self.fail("workspace readiness failed")

            def run(task, code, expected="COMPLETED", identity=None):
                body = {"pool": "fault", "user_id": "alice", "session_id": task,
                    "command": ["python3", "-c", code], "reuse": True,
                    "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}}
                if identity:
                    body["workspace"] = {"id": identity}
                    body["reuse"] = False
                request = helper.request("POST", url, "/v1/submit", body)
                status = helper.wait_state(url, request["request_id"], expected, timeout=60)
                return status, helper.request("GET", url, "/v1/requests/" + request["request_id"] + "/logs")

            def wait_workspace(identity, state):
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    observed = helper.request("GET", url, "/v1/workspaces/" + identity)
                    if observed["state"] == state:
                        return observed
                    time.sleep(0.05)
                self.fail("workspace state mismatch: " + repr(observed) + "\n" + (root / "node.log").read_text()[-4000:])

            def release(identity):
                helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
                return wait_workspace(identity, "retained")

            for tool in ("mkfs", "fuse"):
                flag = root / (tool + "-fail")
                flag.touch()
                status, logs = run(tool + "-failure", "raise AssertionError('user command must not launch')", "FAILED")
                flag.unlink()
                self.assertEqual(status["failure_kind"], "infrastructure", logs)
                self.assertNotIn("AssertionError: user command must not launch", logs["stderr"])
                record = wait_workspace(status["workspace"]["id"], "needs_check")
                directory = root / "images" / ("cape-workspaces-" + str(os.geteuid())) / record["id"]
                self.assertTrue((directory / "workspace.img").is_file())
                before = (directory / "workspace.img").stat()
                with self.assertRaises(urllib.error.HTTPError):
                    helper.request("POST", url, "/v1/submit", {"pool": "fault", "user_id": "alice", "session_id": "unsafe-reopen", "workspace": {"id": record["id"]},
                        "command": ["true"], "reuse": False, "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
                after = (directory / "workspace.img").stat()
                self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns), (after.st_ino, after.st_size, after.st_mtime_ns))

            status, _ = run("busy", "from pathlib import Path; Path('marker').write_text('preserved')")
            record = helper.request("GET", url, "/v1/workspaces/" + status["workspace"]["id"])
            manager = WorkspaceManager(WorkspaceConfig.from_dict(record["config"]), "fault-node", "fault")
            with self.assertRaises(WorkspaceError):
                manager.prepare(record["id"], "alice", "competing-session")
            with self.assertRaises(WorkspaceError):
                manager.delete(record["id"], "alice")
            busy = subprocess.Popen(["/bin/sleep", "60"], cwd=record["path"])
            helper.request("POST", url, "/v1/workspaces/" + record["id"] + "/release", {})
            time.sleep(0.3)
            blocked = helper.request("GET", url, "/v1/workspaces/" + record["id"])
            self.assertNotEqual(blocked["state"], "retained", blocked)
            self.assertEqual((Path(record["path"]) / "marker").read_text(), "preserved")
            busy.terminate(); busy.wait(timeout=5); busy = None
            wait_workspace(record["id"], "retained")

            status, _ = run("wrong-mount", "from pathlib import Path; Path('marker').write_text('original-image')")
            record = release(status["workspace"]["id"])
            other = WorkspaceManager(WorkspaceConfig(image_root=str(root / "foreign-image"), image_size_bytes=64 * 1024 ** 2), "fault-node", "fault")
            other_id = new_workspace_id()
            handle = other.prepare(other_id, "alice", "foreign-test-session", create=True)
            (Path(handle.path) / "foreign-marker").write_text("foreign-image")
            other.release(other_id)
            source = Path(handle.path).parent / "workspace.img"
            foreign_mount = record["path"]
            with (root / "foreign.log").open("w") as stream:
                foreign = subprocess.Popen(["/usr/bin/fuse2fs", str(source), foreign_mount, "-o", "rw", "-f"], stdout=stream, stderr=stream)
            deadline = time.monotonic() + 10
            while not any(entry["target"] == foreign_mount for entry in mount_entries()) and time.monotonic() < deadline:
                self.assertIsNone(foreign.poll())
                time.sleep(0.02)
            failed, _ = run("wrong-mount-refused", "raise AssertionError('must not run on foreign image')", "FAILED", record["id"])
            self.assertEqual(failed["failure_kind"], "infrastructure")
            self.assertIsNone(foreign.poll(), "agent must not kill the foreign mount owner")
            self.assertEqual((Path(foreign_mount) / "foreign-marker").read_text(), "foreign-image")
            subprocess.run([os.environ.get("CAPE_WORKSPACE_FUSERMOUNT_BIN", "fusermount3"), "-u", foreign_mount], check=True, capture_output=True, timeout=10)
            foreign.wait(timeout=10); foreign = None; foreign_mount = None
            helper.request("POST", url, "/v1/workspaces/" + record["id"] + "/check", {})
            wait_workspace(record["id"], "retained")
            run("wrong-mount-recovered", "from pathlib import Path; assert Path('marker').read_text()=='original-image'", identity=record["id"])
            release(record["id"])
            other.delete(other_id, "alice")

            status, _ = run("daemon-death", "from pathlib import Path; Path('marker').write_text('crash-data')")
            identity = status["workspace"]["id"]
            directory = root / "images" / ("cape-workspaces-" + str(os.geteuid())) / identity
            manifest = json.loads((directory / "manifest.json").read_text())
            pid = manifest["daemon_pid"]
            self.assertIn(str(directory / "workspace.img").encode(), Path("/proc/{}/cmdline".format(pid)).read_bytes())
            os.kill(pid, signal.SIGKILL)
            wait_workspace(identity, "needs_check")
            self.assertTrue((directory / "workspace.img").is_file())

            for mode in ("bytes", "inodes"):
                if mode == "bytes":
                    code = "from pathlib import Path\nimport errno\nPath('marker').write_text('safe')\ntry:\n with open('large','wb') as f:\n  for _ in range(128): f.write(b'x'*1048576)\nexcept OSError as e:\n assert e.errno==errno.ENOSPC\nelse: raise AssertionError('expected bounded image exhaustion')\nassert Path('marker').read_text()=='safe'\nprint('ENOSPC preserved marker')"
                else:
                    code = "from pathlib import Path\nimport errno\nPath('marker').write_text('safe')\ntry:\n for i in range(8192):\n  d=Path(str(i//128)); d.mkdir(exist_ok=True); (d/str(i)).touch()\nexcept OSError as e:\n assert e.errno==errno.ENOSPC\nelse: raise AssertionError('expected bounded inode exhaustion')\nassert Path('marker').read_text()=='safe'\nprint('ENOSPC preserved marker')"
                status, logs = run(mode + "-exhaustion", code)
                self.assertIn("ENOSPC preserved marker", logs["stdout"])
                record = release(status["workspace"]["id"])
                self.assertEqual(list(Path(record["path"]).iterdir()), [], "writes escaped into underlying mountpoint")
            print(json.dumps({"binary": binary, "injected_format_backing_enospc": "passed", "injected_mount_failure": "passed",
                              "duplicate_writer": "passed", "active_delete_refusal": "passed", "busy_unmount": "passed",
                              "real_wrong_image_mount": "passed",
                              "real_fuse_death": "passed", "real_image_bytes_full": "passed", "real_image_inodes_full": "passed"}), flush=True)
            passed = True
        finally:
            if foreign is not None and foreign.poll() is None and foreign_mount:
                subprocess.run([os.environ.get("CAPE_WORKSPACE_FUSERMOUNT_BIN", "fusermount3"), "-u", foreign_mount], capture_output=True, timeout=10)
                try:
                    foreign.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    passed = False
            if busy is not None and busy.poll() is None:
                busy.terminate(); busy.wait(timeout=5)
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    node.kill(); node.wait(timeout=10)
            if controller is not None:
                helper.stop_controller(controller)
            mounted = any(entry["target"].startswith(str(root) + "/") for entry in mount_entries())
            if passed and not mounted:
                shutil.rmtree(root)
            else:
                print("retained_fault_root=" + str(root), flush=True)
            self.assertFalse(mounted, "test-owned mounts remain")
