# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from pycape.workspace import WorkspaceConfig
from pycape.workspace_fs import WorkspaceManager, WorkspaceError, mount_entries
from tests.rscape import test_functional_parity as parity
from tests.workspace_fixtures import copy_workspace_evidence


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_MAINTENANCE_TEST") == "1", "requires real workspace maintenance environment")
class WorkspaceMaintenanceRecoveryTests(unittest.TestCase):
    def test_release_reclaim_and_cross_language_reopen(self):
        for first, second in ((parity.PYCAPE_BIN, parity.RSCAPE_BIN), (parity.RSCAPE_BIN, parity.PYCAPE_BIN)):
            with self.subTest(first=first, second=second):
                self.exercise(first, second, False)

    def test_agent_crash_preserves_copy_helper_fence_and_recovery(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                self.exercise(binary, binary, True)

    def exercise(self, first, second, crash):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-maintenance-recovery-"))
        controller = node = None
        passed = False
        try:
            controller, url, _ = helper.start_controller(first, str(root), "--idle-session-timeout-seconds", "180")
            policy = {"backing_reserve_bytes": 0, "timeout_seconds": 30, "reclaim_on_release": not crash,
                      "reclaim_min_allocated_bytes": 32*1024**2, "reclaim_cooldown_seconds": 0,
                      "reclaim_idle_seconds": 1, "reclaim_start_interval_seconds": 1}
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "maint", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 128*1024**2, "maintenance": policy}})
            flag = root / "hold-copy"
            entered = root / "copy-entered"
            copy = root / "copy-wrapper.sh"
            copy.write_text('#!/bin/sh\nif [ -e "' + str(flag) + '" ]; then\n'
                            '  printf entered > "' + str(entered) + '"\n  sleep 8\nfi\nexec /usr/bin/cp "$@"\n')
            copy.chmod(0o700)
            env = dict(os.environ, CAPE_WORKSPACE_CP_BIN=str(copy))

            def launch(binary):
                args = [binary, "node", "enroll", *helper.node_credentials(url, "node", "maint"), "--controller-url", url, "--pool", "maint", "--node-id", "node",
                        "--runtime-adapter", "local-process", "--cpu-cores", "4", "--memory-gb", "8",
                        "--heartbeat-interval", "0.05", "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")]
                with (root / "node.log").open("a") as stream:
                    return subprocess.Popen(args, stdout=stream, stderr=stream, env=env)

            def ready(previous=None):
                deadline=time.monotonic()+90
                observed_heartbeat = None
                while time.monotonic()<deadline:
                    self.assertIsNone(node.poll(),(root/"node.log").read_text()[-5000:])
                    rows=helper.request("GET",url,"/v1/admin/workspace-config?pool=maint")["nodes"]
                    if rows and rows[0]["state"]=="applied":
                        if previous is not None:
                            self.assertEqual(rows[0]["agent_epoch"], previous)
                        if observed_heartbeat is not None and rows[0]["last_heartbeat_at"] > observed_heartbeat:
                            return rows[0]["agent_epoch"]
                        observed_heartbeat = rows[0]["last_heartbeat_at"]
                    time.sleep(.05)
                self.fail("node readiness timeout")

            def command(task, code, identity=None):
                body={"pool":"maint","user_id":"alice","session_id":task,"reuse":False,"command":["/bin/sh","-ec",code],
                      "resources":{"gpus":0,"runtime_adapter":"local-process","isolation_policy":"none"}}
                if identity:body["workspace"]={"id":identity}
                req=helper.request("POST",url,"/v1/submit",body)
                return helper.wait_state(url,req["request_id"],"COMPLETED",timeout=90)

            def wait_workspace(identity, predicate):
                deadline=time.monotonic()+90
                record=None
                while time.monotonic()<deadline:
                    record=helper.request("GET",url,"/v1/workspaces/"+identity)
                    if predicate(record):return record
                    time.sleep(.05)
                self.fail(repr(record))

            node=launch(first);epoch=ready()
            result=command("original", 'dd if=/dev/urandom of=keep bs=1M count=8 status=none; '
                           'sha256sum keep > checksum; dd if=/dev/urandom of=discard bs=1M count=64 status=none; rm discard')
            identity=result["workspace"]["id"]
            before=helper.request("GET",url,"/v1/workspaces/"+identity)
            helper.request("POST",url,"/v1/workspaces/"+identity+"/release",{})
            retained=wait_workspace(identity, lambda r:r["state"]=="retained" and
                                    (crash or (r.get("operation",{}).get("action")=="reclaim" and r["operation"]["state"]=="completed")))
            if not crash:
                self.assertEqual(retained["current_image_size_bytes"],128*1024**2)
                self.assertLess(retained["space"]["allocated_bytes"],retained["operation"]["details"]["before_allocated_bytes"]-32*1024**2)
                old_operation=retained["operation"]["id"]
                time.sleep(.3)
                self.assertEqual(helper.request("GET",url,"/v1/workspaces/"+identity)["operation"]["id"],old_operation)
            else:
                flag.touch()
                helper.request("POST",url,"/v1/workspaces/"+identity+"/grow",{"target_bytes":256*1024**2})
                deadline=time.monotonic()+30
                while not entered.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(entered.exists())
                node.kill();node.wait(timeout=10)
                manager=WorkspaceManager(WorkspaceConfig.from_dict(retained["config"]),"node","maint")
                with self.assertRaisesRegex(WorkspaceError,"writer"):
                    manager.recover_maintenance(identity,"alice")
                flag.unlink()
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    try:
                        lease=manager._lease(manager._directory(identity,False));os.close(lease);break
                    except WorkspaceError:time.sleep(.05)
                else:self.fail("orphan copy helper did not release its inherited fence")
            if node.poll() is None:
                self.assertFalse(any(entry["target"].startswith(str(root) + "/") for entry in mount_entries()))
                # Retained, idle restart; SIGTERM would intentionally retire this enrollment.
                node.kill();node.wait(timeout=30)
            node=launch(second);ready(epoch)
            if crash:
                wait_workspace(identity,lambda r:r.get("operation",{}).get("state")=="failed")
                recovery=helper.request("POST",url,"/v1/workspaces/"+identity+"/recover",{})
                restored=wait_workspace(identity,lambda r:r["operation"]["id"]==recovery["operation"]["id"] and r["operation"]["state"]!="pending")
                self.assertEqual(restored["state"],"retained")
                self.assertEqual(restored["operation"]["state"],"completed")
                self.assertEqual(restored["current_image_size_bytes"],128*1024**2)
            resumed=command("resumed", "sha256sum -c checksum; test ! -e discard",identity)
            self.assertEqual(resumed["workspace"]["id"],identity)
            helper.request("POST",url,"/v1/workspaces/"+identity+"/release",{})
            wait_workspace(identity,lambda r:r["state"]=="retained" and r.get("operation",{}).get("state")!="pending")
            passed=True
            print(json.dumps({"first":first,"second":second,"crash":crash,
                              "inherited_copy_fence":"passed" if crash else "not-applicable",
                              "automatic_reclaim_once":"passed" if not crash else "not-applicable",
                              "reopen_and_content":"passed"}),flush=True)
        finally:
            for process in (node,controller):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:
                    for stream in (process.stdin,process.stdout,process.stderr):
                        if stream is not None:stream.close()
            output=os.environ.get("CAPE_WORKSPACE_MAINTENANCE_OUTPUT")
            if output:
                dest=Path(output)/("recovery-"+Path(first).name+"-"+str(crash))
                copy_workspace_evidence(root,dest)
            if any(row["target"].startswith(str(root)+"/") for row in mount_entries()):
                raise AssertionError("recovery test mounts remain: "+str(root))
            if passed:shutil.rmtree(root)
            else:print("maintenance recovery diagnostic root="+str(root),flush=True)
