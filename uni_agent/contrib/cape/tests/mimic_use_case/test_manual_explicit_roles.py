# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real controller/user Makefile smoke; opt in with CAPE_ROLE_TEST_BINARIES."""
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import tempfile
import unittest

MANUAL = Path(__file__).with_name("manual")


class ExplicitRoleTests(unittest.TestCase):
    def test_private_credential_executor_rejects_public_files(self):
        with tempfile.TemporaryDirectory() as directory:
            credential = Path(directory) / "user.json"
            credential.write_text('{"secret":"test-only"}')
            credential.chmod(0o644)
            result = subprocess.run(
                ["bash", str(MANUAL / "credential_exec.sh"), str(credential), "true"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(os.environ.get("CAPE_ROLE_TEST_BINARIES"), "explicit real CLI binaries required")
    def test_make_controller_user_and_restart_without_handoffs(self):
        for binary in os.environ["CAPE_ROLE_TEST_BINARIES"].split(os.pathsep):
            with self.subTest(binary=binary):
                root = Path(tempfile.mkdtemp(prefix="cape-explicit-role-"))
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                env = {k: v for k, v in os.environ.items() if not k.startswith("CAPE_")}
                env.update(CAPE_BIN=binary, CAPE_SUBMIT_BIN=binary, CAPE_NODE_BIN=binary,
                           CAPE_RETIRE_BIN=binary, CAPE_STATUS_BIN=binary,
                           CAPE_PORT=str(port), CAPE_URL=f"http://127.0.0.1:{port}",
                           CAPE_BIND_HOST="127.0.0.1", CAPE_SHUTDOWN_SETTLE="0")
                admin = root / "credentials/admin.json"
                user = root / "credentials/alice.json"
                base = ["make", "-C", str(MANUAL), "LOG_DIR=" + str(root / "logs"),
                        "BUNDLE=" + str(root), "CAPE_STATE_DIR=" + str(root / "instance"),
                        "CAPE_ADMIN_TOKEN_FILE=" + str(admin), "CAPE_USER_TOKEN_FILE=" + str(user),
                        "CAPE_USER=alice"]

                def make(target):
                    result = subprocess.run(base + [target], env=env, capture_output=True,
                                            text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    return result

                passed = False
                try:
                    make("controller")
                    self.assertTrue(admin.exists())
                    self.assertFalse(user.exists())
                    self.assertFalse(list(root.rglob("*.env")))
                    make("user")
                    original = user.read_bytes()
                    make("user")
                    self.assertEqual(user.read_bytes(), original)
                    result = subprocess.run([binary, "user", "pools", "--controller-url",
                                             env["CAPE_URL"], "--token-file", str(user)],
                                            env=env, capture_output=True, text=True, check=True)
                    pools = json.loads(result.stdout)
                    self.assertEqual({p["name"] for p in pools}, {"exclusive-pool", "shared-pool"})
                    bob = root / "credentials/bob.json"
                    subprocess.run([binary, "user", "token", "create", "--controller-url",
                                    env["CAPE_URL"], "--user", "bob", "--token-out", str(bob)],
                                   env=env, capture_output=True, text=True, check=True)
                    queued = subprocess.run(
                        [binary, "run", "--controller-url", env["CAPE_URL"], "--token-file", str(user),
                         "--pool", "exclusive-pool", "--session-id", "ownership-check", "--gpus", "0", "--", "true"],
                        env=env, capture_output=True, text=True, check=True).stdout.strip()
                    for verb in ("status", "logs", "cancel"):
                        denied = subprocess.run([binary, verb, queued, "--controller-url",
                                                 env["CAPE_URL"], "--token-file", str(bob)],
                                                env=env, capture_output=True, text=True)
                        self.assertNotEqual(denied.returncode, 0, verb)
                        self.assertRegex(denied.stderr, "(?i)forbidden|denied|403|not found|404")
                    subprocess.run([binary, "cancel", queued, "--controller-url", env["CAPE_URL"],
                                    "--token-file", str(user)], env=env, capture_output=True, check=True)
                    if os.environ.get("CAPE_ROLE_GPU_UUID"):
                        env.update(CAPE_GPU_UUIDS=os.environ["CAPE_ROLE_GPU_UUID"],
                                   CAPE_RUNTIME_ADAPTERS="local-process",
                                   CAPE_NODE_CPU_CORES="2", CAPE_NODE_MEMORY_GB="4",
                                   CAPE_NODE_BUNDLE=str(root / "nodes"))
                        for pool in ("exclusive-pool", "shared-pool"):
                            subprocess.run([binary, "admin", "set-workspace-config",
                                            "--controller-url", env["CAPE_URL"], "--token-file", str(admin),
                                            "--pool", pool, "--image-root", str(root / "images"),
                                            "--image-size-gib", "1"], env=env, check=True,
                                           capture_output=True, text=True)
                        for role in ("exclusive", "shared"):
                            make("register_" + role)
                            make(role + "_ready")
                            membership = json.loads((root / "logs" / (role + "_node.json")).read_text())
                            self.assertEqual(membership["pool"], role + "-pool")
                            self.assertNotIn("token", membership)
                            denied = subprocess.run([binary, "node", "drain", membership["node_id"],
                                                     "--controller-url", env["CAPE_URL"], "--token-file", str(bob)],
                                                    env=env, capture_output=True, text=True)
                            self.assertNotEqual(denied.returncode, 0)
                            self.assertRegex(denied.stderr, "(?i)forbidden|denied|403|not found|404")
                            submitted = subprocess.run(
                                ["bash", str(MANUAL / "submit.sh"), "--controller-url", env["CAPE_URL"],
                                 "--token-file", str(user), "--pool", role + "-pool", "--user", "alice",
                                 "--runtime-adapter", "local-process", "--isolation-policy", "none",
                                 "--gpus", "1", "--", "python3", "-c",
                                 "import torch; assert torch.cuda.device_count()==1; "
                                 "assert torch.ones(8,device='cuda').sum().item()==8"],
                                env=env, capture_output=True, text=True, check=True, timeout=60)
                            request = submitted.stdout.strip()
                            finished = subprocess.run([binary, "wait", request, "--controller-url",
                                                       env["CAPE_URL"], "--token-file", str(user),
                                                       "--timeout", "90"], env=env, capture_output=True,
                                                      text=True, check=True, timeout=100)
                            self.assertIn('"COMPLETED"', finished.stdout)
                            status = subprocess.run([binary, "status", request, "--controller-url",
                                                     env["CAPE_URL"], "--token-file", str(user)],
                                                    env=env, capture_output=True, text=True, check=True)
                            workspace = json.loads(status.stdout)["workspace"]["id"]
                            denied = subprocess.run([binary, "workspace", "status", workspace,
                                                     "--controller-url", env["CAPE_URL"], "--token-file", str(bob)],
                                                    env=env, capture_output=True, text=True)
                            self.assertNotEqual(denied.returncode, 0)
                            self.assertRegex(denied.stderr, "(?i)forbidden|denied|403|not found|404")
                    make("shutdown")
                    admin_before = admin.read_bytes()
                    make("controller")
                    self.assertEqual(admin.read_bytes(), admin_before)
                    self.assertEqual(user.read_bytes(), original)
                    passed = True
                finally:
                    try:
                        make("shutdown")
                    except Exception:
                        print("retained explicit-role fixture: " + str(root), flush=True)
                        raise
                    if passed:
                        shutil.rmtree(root)
                    else:
                        print("retained explicit-role fixture: " + str(root), flush=True)
