# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


RETIRE_SCRIPT = (
    Path(__file__).resolve().parent / "manual"
    / "retire_node.sh"
)


class ManualRetireNodeScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.calls_path = self.tmp_path / "cape-calls.jsonl"
        self.sequence_path = self.tmp_path / "utilization-sequence.json"
        self.index_path = self.tmp_path / "utilization-index"
        self.fake_cape = self.tmp_path / "cape"
        self.fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "if args == ['--version']:\n"
            "    print('rscape 0.1.0')\n"
            "    raise SystemExit(0)\n"
            "with open(os.environ['FAKE_CAPE_CALLS'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "if args and args[0] == 'utilization':\n"
            "    sequence = json.loads(pathlib.Path(os.environ['FAKE_UTIL_SEQUENCE']).read_text())\n"
            "    index_path = pathlib.Path(os.environ['FAKE_UTIL_INDEX'])\n"
            "    index = int(index_path.read_text()) if index_path.exists() else 0\n"
            "    payload = sequence[min(index, len(sequence) - 1)]\n"
            "    index_path.write_text(str(index + 1))\n"
            "    print(json.dumps(payload, sort_keys=True))\n"
            "elif args[:2] == ['node', 'drain']:\n"
            "    print(json.dumps({'node_id': args[2], 'state': 'DRAINING'}, sort_keys=True))\n"
            "else:\n"
            "    sys.stderr.write('unexpected fake cape argv: {}\\n'.format(args))\n"
            "    raise SystemExit(3)\n",
            encoding="utf-8",
        )
        self.fake_cape.chmod(
            self.fake_cape.stat().st_mode | stat.S_IXUSR
        )
        self.controller_env = self.tmp_path / "controller.env"
        self.controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_ADMIN_TOKEN=admin-secret-value\n"
            "export CAPE_JOIN_TOKEN=join-secret-value\n"
            "export RUN_ID=run-a\n"
            "export CASE=manual\n"
            "export CAPE_RETIRE_BIN=/foreign/controller/cape\n",
            encoding="utf-8",
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_")
            and not key.startswith("FAKE_")
            and key not in ("REPO_ROOT", "BUNDLE", "RUN_ID", "CASE")
        }
        self.env.update(
            {
                "CAPE_RETIRE_BIN": str(self.fake_cape),
                "FAKE_CAPE_CALLS": str(self.calls_path),
                "FAKE_UTIL_SEQUENCE": str(self.sequence_path),
                "FAKE_UTIL_INDEX": str(self.index_path),
            }
        )

    @staticmethod
    def utilization(nodes):
        return {
            "pool": "pool-a",
            "nodes": len(nodes),
            "active_nodes": sum(
                node.get("state") == "ACTIVE" for node in nodes
            ),
            "registering_nodes": sum(
                node.get("state") == "REGISTERING" for node in nodes
            ),
            "gpus_schedulable": sum(
                node.get("gpus_total", 0)
                for node in nodes
                if node.get("state") == "ACTIVE"
            ),
            "node_states": nodes,
        }

    @staticmethod
    def node(
        node_id="node-a",
        allocation_id="alloc-a",
        state="ACTIVE",
        sessions=(),
        requests=0,
    ):
        return {
            "node_id": node_id,
            "allocation_id": allocation_id,
            "state": state,
            "gpus_total": 1,
            "gpus_available": 1 if state == "ACTIVE" else 0,
            "active_sessions": list(sessions),
            "active_session_count": len(sessions),
            "nonterminal_requests": requests,
        }

    def run_retire(self, sequence, *arguments, env=None, check=False):
        self.sequence_path.write_text(
            json.dumps(sequence), encoding="utf-8"
        )
        self.index_path.write_text("0", encoding="utf-8")
        self.calls_path.unlink(missing_ok=True)
        return subprocess.run(
            [str(RETIRE_SCRIPT), *arguments],
            env=env or self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def calls(self):
        if not self.calls_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.calls_path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def output_records(result):
        return [json.loads(line) for line in result.stdout.splitlines()]

    def common_args(self):
        return ("--controller-env", str(self.controller_env))

    def test_graceful_allocation_drain_wait_remove_order_and_reason(self):
        sequence = [
            self.utilization(
                [self.node(sessions=("sess-a",), requests=1)]
            ),
            self.utilization(
                [
                    self.node(
                        state="DRAINING",
                        sessions=("sess-a",),
                        requests=1,
                    )
                ]
            ),
            self.utilization([self.node(state="DRAINING")]),
            self.utilization([]),
        ]
        result = self.run_retire(
            sequence,
            *self.common_args(),
            "--allocation-id",
            "alloc-a",
            "--reason",
            "maintenance-window",
            "--timeout",
            "5",
            "--poll-interval",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(
            [call[:2] for call in calls],
            [
                ["utilization", "--pool"],
                ["node", "drain"],
                ["utilization", "--pool"],
                ["utilization", "--pool"],
                ["utilization", "--pool"],
            ],
        )
        drain = calls[1]
        self.assertEqual(
            drain,
            [
                "node", "drain", "node-a",
                "--controller-url",
                "http://controller.example:18892",
                "--token",
                "admin-secret-value",
            ],
        )
        records = self.output_records(result)
        self.assertEqual(
            [record["phase"] for record in records],
            ["before", "drain", "wait", "wait", "wait", "after"],
        )
        self.assertNotIn("admin-secret-value", result.stdout + result.stderr)
        self.assertNotIn("join-secret-value", result.stdout + result.stderr)
        self.assertIn("pool-a is now empty and reconfigurable", result.stderr)
        self.assertIn("admin pool edit", result.stderr)

    def test_default_rscape_skips_retirement_venv_and_reports_version(self):
        binary_dir = self.tmp_path / "rscape-bin"
        binary_dir.mkdir()
        rscape = binary_dir / "rscape"
        shutil.copy2(self.fake_cape, rscape)
        rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
        venv_root = self.tmp_path / "must-not-create-retire-venv"
        env = dict(self.env)
        env.pop("CAPE_RETIRE_BIN")
        env.update({
            "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
            "CAPE_RETIRE_VENV_ROOT": str(venv_root),
            "CAPE_RETIRE_VENV": str(venv_root / "retire"),
        })
        sequence = [self.utilization([self.node()]), self.utilization([])]

        result = self.run_retire(
            sequence, *self.common_args(), "--allocation-id", "alloc-a",
            "--force", env=env,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(venv_root.exists())
        self.assertIn("cape_implementation=rscape", result.stderr)
        self.assertIn("cape_version=0.1.0", result.stderr)

    def test_node_id_resolves_exactly_one_allocation(self):
        sequence = [
            self.utilization(
                [
                    self.node(node_id="node-a", allocation_id="alloc-a"),
                    self.node(node_id="node-b", allocation_id="alloc-b"),
                ]
            ),
            self.utilization(
                [
                    self.node(
                        node_id="node-a",
                        allocation_id="alloc-a",
                        state="DRAINING",
                    ),
                    self.node(node_id="node-b", allocation_id="alloc-b"),
                ]
            ),
            self.utilization(
                [self.node(node_id="node-b", allocation_id="alloc-b")]
            ),
        ]
        result = self.run_retire(
            sequence,
            *self.common_args(),
            "--node-id",
            "node-a",
            "--poll-interval",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        admin_calls = [call for call in self.calls() if call[0] == "node"]
        self.assertEqual(len(admin_calls), 1)
        for call in admin_calls:
            self.assertEqual(
                call[2], "node-a"
            )
        self.assertNotIn("now empty and reconfigurable", result.stderr)

    def test_drain_only_never_removes_and_verifies_unschedulable(self):
        result = self.run_retire(
            [
                self.utilization([self.node(requests=1)]),
                self.utilization(
                    [self.node(state="DRAINING", requests=1)]
                ),
            ],
            *self.common_args(),
            "--allocation-id",
            "alloc-a",
            "--drain-only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(
            sum(call[:2] == ["node", "drain"] for call in calls),
            1,
        )
        self.assertFalse(
            any(call[:2] == ["node", "exit"] for call in calls)
        )
        self.assertEqual(
            [record["phase"] for record in self.output_records(result)],
            ["before", "drain", "after"],
        )
        self.assertNotIn("now empty and reconfigurable", result.stderr)

    def test_force_cancels_and_waits_for_authoritative_removal(self):
        result = self.run_retire(
            [
                self.utilization(
                    [self.node(sessions=("sess-a",), requests=1)]
                ),
                self.utilization([]),
            ],
            *self.common_args(),
            "--allocation-id",
            "alloc-a",
            "--force",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        remove = next(call for call in calls if call[:2] == ["node", "drain"])
        self.assertIn("--force", remove)
        self.assertEqual(
            [record["phase"] for record in self.output_records(result)],
            ["before", "drain", "wait", "after"],
        )
        self.assertIn("pool-a is now empty and reconfigurable", result.stderr)

    def test_timeout_is_nonzero_and_never_escalates_to_force(self):
        active = self.utilization(
            [
                self.node(
                    state="DRAINING",
                    sessions=("sess-a",),
                    requests=1,
                )
            ]
        )
        result = self.run_retire(
            [
                self.utilization(
                    [self.node(sessions=("sess-a",), requests=1)]
                ),
                active,
                active,
            ],
            *self.common_args(),
            "--allocation-id",
            "alloc-a",
            "--timeout",
            "0",
            "--poll-interval",
            "0",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        calls = self.calls()
        self.assertFalse(
            any(call[:2] == ["node", "exit"] for call in calls)
        )
        self.assertNotIn("--force", [value for call in calls for value in call])
        self.assertEqual(
            [record["phase"] for record in self.output_records(result)],
            ["before", "drain", "wait", "after"],
        )

    def test_each_discovered_member_is_drained_without_touching_another_allocation(self):
        other = self.node(node_id="other", allocation_id="other-allocation")
        result = self.run_retire([
            self.utilization([self.node(node_id="first"), self.node(node_id="second"), other]),
            self.utilization([other]),
        ], *self.common_args(), "--allocation-id", "alloc-a", "--poll-interval", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[2] for call in self.calls() if call[:2] == ["node", "drain"]], ["first", "second"])

    def test_absent_allocation_is_idempotent_unless_strict(self):
        sequence = [self.utilization([])]
        result = self.run_retire(
            sequence,
            *self.common_args(),
            "--allocation-id",
            "alloc-missing",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already absent", result.stderr)
        self.assertFalse(any(call[0] == "admin" for call in self.calls()))
        self.assertEqual(
            [record["phase"] for record in self.output_records(result)],
            ["before", "after"],
        )

        strict = self.run_retire(
            sequence,
            *self.common_args(),
            "--allocation-id",
            "alloc-missing",
            "--strict",
        )
        self.assertEqual(strict.returncode, 1)
        self.assertIn("already absent", strict.stderr)
        self.assertFalse(any(call[0] == "admin" for call in self.calls()))

    def test_node_id_rejects_unknown_missing_allocation_and_ambiguity(self):
        cases = (
            (self.utilization([]), "unknown node", "node-a"),
            (
                self.utilization(
                    [self.node(node_id="node-a", allocation_id=None)]
                ),
                "has no allocation",
                "node-a",
            ),
            (
                self.utilization(
                    [
                        self.node(node_id="node-a", allocation_id="alloc-a"),
                        self.node(node_id="node-a", allocation_id="alloc-b"),
                    ]
                ),
                "ambiguous",
                "node-a",
            ),
        )
        for payload, message, node_id in cases:
            with self.subTest(message=message):
                result = self.run_retire(
                    [payload],
                    *self.common_args(),
                    "--node-id",
                    node_id,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse(any(call[0] == "admin" for call in self.calls()))

    def test_token_override_wins_and_never_leaks(self):
        result = self.run_retire(
            [self.utilization([self.node()]), self.utilization([])],
            *self.common_args(),
            "--allocation-id",
            "alloc-a",
            "--force",
            "--token",
            "override-secret-value",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        flattened = [value for call in self.calls() for value in call]
        self.assertIn("override-secret-value", flattened)
        self.assertNotIn("admin-secret-value", flattened)
        self.assertNotIn(
            "override-secret-value", result.stdout + result.stderr
        )
        self.assertNotIn("admin-secret-value", result.stdout + result.stderr)

    def test_requires_one_identity_and_rejects_conflicting_modes(self):
        payload = [self.utilization([])]
        cases = (
            ((), "exactly one"),
            (
                ("--allocation-id", "alloc-a", "--node-id", "node-a"),
                "exactly one",
            ),
            (
                ("--allocation-id", "alloc-a", "--force", "--drain-only"),
                "mutually exclusive",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.run_retire(
                    payload, *self.common_args(), *arguments
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_help_documents_agent_self_exit_and_cluster_ownership(self):
        result = subprocess.run(
            [str(RETIRE_SCRIPT), "--help"],
            env=self.env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("graceful", result.stdout)
        self.assertIn("--drain-only", result.stdout)
        self.assertIn("--force", result.stdout)
        self.assertIn("logical", result.stdout)
        self.assertIn("register_node.sh", result.stdout)
        self.assertIn("exits", result.stdout)
        self.assertIn("does not release", result.stdout)
        self.assertNotIn("must stop that foreground process", result.stdout)


if __name__ == "__main__":
    unittest.main()
