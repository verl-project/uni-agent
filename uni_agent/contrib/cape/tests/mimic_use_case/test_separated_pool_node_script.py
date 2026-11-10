# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "separated_pool_node.sh"


class SeparatedPoolNodeScriptTests(unittest.TestCase):
    def test_teardown_waits_for_agent_to_consume_stop_session_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            state_path = root / "agent-state.json"
            cleanup_seen = root / "agent-cleanup-seen"
            terminated_early = root / "agent-terminated-early"
            remove_marker = root / "remove-called"
            force_seen = root / "force-seen"
            stop_flag = root / "stop"
            stop_flag.touch()

            fake_python = fake_bin / "python3"
            fake_python.write_text(textwrap.dedent("""\
                #!/usr/bin/env bash
                set -euo pipefail
                command_name="${1:-}:${2:-}"
                case "$command_name" in
                  node:enroll)
                    printf '{"local_sessions":["sess-test"],"running_commands":[]}\n' > "$FAKE_STATE_PATH"
                    on_term() {
                      if grep -Eq '"local_sessions"[[:space:]]*:[[:space:]]*\\[[[:space:]]*\\]' "$FAKE_STATE_PATH"; then
                        touch "$FAKE_CLEANUP_SEEN"
                      else
                        touch "$FAKE_TERMINATED_EARLY"
                      fi
                      exit 0
                    }
                    trap on_term TERM INT
                    cleaned=0
                    while :; do
                      if [ "$cleaned" -eq 0 ] && [ -f "$FAKE_REMOVE_MARKER" ]; then
                        sleep 0.6
                        printf '{"local_sessions":[],"running_commands":[]}\n' > "$FAKE_STATE_PATH"
                        cleaned=1
                      fi
                      sleep 0.05
                    done
                    ;;
                  node:drain)
                    case " $* " in
                      *" --force "*) touch "$FAKE_FORCE_SEEN" ;;
                      *) exit 3 ;;
                    esac
                    touch "$FAKE_REMOVE_MARKER"
                    ;;
                  *)
                    printf 'unexpected fake python invocation: %s\n' "$*" >&2
                    exit 2
                    ;;
                esac
                """))
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env.update({
                "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                "PYTHONPATH": str(SCRIPT.parents[2]),
                "CAPE_CONTROLLER": "off",
                "CAPE_URL": "http://controller.invalid:8765",
                "CAPE_NODE_ID": "pool-test",
                "CAPE_ALLOCATION_ID": "allocation-test",
                "CAPE_GPU_UUIDS": "GPU-test",
                "CAPE_RUNTIME_ADAPTERS": "local-process",
                "RESULTS_DIR": str(root / "results"),
                "POOL_STATE_DIR": str(root / "pool-state"),
                "CAPE_JOURNAL_PATH": str(root / "journal.sqlite"),
                "CAPE_AUTH_FILE": str(root / "auth.json"),
                "CAPE_ADMIN_TOKEN": "admin-token",
                "CAPE_NODE_TOKEN": "node-owner-token",
                "STOP_FLAG": str(stop_flag),
                "CAPE_BIN_DIR": str(root / "cape-bin"),
                "CAPE_BIN": str(fake_python),
                "CAPE_AGENT_STATE_PATH": str(state_path),
                "CAPE_AGENT_CLEANUP_TIMEOUT": "3",
                "FAKE_STATE_PATH": str(state_path),
                "FAKE_CLEANUP_SEEN": str(cleanup_seen),
                "FAKE_TERMINATED_EARLY": str(terminated_early),
                "FAKE_REMOVE_MARKER": str(remove_marker),
                "FAKE_FORCE_SEEN": str(force_seen),
            })

            started = time.monotonic()
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=10,
            )
            elapsed = time.monotonic() - started

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(cleanup_seen.exists(), result.stderr)
            self.assertFalse(terminated_early.exists(), result.stderr)
            self.assertTrue(force_seen.exists(), result.stderr)
            self.assertGreaterEqual(elapsed, 0.5)


if __name__ == "__main__":
    unittest.main()
