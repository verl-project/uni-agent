# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import re
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape.observability import (  # noqa: E402
    NODE_REASON_MESSAGES,
    REQUEST_REASON_MESSAGES,
    SESSION_REASON_MESSAGES,
)


REQUEST_REASONS = {
    "pool_empty",
    "node_probing",
    "no_capable_node",
    "waiting_for_gpu",
    "waiting_for_cpu_or_memory",
    "waiting_for_runtime_capability",
    "waiting_for_image_cache_or_prepare",
    "waiting_for_session",
    "allocation_expiring",
    "node_draining",
    "quota_exceeded",
    "pool_session_limit",
    "fairness_delay",
    "node_reregistered",
    "controller_restart",
    "reconcile_missing",
    "command_failed",
    "infrastructure_failed",
    "command_status_unknown",
    "cancelled",
    "node_lost",
    "max_lifetime",
    "allocation_removed",
    "node_removed",
    "allocation_drain",
    "operator_drain",
    "stopped",
}
SESSION_REASONS = {
    "command_failed",
    "infrastructure_failed",
    "node_lost",
    "idle_timeout",
    "max_lifetime",
    "stopped",
    "evict_for_queued_work",
    "allocation_drain",
    "allocation_removed",
    "node_removed",
    "operator_drain",
    "allocation_expiring",
    "reconcile_missing",
    "reconciled",
    "node_reregistered",
    "lost_session_cleanup_confirmed",
}
NODE_REASONS = {
    "heartbeat_timeout",
    "heartbeat_lost",
    "allocation_expiring",
    "allocation_deadline",
    "allocation_drain",
    "allocation_removed",
    "node_removed",
    "operator_drain",
    "remove_pending_sessions",
}


class ReasonVocabularyTests(unittest.TestCase):
    def test_every_emitted_reason_has_an_entity_scoped_message(self):
        self.assertEqual(REQUEST_REASONS - set(REQUEST_REASON_MESSAGES), set())
        self.assertEqual(SESSION_REASONS - set(SESSION_REASON_MESSAGES), set())
        self.assertEqual(NODE_REASONS - set(NODE_REASON_MESSAGES), set())

    def test_rust_and_python_reason_vocabularies_are_closed(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "rscape"
            / "src"
            / "controller.rs"
        ).read_text(encoding="utf-8")
        for reason in REQUEST_REASONS | SESSION_REASONS | NODE_REASONS:
            self.assertRegex(
                source,
                r'"{}"\s*=>\s*(?:\{{\s*)?Some\('.format(
                    re.escape(reason)
                ),
            )

    def test_literal_controller_emitters_are_covered_automatically(self):
        controller_path = (
            Path(__file__).resolve().parents[2]
            / "pycape"
            / "controller.py"
        )
        tree = ast.parse(controller_path.read_text(encoding="utf-8"))
        discovered = {
            "request": set(),
            "session": set(),
            "node": set(),
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and any(isinstance(target, ast.Attribute) and target.attr == "reason" for target in targets)
                ):
                    discovered["request"].add(value.value)
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "_set_queued_reason" and len(node.args) >= 2:
                reason = node.args[1]
                if isinstance(reason, ast.Constant) and isinstance(reason.value, str):
                    discovered["request"].add(reason.value)
            if node.func.attr == "_emit" and len(node.args) >= 4:
                event_type, reason = node.args[0], node.args[3]
                if (
                    isinstance(event_type, ast.Constant)
                    and isinstance(event_type.value, str)
                    and isinstance(reason, ast.Constant)
                    and isinstance(reason.value, str)
                ):
                    entity = {
                        "request_state": "request",
                        "request_queued": "request",
                        "session_state": "session",
                        "node_state": "node",
                    }.get(event_type.value)
                    if entity:
                        discovered[entity].add(reason.value)

        self.assertEqual(discovered["request"] - set(REQUEST_REASON_MESSAGES), set())
        self.assertEqual(discovered["session"] - set(SESSION_REASON_MESSAGES), set())
        self.assertEqual(discovered["node"] - set(NODE_REASON_MESSAGES), set())

        rust = (
            Path(__file__).resolve().parents[2]
            / "rscape"
            / "src"
            / "controller.rs"
        ).read_text(encoding="utf-8")
        rust_request_assignments = set(
            re.findall(r'request\.reason\s*=\s*Some\("([a-z0-9_]+)"', rust)
        )
        self.assertEqual(
            rust_request_assignments - set(REQUEST_REASON_MESSAGES),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
