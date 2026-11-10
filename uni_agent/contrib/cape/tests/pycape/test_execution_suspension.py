# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from pycape import CAPEController, SessionState, RequestState
from pycape.session_api import execution_action
from tests.pycape.test_pycape import controller, register_gpu_node, spec


class ExecutionSuspensionTests(unittest.TestCase):
    def setUp(self):
        self.controller, self.clock = controller()
        register_gpu_node(self.controller)
        request = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        command = self.controller.node_heartbeat("node-a")["commands"][0]
        self.controller.report_command_result("node-a", command["command_id"], exit_code=0)
        self.session = self.controller.sessions[request.execution_session_id]
        self.workspace = self.controller.workspaces[self.session.workspace["id"]]

    def acknowledge(self):
        self.workspace["state"] = "retained"
        self.controller._emit("workspace_state", self.workspace["id"], "retained", None, dict(self.workspace))
        return self.controller.node_heartbeat("node-a", status={"authoritative": True, "sessions": [],
            "suspended_sessions": {self.session.execution_session_id: self.session.suspend_version}})

    def test_suspend_ack_releases_accounting_and_resume_reuses_identity(self):
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.assertEqual(self.session.state, SessionState.SUSPENDING)
        self.assertEqual(self.controller.nodes["node-a"].used_cpu_cores, 1)
        self.acknowledge()
        self.assertEqual(self.session.state, SessionState.SUSPENDED)
        self.assertEqual(self.controller.nodes["node-a"].used_cpu_cores, 0)
        restored = CAPEController.reconcile_from_journal(self.controller.journal, clock=self.clock)
        self.assertEqual(restored.sessions[self.session.execution_session_id].state, SessionState.SUSPENDED)
        self.assertEqual(restored.nodes["node-a"].used_cpu_cores, 0)
        request = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        self.assertEqual(request.execution_session_id, self.session.execution_session_id)
        self.assertEqual(self.session.workspace["id"], self.workspace["id"])
        self.assertFalse(self.session.workspace["create"])
        commands = self.controller.node_heartbeat("node-a")["commands"]
        run = next(c for c in commands if c.get("action") == "run")
        self.assertEqual(run["session_spec"]["suspend_version"], 1)

    def test_purge_is_explicit_and_does_not_delete_workspace(self):
        with self.assertRaisesRegex(ValueError, "suspended"):
            execution_action(self.controller, self.session.execution_session_id, "purge", None)
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        execution_action(self.controller, self.session.execution_session_id, "purge", None)
        self.assertEqual(self.session.state, SessionState.PURGING)
        self.controller.node_heartbeat("node-a", status={"authoritative": True, "sessions": [],
            "purged_executions": [self.session.execution_session_id]})
        self.assertEqual(self.session.state, SessionState.PURGED)
        self.assertEqual(self.workspace["state"], "retained")
        self.assertIn(("user-a", "sequence"), self.controller.logical_sessions)

    def test_resume_reclaims_idle_capacity_and_waits_for_retention_ack(self):
        node = self.controller.nodes["node-a"]
        node.resources.cpu_cores = 1
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        other_request = self.controller.submit_workload(spec(session_id="other"))
        self.controller.schedule_once()
        command = self.controller.node_heartbeat("node-a")["commands"][0]
        self.controller.report_command_result("node-a", command["command_id"], exit_code=0)
        other = self.controller.sessions[other_request.execution_session_id]
        request = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.QUEUED)
        self.assertEqual(other.state, SessionState.SUSPENDING)
        self.assertEqual(node.used_cpu_cores, 1)
        self.controller.workspaces[other.workspace["id"]]["state"] = "retained"
        self.controller.node_heartbeat("node-a", status={"authoritative": True, "sessions": [],
            "suspended_sessions": {self.session.execution_session_id: self.session.suspend_version,
                                   other.execution_session_id: other.suspend_version}})
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(request.execution_session_id, self.session.execution_session_id)
        self.assertEqual(other.state, SessionState.SUSPENDED)
        self.assertEqual(node.used_cpu_cores, 1)

    def test_suspended_execution_releases_pool_slot_and_resume_checks_limits(self):
        self.controller.policy.max_active_sessions_per_pool = 1
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        other = self.controller.submit_workload(spec(session_id="occupant"))
        self.controller.schedule_once()
        self.assertEqual(other.state, RequestState.RUNNING)
        resumed = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        self.assertEqual(resumed.state, RequestState.QUEUED)
        self.assertEqual(resumed.reason, "pool_session_limit")

    def test_retained_resume_checks_current_gpu_quota(self):
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        self.controller.policy.max_active_gpu_sessions_per_user = 0
        resumed = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        self.assertEqual(resumed.state, RequestState.QUEUED)
        self.assertEqual(resumed.reason, "quota_exceeded")

    def test_confirmed_suspend_does_not_block_pool_edit_or_offline_workspace_check(self):
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.assertIn(self.session.execution_session_id,
                      self.controller._pool_config_blockers(self.session.pool)["execution_session_ids"])
        self.acknowledge()
        self.assertEqual(self.controller._pool_config_blockers(self.session.pool)["execution_session_ids"], [])
        operation = self.controller.workspace_operation(self.workspace["id"], "check")
        self.assertEqual(operation["operation"]["action"], "check")

    def test_missing_state_fails_instead_of_new_empty_instance(self):
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        self.workspace["state"] = "needs_recovery"
        request = self.controller.submit_workload(spec(session_id="sequence"))
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(len(self.controller.sessions), 1)

    def test_explicit_shared_workspace_waits_and_reference_blocks_delete(self):
        self.workspace["state"] = "mounted"
        other = spec(session_id="other-sequence")
        other.workspace["id"] = self.workspace["id"]
        request = self.controller.submit_workload(other)
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.QUEUED)
        self.assertEqual(request.reason, "waiting_for_session")
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertNotEqual(request.execution_session_id, self.session.execution_session_id)
        self.assertEqual(self.controller.sessions[request.execution_session_id].workspace["id"], self.workspace["id"])
        with self.assertRaisesRegex(ValueError, "another execution"):
            self.controller.workspace_operation(self.workspace["id"], "delete")

    def test_suspended_time_does_not_expire_a_new_activation(self):
        execution_action(self.controller, self.session.execution_session_id, "suspend", None)
        self.acknowledge()
        created = self.session.created_at
        self.clock.advance(200)
        self.acknowledge()
        request = self.controller.submit_workload(spec(session_id="sequence", max_duration=100))
        self.controller.schedule_once()
        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(self.session.created_at, created)
        self.assertEqual(self.session.activated_at, self.clock())
        self.assertEqual(self.controller.expire_max_lifetime_sessions(), [])

    def test_force_before_dispatch_preserves_unstarted_identity_without_waiting_for_agent(self):
        first = self.controller.submit_workload(spec(session_id="never-started"))
        self.controller.schedule_once()
        execution = self.controller.sessions[first.execution_session_id]
        command = self.controller.request_commands[first.request_id]
        execution_action(self.controller, execution.execution_session_id, "suspend", None, force=True)
        self.assertEqual(first.state, RequestState.CANCELLED)
        self.assertEqual(execution.state, SessionState.SUSPENDED)
        self.assertTrue(execution.empty_retained)
        self.assertFalse(self.controller.command_assignments[command].result["was_dispatched"])
        second = self.controller.submit_workload(spec(session_id="never-started"))
        self.controller.schedule_once()
        self.assertEqual(second.execution_session_id, execution.execution_session_id)
        commands = self.controller.node_heartbeat("node-a")["commands"]
        self.assertNotIn(command, [c.get("command_id") for c in commands])
        resumed = next(c for c in commands if c.get("request_id") == second.request_id)
        self.assertTrue(resumed["session_spec"]["workspace"]["create"])
        self.assertFalse(resumed["session_spec"]["resume_retained"])


if __name__ == "__main__":
    unittest.main()
