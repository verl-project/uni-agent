# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit workspace doubles for hand-written unit assignments, never live gates."""

import copy
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock
import uuid

from pycape.agent import NodeAgentClient
from pycape.workspace import WorkspaceConfig


def stop_test_agent_workers(agent, stop, thread):
    """Finish report/ACK persistence before a test removes its state directory."""
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive(), "test heartbeat thread did not stop"
    agent.join(timeout=10)
    assert not any(worker.is_alive() for worker in agent._threads), "test execution thread did not stop"
    agent._stop_all_local_sessions()


def copy_workspace_evidence(root, destination):
    """Snapshot committed SQLite WAL contents, not just the main database file."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("node.log", "state.json"):
        if (root / name).is_file():
            shutil.copyfile(root / name, destination / name)
    journal = root / "journal.sqlite"
    if journal.is_file():
        with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(str(destination / "journal.sqlite"))) as snapshot:
                source.backup(snapshot)
                assert snapshot.execute("select count(*) from events").fetchone()[0] > 0, "empty journal evidence"


class AssignmentWorkspaceDouble:
    def __init__(self, workspace, path):
        self.workspace = workspace
        self.path = path
        self.revision = 0
        self.mounted = False
        self.handles = {}
        self._lock = threading.RLock()

    def _guard(self, identity):
        assert identity == self.workspace["id"]
        return self._lock

    def prepare(self, identity, user, session, create=False):
        assert identity == self.workspace["id"] and user == self.workspace["user_id"]
        Path(self.path).mkdir(parents=True, exist_ok=True)
        if not self.mounted:
            self.revision += 1
        self.mounted = True
        handle = SimpleNamespace(path=self.path, manifest={"workspace_id": identity})
        self.handles[identity] = handle
        return handle

    def verify(self, handle):
        assert self.mounted and handle.path == self.path

    def release(self, identity):
        assert identity == self.workspace["id"]
        self.mounted = False
        self.handles.clear()
        self.revision += 1

    def inspect(self, identity, user):
        assert identity == self.workspace["id"] and user == self.workspace["user_id"]
        return {"workspace_id": identity, "user_id": user, "config": self.workspace["config"],
                "state": "mounted" if self.mounted else "retained", "revision": self.revision,
                "path": self.path, "last_error": None}

    def reconcile_released(self, identity, user):
        assert not self.mounted


def install_synthetic_assignment_workspaces(test):
    """Supply the newly required protocol only to legacy hand-authored assignments.

    Controller-created assignments already carry workspace metadata and retain the
    actual filesystem implementation. This double cannot establish mount acceptance.
    """
    root = tempfile.TemporaryDirectory(prefix="cape-unit-workspace-")
    test.addCleanup(root.cleanup)
    original = NodeAgentClient.execute_assignment
    managers = {}

    def execute(agent, assignment):
        if "workspace" not in assignment.get("session_spec", {}):
            assignment = copy.deepcopy(assignment)
            session = assignment["execution_session_id"]
            identity = "ws-" + uuid.uuid5(uuid.NAMESPACE_OID, agent.node_id + ":" + session).hex
            workspace = {"id": identity, "user_id": "unit-user", "node_id": agent.node_id, "pool": agent.pool,
                         "config": WorkspaceConfig(image_root=root.name, image_size_bytes=64 * 1024 ** 2).to_dict(),
                         "target": "/cape-unit-workspace", "create": True}
            assignment["session_spec"]["workspace"] = workspace
            key = (id(agent), session)
            if key not in managers:
                managers[key] = AssignmentWorkspaceDouble(workspace, str(Path(root.name) / identity))
            agent.workspace_managers[session] = managers[key]
        return original(agent, assignment)

    patch = mock.patch.object(NodeAgentClient, "execute_assignment", execute)
    patch.start()
    test.addCleanup(patch.stop)
    agents = []
    original_init = NodeAgentClient.__init__

    def initialize(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        agents.append(agent)

    constructor = mock.patch.object(NodeAgentClient, "__init__", initialize)
    constructor.start()
    test.addCleanup(constructor.stop)

    def cleanup_agents():
        for agent in agents:
            # Unit callers may dispose their log directory before addCleanup.
            # Teardown must stop real runtime/FUSE references without recreating it.
            with mock.patch.object(agent, "_write_log"), mock.patch.object(agent, "_save_state"):
                agent._terminate_running_processes()
                agent.join(timeout=10)
                test.assertFalse(any(thread.is_alive() for thread in agent._threads))
                agent._stop_all_local_sessions()
                for manager in agent.workspace_managers.values():
                    test.assertFalse(manager.handles, "unit fixture retained a live workspace writer")

    test.addCleanup(cleanup_agents)


def install_embedded_workspace_double(test):
    """Keep controller + fake-runtime unit tests independent of host FUSE."""
    root = tempfile.TemporaryDirectory(prefix="cape-embedded-unit-workspace-")
    test.addCleanup(root.cleanup)

    class EmbeddedDouble(AssignmentWorkspaceDouble):
        def __init__(self, config, node, pool):
            self.config = config
            self.workspace = None

        def prepare(self, identity, user, session, create=False):
            if self.workspace is None:
                super().__init__({"id": identity, "user_id": user, "config": self.config.to_dict()},
                                 str(Path(root.name) / identity))
            return super().prepare(identity, user, session, create)

    patch = mock.patch("pycape.controller.WorkspaceManager", EmbeddedDouble)
    patch.start()
    test.addCleanup(patch.stop)
