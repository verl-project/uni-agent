# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import re
import shlex
import sys
import threading
import time
import uuid

import aiohttp

from .models import (
    EnforcementMode,
    MountSpec,
    NodeResources,
    PoolConfig,
    RuntimeCapabilities,
    aggregate_runtime_capabilities,
    normalize_runtime_capabilities,
)
from .observability import MetricsCollector
from .runtime import (
    ContainerSessionSpec, GPUVisibilityError, LocalProcessRuntimeAdapter,
    RuntimeResult, leased_gpu_visibility,
)
from .workspace import WorkspaceConfig, workspace_bind_conflict
from .workspace_fs import WorkspaceError, WorkspaceManager, probe_configuration


AUTHORITATIVE_SHUTDOWN_REASONS = frozenset(
    ("allocation_removed", "node_removed")
)


class AgentHTTPError(RuntimeError):
    def __init__(self, status, payload):
        self.status = int(status)
        self.payload = payload
        RuntimeError.__init__(
            self, "CAPE HTTP {}: {}".format(self.status, payload)
        )


REPORT_DELIVERY_ERRORS = (
    AgentHTTPError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
    asyncio.TimeoutError, OSError,
)
REPORT_FALLBACK_ATTEMPTS = 3


class ReportRetryExhausted(RuntimeError):
    pass


def classify_agent_shutdown(payload, node_id=None, allocation_id=None):
    if not isinstance(payload, dict):
        return None
    control = payload.get("agent_control")
    if not isinstance(control, dict):
        return None
    if control.get("authoritative") is not True:
        return None
    if control.get("action") != "shutdown":
        return None
    if control.get("reason") not in AUTHORITATIVE_SHUTDOWN_REASONS:
        return None
    if not control.get("node_id"):
        return None
    if node_id is not None and control.get("node_id") != node_id:
        return None
    if allocation_id is not None and control.get("allocation_id") != allocation_id:
        return None
    return dict(control)


def _run_sync(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, name="cape-aiohttp-client")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class NodeAgentClient(object):
    def __init__(
        self,
        controller_url,
        node_id,
        pool,
        resources,
        capabilities=None,
        runtime_adapters=None,
        runtime_capabilities=None,
        join_token=None,
        heartbeat_token=None,
        allocation_id=None,
        allocation_deadline=None,
        drain_before_sec=0,
        cleanup_buffer_sec=0,
        runtime_cache_config="default",
        cached_images=(),
        log_dir=None,
        state_path=None,
        metrics_collector=None,
        redact_env_keys=(),
        agent_epoch=None,
        exit_on_removal=True,
        pool_mode="exclusive",
        gpu_slots=1,
        placement="pack",
        workspace_config=None,
        metrics_enabled=False,
    ):
        self.controller_url = controller_url.rstrip("/")
        self.node_id = node_id
        self.pool = pool
        self.resources = resources
        pool_config = PoolConfig(
            pool_mode=pool_mode,
            gpu_slots=gpu_slots,
            placement=placement,
        )
        self.pool_mode = pool_config.pool_mode
        self.gpu_slots = pool_config.gpu_slots
        self.placement = pool_config.placement
        if capabilities is None and not runtime_capabilities:
            capabilities = local_runtime_capabilities()
        self.runtime_capabilities = normalize_runtime_capabilities(
            capabilities=capabilities,
            runtime_capabilities=runtime_capabilities,
        )
        self.capabilities = aggregate_runtime_capabilities(
            self.runtime_capabilities
        )
        self.runtime_adapters = dict(runtime_adapters or {"local-process": LocalProcessRuntimeAdapter()})
        self.join_token = join_token
        self.heartbeat_token = heartbeat_token
        self.agent_epoch = agent_epoch or uuid.uuid4().hex
        self.exit_on_removal = bool(exit_on_removal)
        self.allocation_id = allocation_id
        self.allocation_deadline = allocation_deadline
        self.drain_before_sec = drain_before_sec
        self.cleanup_buffer_sec = cleanup_buffer_sec
        self.runtime_cache_config = runtime_cache_config
        self.cached_images = tuple(cached_images)
        self.log_dir = log_dir or os.path.join(os.getcwd(), ".pycape-agent-logs")
        self.state_path = state_path or os.path.join(self.log_dir, "state-{}.json".format(node_id))
        self._state_lock = threading.RLock()
        self._heartbeat_loop = None
        self._heartbeat_wakeup = None
        self._suspension_wakeups = {}
        self.local_sessions = set()
        self.running_commands = {}
        self._pending_command_reports = set()
        self.running_processes = {}
        self.session_contexts = {}
        self.workspace_managers = {}
        self.workspace_updates = {}
        self.workspace_operation_receipts = {}
        self.workspace_acknowledged = {}
        self.workspace_bindings = {}
        self.workspace_config_status = {"state": "pending", "applied_version": None, "error": None, "protocol": 1}
        self._workspace_probe_thread = None
        self._workspace_desired = None
        self._cancelled_commands = set()
        self.redact_env_keys = {
            str(key).strip().upper()
            for key in redact_env_keys
            if str(key).strip()
        }
        self.metrics_collector = metrics_collector if metrics_collector is not None else MetricsCollector(enabled=False)
        self.metrics_link = None
        if metrics_enabled:
            from .telemetry_agent import MetricsLink
            self.metrics_link = MetricsLink(self.controller_url, self.node_id, self.agent_epoch, self.pool,
                                           self.resources.gpu_uuids, self.heartbeat_token)
        self._threads = []
        self._async_tasks = set()
        self._registered = False
        self._registration_response = None
        self._shutting_down = False
        self._pending_shutdown_control = None
        self._drain_requested = threading.Event()
        self._pending_workspace_stops = set()
        self.suspended_contexts = {}
        self.resumed_versions = {}
        self.pending_suspensions = {}
        self.purged_executions = set()
        self._workspace_operations_running = set()
        self._workspace_maintenance_sessions = set()
        self._workspace_sample_thread = None
        self._workspace_operation_threads = []
        self._closed_workspace_sessions = set()
        self._report_retries_exhausted = threading.Event()
        self._last_lifecycle_message = None
        self._load_state()
        if self._unreconciled_workspace_commands:
            raise WorkspaceError("agent state contains interrupted workspace commands; fence predecessor process groups before restart")
        if workspace_config is not None:
            self.configure_workspace(workspace_config)

    def request_drain(self):
        self._drain_requested.set()

    def _registration_payload(self):
        return {
            "node_id": self.node_id,
            "pool": self.pool,
            "pool_mode": self.pool_mode,
            "gpu_slots": self.gpu_slots,
            "placement": self.placement,
            "resources": self._resources_dict(self.resources),
            "capabilities": self._capabilities_dict(self.capabilities),
            "runtime_capabilities": {
                adapter: self._capabilities_dict(capabilities)
                for adapter, capabilities in sorted(
                    self.runtime_capabilities.items()
                )
            },
            "allocation_id": self.allocation_id,
            "allocation_deadline": self.allocation_deadline,
            "drain_before_sec": self.drain_before_sec,
            "cleanup_buffer_sec": self.cleanup_buffer_sec,
            "runtime_cache_config": self.runtime_cache_config,
            "cached_images": list(self.cached_images),
            "join_token": self.join_token,
            "agent_epoch": self.agent_epoch,
            "workspace_protocol": 1,
            "workspace_config": dict(self.workspace_config_status),
        }

    def configure_workspace(self, values):
        config = WorkspaceConfig.from_dict(values)
        self._workspace_desired = config.to_dict()
        try:
            status = probe_configuration(config, self.node_id, self.pool)
        except Exception as error:
            with self._state_lock:
                self.workspace_config_status.update(state="failed", desired_version=config.version, error=str(error))
            raise
        with self._state_lock:
            self.workspace_config_status = dict(status, desired_version=config.version)
            self._save_state()

    def _workspace_response(self, response):
        with self._state_lock:
            changed = False
            for identity, ack in response.get("workspace_acks", {}).items():
                pending = self.workspace_updates.get(identity)
                if pending and pending.get("execution_session_id") == ack.get("execution_session_id") and pending.get("revision", 0) <= ack.get("revision", -1) and pending.get("operation_result", {}).get("id") == ack.get("operation_id") and pending.get("telemetry_id") == ack.get("telemetry_id"):
                    self.workspace_updates.pop(identity, None)
                    self.workspace_acknowledged[identity] = dict(ack)
                    changed = True
            desired = response.get("workspace_config")
            if desired is not None and desired != self._workspace_desired and not self._shutting_down:
                if not self._workspace_operations_running and (self._workspace_probe_thread is None or not self._workspace_probe_thread.is_alive()):
                    self._workspace_desired = dict(desired)
                    self.workspace_config_status.update(state="applying", desired_version=desired.get("version"), error=None)
                    def apply():
                        try:
                            self.configure_workspace(desired)
                        except Exception as error:
                            self._log_lifecycle("workspace configuration failed: " + str(error))
                    self._workspace_probe_thread = threading.Thread(target=apply, name="cape-workspace-config", daemon=True)
                    self._workspace_probe_thread.start()
                    changed = True
            if changed:
                self._save_state()
        for workspace in response.get("workspace_operations", ()):
            operation = workspace["operation"]
            with self._state_lock:
                receipt = self.workspace_operation_receipts.get(workspace["id"])
                sequence = operation.get("sequence")
                if receipt is not None and type(sequence) is int and sequence <= receipt["sequence"]:
                    if sequence == receipt["sequence"] and operation["id"] == receipt["id"]:
                        self.workspace_updates.setdefault(workspace["id"], dict(receipt["update"]))
                    continue
                pending = self.workspace_updates.get(workspace["id"], {})
                busy = any(item.get("execution_session_id") == workspace.get("execution_session_id") for item in self.running_commands.values())
                probing = self._workspace_probe_thread is not None and self._workspace_probe_thread.is_alive()
                if self._shutting_down or busy or probing or self._workspace_operations_running or pending.get("operation_result", {}).get("id") == operation["id"]:
                    continue
                self._workspace_operations_running.add(operation["id"])
                self._workspace_maintenance_sessions.add(workspace["execution_session_id"])
            thread = threading.Thread(target=self._execute_workspace_operation, args=(workspace,), daemon=True)
            with self._state_lock:
                self._workspace_operation_threads.append(thread)
                try:
                    thread.start()
                except Exception:
                    self._workspace_operations_running.discard(operation["id"])
                    self._workspace_maintenance_sessions.discard(workspace["execution_session_id"])
                    self._workspace_operation_threads.remove(thread)
                    raise

    def _execute_workspace_operation(self, workspace):
        operation = workspace["operation"]
        result = dict(operation, state="completed", error=None)
        manager = None
        runtime = None
        context = {}
        boundary = operation.get("boundary", False)
        execution_session_id = workspace["execution_session_id"]
        try:
            if workspace["node_id"] != self.node_id or workspace["pool"] != self.pool:
                raise WorkspaceError("workspace maintenance belongs to another node or pool")
            with self._state_lock:
                context = self.session_contexts.get(execution_session_id, {})
                manager = self.workspace_managers.get(execution_session_id) if boundary else None
                runtime = self.runtime_adapters.get(context.get("runtime_adapter")) if boundary else None
            if boundary:
                if manager is None or runtime is None:
                    raise WorkspaceError("boundary maintenance requires the live owning agent and runtime")
                if hasattr(runtime, "suspend_workspace"):
                    runtime.suspend_workspace(execution_session_id)
                elif context.get("runtime_adapter") != "local-process":
                    raise WorkspaceError("runtime cannot safely suspend workspace references")
                try:
                    manager.release(workspace["id"])
                except Exception:
                    observed = manager.inspect(workspace["id"], workspace["user_id"])
                    if operation["action"] != "recover" or observed.get("mounted") or observed["state"] not in ("needs_check", "maintenance", "needs_recovery"):
                        raise
            else:
                manager = WorkspaceManager(WorkspaceConfig.from_dict(workspace["config"]), self.node_id, self.pool)
            def report_progress(manifest):
                manifest.update(state="maintenance", execution_session_id=execution_session_id,
                                telemetry_id=uuid.uuid4().hex,
                                operation_progress={"id": operation["id"], "phase": manifest.get("maintenance", {}).get("phase", "applying")})
                with self._state_lock:
                    self.workspace_updates[workspace["id"]] = manifest
                    self._save_state()
            manager.maintenance_observer = report_progress
            if operation["action"] == "reclaim":
                from .workspace_policy import MaintenancePolicy
                manager.maintenance_nice = MaintenancePolicy(operation.get("policy")).to_dict()["reclaim_nice"]
            if operation["action"] not in ("check", "delete", "grow", "reclaim", "shrink", "recover"):
                raise WorkspaceError("unknown workspace maintenance action")
            if operation["action"] in ("grow", "reclaim", "shrink"):
                output = manager.maintain(workspace["id"], workspace["user_id"], operation["action"],
                                         operation.get("target_bytes"), operation["id"], operation.get("policy"), operation.get("max_image_size_bytes"))
                result["reclaimed_bytes"] = output.get("reclaimed_bytes", 0)
                output = manager.recover_maintenance(workspace["id"], workspace["user_id"])
                result["details"] = output
            elif operation["action"] == "recover":
                observed = manager.inspect(workspace["id"], workspace["user_id"])
                output = (manager.recover_maintenance(workspace["id"], workspace["user_id"]) if observed.get("maintenance")
                          else manager.check(workspace["id"], workspace["user_id"]))
                result["details"] = output
            else:
                output = getattr(manager, operation["action"])(workspace["id"], workspace["user_id"])
            result["exit_code"] = output.get("exit_code", 0)
            if result["exit_code"]:
                result.update(state="failed", error="offline filesystem check failed")
            if boundary and not self._shutting_down and execution_session_id not in self._closed_workspace_sessions:
                manager.prepare(workspace["id"], workspace["user_id"], execution_session_id)
                if hasattr(runtime, "resume_workspace"):
                    runtime.resume_workspace(execution_session_id)
                with self._state_lock:
                    self.session_contexts[execution_session_id]["runtime_generation"] = context.get("runtime_generation", 0) + 1
        except Exception as error:
            result.update(state="failed", error=str(error))
            if boundary and manager is not None and runtime is not None:
                try:
                    state = manager.inspect(workspace["id"], workspace["user_id"])["state"]
                    if state == "retained":
                        manager.prepare(workspace["id"], workspace["user_id"], execution_session_id)
                        if hasattr(runtime, "resume_workspace"):
                            runtime.resume_workspace(execution_session_id)
                except Exception as recovery_error:
                    result["error"] += "; runtime recovery: " + str(recovery_error)
        try:
            if manager is not None:
                manager.maintenance_observer = None
            update = manager.inspect(workspace["id"], workspace["user_id"]) if manager else {}
        except Exception:
            update = {}
        if not update:
            update = dict(workspace, workspace_id=workspace["id"])
        update.update(execution_session_id=workspace["execution_session_id"], operation_result=result)
        result["phase"] = "completed" if result["state"] == "completed" else "failed"
        result["completed_at"] = time.time()
        if boundary and context.get("workspace_path") is not None:
            update["path"] = context["workspace_path"]
        with self._state_lock:
            self.workspace_updates[workspace["id"]] = update
            if type(operation.get("sequence")) is int:
                self.workspace_operation_receipts[workspace["id"]] = {"sequence": operation["sequence"], "id": operation["id"], "update": dict(update)}
            self._workspace_operations_running.discard(operation["id"])
            self._workspace_maintenance_sessions.discard(execution_session_id)
            self._workspace_operation_threads = [thread for thread in self._workspace_operation_threads if thread.is_alive()]
            self._save_state()

    def _workspace_pending(self):
        result, size = [], 0
        with self._state_lock:
            for identity in sorted(self.workspace_updates):
                update = self.workspace_updates[identity]
                encoded = len(json.dumps(update).encode("utf-8"))
                if result and (len(result) >= 64 or size + encoded > 512 * 1024):
                    break
                if encoded > 512 * 1024:
                    raise WorkspaceError("workspace update exceeds the bounded reporting budget")
                result.append(dict(update))
                size += encoded
        return result

    def _queue_workspace(self, workspace, execution_session_id, manager=None, error=None):
        if not isinstance(workspace, dict):
            return
        identity = workspace["id"]
        with self._state_lock:
            if self.workspace_bindings.get(identity, execution_session_id) != execution_session_id:
                return
        try:
            update = manager.inspect(identity, workspace["user_id"]) if manager is not None else None
        except Exception:
            update = None
        if update is None:
            update = {"workspace_id": identity, "user_id": workspace["user_id"], "config": workspace["config"],
                      "state": "needs_check", "revision": workspace.get("revision", 0), "path": None, "last_error": str(error or "workspace preparation did not complete")}
        update["execution_session_id"] = execution_session_id
        update["maintenance_deferred"] = self._shutting_down
        if ((self.workspace_config_status.get("config") or {}).get("maintenance") or {}).get("auto_grow"):
            update["telemetry_id"] = uuid.uuid4().hex
        if error is not None and update["state"] in ("needs_check", "creating", "mounting"):
            update["last_error"] = str(error)[-4000:]
        with self._state_lock:
            context = self.session_contexts.get(execution_session_id, {})
            if update.get("path") is not None:
                update["path"] = context.get("workspace_path", update["path"])
            previous = self.workspace_updates.get(identity)
            if previous and previous.get("operation_result") and "operation_result" not in update:
                update["operation_result"] = previous["operation_result"]
            ack = self.workspace_acknowledged.get(identity, {})
            if error is None and "space" not in update and ack.get("execution_session_id") == execution_session_id and ack.get("revision", -1) >= update["revision"]:
                return
            if previous is None or previous.get("execution_session_id") != execution_session_id or previous.get("revision", 0) <= update["revision"]:
                self.workspace_updates[identity] = update
            self._save_state()

    def _prepare_workspace(self, assignment, runtime_name, session_spec, command_spec, command_env):
        workspace = assignment.get("session_spec", {}).get("workspace")
        if not isinstance(workspace, dict):
            raise WorkspaceError("assignment lacks a managed workspace; upgrade the controller protocol")
        if workspace.get("pool") != self.pool or workspace.get("node_id") != self.node_id:
            raise WorkspaceError("workspace assignment belongs to another node or pool")
        execution_session_id, identity = session_spec.execution_session_id, workspace["id"]
        target = workspace.get("target", "/workspace")
        if runtime_name != "local-process" and any(workspace_bind_conflict(mount.target, target) for mount in session_spec.mounts):
            raise WorkspaceError("managed workspace target conflicts with an explicit bind; select --workspace-target")
        with self._state_lock:
            context = self.session_contexts.get(execution_session_id, {})
            previous = context.get("session_spec", {}).get("workspace")
            if previous and previous.get("id") != identity:
                raise WorkspaceError("session workspace identity changed")
            manager = self.workspace_managers.get(execution_session_id)
            if manager is None:
                manager = WorkspaceManager(WorkspaceConfig.from_dict(workspace["config"]), self.node_id, self.pool)
                self.workspace_managers[execution_session_id] = manager
            self.workspace_bindings[identity] = execution_session_id
            self.session_contexts[execution_session_id] = {"runtime_adapter": runtime_name, "request_id": assignment.get("request_id"),
                                                  "session_spec": self._cleanup_session_spec(assignment["session_spec"])}
            self._save_state()
        handle = manager.prepare(identity, workspace["user_id"], execution_session_id, create=workspace.get("create", False))
        with self._state_lock:
            cancelled = assignment["command_id"] in self._cancelled_commands or self._shutting_down or execution_session_id not in self.session_contexts
        if cancelled:
            manager.release(identity)
            self._queue_workspace(workspace, execution_session_id, manager)
            raise WorkspaceError("command cancelled during workspace preparation")
        visible = handle.path if runtime_name == "local-process" else target
        if runtime_name != "local-process":
            session_spec.mounts = (MountSpec(handle.path, target),) + tuple(session_spec.mounts)
        session_spec.env["CAPE_WORKSPACE_DIR"] = visible
        command_env["CAPE_WORKSPACE_DIR"] = visible
        if session_spec.cwd is None:
            session_spec.cwd = visible
        if command_spec.get("cwd") is None:
            command_spec["cwd"] = visible
        with self._state_lock:
            if execution_session_id not in self.session_contexts:
                raise WorkspaceError("session was stopped during workspace preparation")
            self.session_contexts[execution_session_id]["workspace_path"] = visible
            self._save_state()
        self._queue_workspace(workspace, execution_session_id, manager)
        return manager, handle

    def register(self):
        return _run_sync(self.register_async())

    async def register_async(self, session=None):
        if self._registered:
            return dict(self._registration_response or {})
        if self.workspace_config_status.get("state") != "applied":
            announcement = await self._post_async("/v1/agent/announce", self._registration_payload(), token=self.heartbeat_token, session=session)
            config = announcement.get("workspace_config", WorkspaceConfig().to_dict())
            async def pulse():
                while True:
                    await asyncio.sleep(1)
                    try:
                        await self._post_async("/v1/agent/heartbeat", {"node_id": self.node_id, "agent_epoch": self.agent_epoch,
                            "state": "REGISTERING", "workspace_protocol": 1, "max_commands": 0,
                            "status": {"authoritative": False, "phase": "workspace_probe"}}, token=self.heartbeat_token, session=session)
                    except REPORT_DELIVERY_ERRORS:
                        pass
            heartbeat = asyncio.create_task(pulse())
            try:
                await asyncio.get_running_loop().run_in_executor(None, self.configure_workspace, config)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
        response = await self._post_async(
            "/v1/agent/register",
            self._registration_payload(),
            token=self.heartbeat_token,
            session=session,
        )
        self._registered = True
        self._registration_response = dict(response or {})
        self._prepare_received_response(self._registration_response)
        return dict(self._registration_response)

    def heartbeat_once(self, background=True):
        if self._report_retries_exhausted.is_set():
            raise ReportRetryExhausted("report_retry_exhausted; heartbeats stopped")
        self._prune_threads()
        response = self._post(
            "/v1/agent/heartbeat",
            {
                "node_id": self.node_id,
                "agent_epoch": self.agent_epoch,
                "status": self.collect_status(),
                "max_commands": 8,
            },
            token=self.heartbeat_token,
        )
        control = self._shutdown_control(response)
        self._prepare_received_response(response)
        if control is not None and self.exit_on_removal:
            return response
        self._dispatch_response_commands(
            response,
            background=background,
        )
        if self._report_retries_exhausted.is_set():
            _run_sync(self.shutdown_async(
                {"reason": "report_retry_exhausted"}, authoritative=False
            ))
            raise ReportRetryExhausted("report_retry_exhausted; agent stopped")
        self._prune_threads()
        return response

    async def heartbeat_once_async(self, background=True, session=None):
        if self._report_retries_exhausted.is_set():
            raise ReportRetryExhausted("report_retry_exhausted; heartbeats stopped")
        loop = asyncio.get_running_loop()
        if self._heartbeat_wakeup is not None:
            self._heartbeat_wakeup.clear()
        status = await loop.run_in_executor(None, self.collect_status)
        response = await self._post_async(
            "/v1/agent/heartbeat",
            {
                "node_id": self.node_id,
                "agent_epoch": self.agent_epoch,
                "status": status,
                "max_commands": 8,
            },
            token=self.heartbeat_token,
            session=session,
        )
        control = self._shutdown_control(response)
        self._prepare_received_response(response)
        if control is not None and self.exit_on_removal:
            return response
        await self._dispatch_response_commands_async(
            response,
            background=background,
            loop=loop,
        )
        return response

    def _prune_threads(self):
        with self._state_lock:
            self._threads = [
                thread for thread in self._threads if thread.is_alive()
            ]

    def run_forever(self, interval=1.0):
        return _run_sync(self.run_forever_async(interval=interval))

    def register_until_available(self, interval=1.0):
        return _run_sync(
            self.register_until_available_async(interval=interval)
        )

    async def register_until_available_async(self, interval=1.0):
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await self._register_with_retry_async(session, interval)

    async def _register_with_retry_async(self, session, interval):
        while True:
            try:
                return await self.register_async(session=session)
            except AgentHTTPError as exc:
                self._log_lifecycle(
                    "registration rejected with HTTP {}; retrying: {}".format(
                        exc.status, exc.payload
                    )
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
            ) as exc:
                self._log_lifecycle(
                    "registration transport failure; retrying: {}".format(
                        exc
                    )
                )
            await asyncio.sleep(max(0.0, interval))

    async def run_forever_async(self, interval=1.0):
        self._heartbeat_loop = asyncio.get_running_loop()
        self._heartbeat_wakeup = asyncio.Event()
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            registration = await self._register_with_retry_async(
                session, interval
            )
            control = self._shutdown_control(registration)
            if control is not None and self.exit_on_removal:
                await self.shutdown_async(control)
                return control
            drain_announced = False
            while True:
                if self._report_retries_exhausted.is_set():
                    await self.shutdown_async(
                        {"reason": "report_retry_exhausted"}, authoritative=False
                    )
                    raise RuntimeError("report_retry_exhausted; agent stopped")
                if self._drain_requested.is_set() and not drain_announced:
                    try:
                        await self._post_async(
                            "/v1/admin/drain-node",
                            {
                                "node_id": self.node_id,
                                "reason": "operator_drain",
                            },
                            token=self.heartbeat_token,
                            session=session,
                        )
                    except (
                        AgentHTTPError,
                        aiohttp.ClientError,
                        asyncio.TimeoutError,
                        OSError,
                    ) as exc:
                        self._log_lifecycle(
                            "SIGTERM drain notification failed; retrying: {}".format(
                                exc
                            )
                        )
                    else:
                        drain_announced = True
                        self._log_lifecycle(
                            "SIGTERM drain acknowledged; waiting for active work and cleanup"
                        )
                control = self._take_pending_shutdown_control()
                if control is not None:
                    if self.exit_on_removal:
                        await self.shutdown_async(control)
                        return control
                    self._log_lifecycle(
                        "authoritative removal received but "
                        "--stay-after-removal is active: {}".format(
                            control.get("reason")
                        )
                    )
                try:
                    response = await self.heartbeat_once_async(
                        background=True, session=session
                    )
                except ReportRetryExhausted:
                    await self.shutdown_async(
                        {"reason": "report_retry_exhausted"}, authoritative=False
                    )
                    raise
                except AgentHTTPError as exc:
                    self._log_lifecycle(
                        "heartbeat rejected with HTTP {}; retrying: {}".format(
                            exc.status, exc.payload
                        )
                    )
                    if exc.status == 404:
                        self._registered = False
                        self._registration_response = None
                        try:
                            registration = await self.register_async(
                                session=session
                            )
                        except (
                            AgentHTTPError,
                            aiohttp.ClientError,
                            asyncio.TimeoutError,
                            OSError,
                        ) as register_exc:
                            self._log_lifecycle(
                                "re-registration deferred; retrying: {}".format(
                                    register_exc
                                )
                            )
                        else:
                            control = self._shutdown_control(registration)
                            if (
                                control is not None
                                and self.exit_on_removal
                            ):
                                await self.shutdown_async(control)
                                return control
                    await asyncio.sleep(max(0.0, interval))
                    continue
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    OSError,
                ) as exc:
                    self._log_lifecycle(
                        "heartbeat transport failure; retrying: {}".format(
                            exc
                        )
                    )
                    await asyncio.sleep(max(0.0, interval))
                    continue
                control = self._shutdown_control(response)
                if control is not None:
                    if self.exit_on_removal:
                        await self.shutdown_async(control)
                        return control
                    self._log_lifecycle(
                        "authoritative removal received but "
                        "--stay-after-removal is active: {}".format(
                            control.get("reason")
                        )
                    )
                self._prune_threads()
                if drain_announced:
                    with self._state_lock:
                        locally_empty = (
                            not self._threads
                            and not self._async_tasks
                            and not self.running_commands
                            and not self.local_sessions
                        )
                    if locally_empty:
                        await self._flush_workspace_updates_async()
                        self._log_lifecycle(
                            "SIGTERM drain complete; node enroll exits 0"
                        )
                        return {
                            "action": "drain",
                            "reason": "operator_drain",
                            "node_id": self.node_id,
                        }
                try:
                    await asyncio.wait_for(self._heartbeat_wakeup.wait(), timeout=max(0.0, interval))
                except asyncio.TimeoutError:
                    pass

    def _notify_retention_change(self):
        if self._heartbeat_loop is not None and self._heartbeat_wakeup is not None:
            try:
                self._heartbeat_loop.call_soon_threadsafe(self._heartbeat_wakeup.set)
            except RuntimeError:
                pass  # The heartbeat loop has already shut down.

    def _shutdown_control(self, payload):
        return classify_agent_shutdown(
            payload,
            node_id=self.node_id,
            allocation_id=self.allocation_id,
        )

    def _reserve_response_commands(self, payload):
        if self._shutdown_control(payload) is not None:
            return []
        with self._state_lock:
            if self._shutting_down or self._report_retries_exhausted.is_set():
                return []
            reserved = []
            for command in payload.get("commands", []):
                if command.get("action", "run") != "run":
                    continue
                command_id, execution_session_id = command.get("command_id"), command.get("execution_session_id")
                if (command_id and execution_session_id and (execution_session_id not in self._closed_workspace_sessions
                        or self._retained_resume_context(command) is not None)
                        and command_id not in self.running_commands):
                    self.running_commands[command_id] = command
                    reserved.append(command_id)
            if reserved:
                try:
                    self._save_state()
                except BaseException:
                    for command_id in reserved:
                        self.running_commands.pop(command_id, None)
                    raise
            return reserved

    def _retained_resume_context(self, assignment):
        context = getattr(self, "suspended_contexts", {}).get(assignment.get("execution_session_id"))
        version = assignment.get("session_spec", {}).get("suspend_version", 0)
        if (context is not None and assignment.get("session_spec", {}).get("resume_retained") is True
                and type(version) is int and version >= 0 and context.get("suspend_version") == version):
            return context
        return None

    def _prepare_received_response(self, payload):
        reserved = self._reserve_response_commands(payload)
        try:
            self._workspace_response(payload)
        except BaseException:
            with self._state_lock:
                for command_id in reserved:
                    self.running_commands.pop(command_id, None)
            raise

    def _response_command_invocations(self, payload):
        self._reserve_response_commands(payload)
        control = self._shutdown_control(payload)
        invocations = []
        for command in payload.get("commands", []):
            action = command.get("action", "run")
            if (
                control is not None
                and action not in ("cancel", "stop_session")
            ):
                continue
            if action == "cancel":
                invocations.append(
                    ("cleanup", self.cancel_assignment, (command,))
                )
            elif action == "purge_session":
                invocations.append(("cleanup", self.purge_session, (command,)))
            elif action in ("stop_session", "suspend_session"):
                invocations.append(
                    (
                        "cleanup",
                        self.stop_session,
                        (
                            command["execution_session_id"],
                            command.get("runtime_adapter"),
                            command.get("session_spec"),
                            command.get("suspend_version"),
                        ) + ((True,) if action == "suspend_session" else ()),
                    )
                )
            else:
                invocations.append(
                    ("run", self.execute_assignment, (command,))
                )
        return invocations

    def _dispatch_response_commands(self, payload, background=True):
        for kind, function, args in self._response_command_invocations(
            payload
        ):
            if kind == "run" and self._report_retries_exhausted.is_set():
                continue
            if kind == "run" and background:
                thread = threading.Thread(target=function, args=args)
                thread.daemon = True
                with self._state_lock:
                    thread.start()
                    self._threads.append(thread)
            else:
                function(*args)

    async def _dispatch_response_commands_async(
        self,
        payload,
        background=True,
        loop=None,
    ):
        loop = loop or asyncio.get_running_loop()
        for kind, function, args in self._response_command_invocations(
            payload
        ):
            if kind == "run" and self._report_retries_exhausted.is_set():
                continue
            future = loop.run_in_executor(None, function, *args)
            if kind == "run" and background:
                self._async_tasks.add(future)
                future.add_done_callback(self._async_tasks.discard)
            else:
                await future

    def _record_shutdown_control(self, payload):
        control = self._shutdown_control(payload)
        if control is not None:
            with self._state_lock:
                self._pending_shutdown_control = dict(control)
        return control

    def _take_pending_shutdown_control(self):
        with self._state_lock:
            control = self._pending_shutdown_control
            self._pending_shutdown_control = None
        return None if control is None else dict(control)

    def _log_lifecycle(self, message):
        message = str(message)
        if message == self._last_lifecycle_message:
            return
        self._last_lifecycle_message = message
        sys.stderr.write("[cape-node {}] {}\n".format(self.node_id, message))
        sys.stderr.flush()

    def _terminate_running_processes(self):
        errors = []
        with self._state_lock:
            handles = list(self.running_processes.items())
        for command_id, handle in handles:
            if not hasattr(handle, "terminate_tree"):
                continue
            try:
                handle.terminate_tree()
            except Exception as exc:
                errors.append(
                    "failed to terminate {}: {}".format(command_id, exc)
                )
        return errors

    def _stop_all_local_sessions(self):
        with self._state_lock:
            execution_session_ids = set(self.local_sessions)
            execution_session_ids.update(self.session_contexts)
            execution_session_ids.update(
                assignment.get("execution_session_id")
                for assignment in self.running_commands.values()
                if assignment.get("execution_session_id")
            )
        for execution_session_id in sorted(execution_session_ids):
            self.stop_session(execution_session_id)

    def _flush_workspace_updates(self):
        return _run_sync(self._flush_workspace_updates_async())

    async def _flush_workspace_updates_async(self):
        self._shutting_down = True
        if self._report_retries_exhausted.is_set() or not self._registered:
            return True
        if not self._workspace_pending():
            return True
        status = await asyncio.get_running_loop().run_in_executor(None, self.collect_status)
        try:
            response = await self._post_async("/v1/agent/heartbeat", {
                "node_id": self.node_id, "agent_epoch": self.agent_epoch,
                "status": status, "max_commands": 0,
            }, token=self.heartbeat_token)
        except REPORT_DELIVERY_ERRORS as error:
            self._log_lifecycle("workspace final status remains persisted for retry: " + str(error))
            return False
        self._prepare_received_response(response)
        with self._state_lock:
            clean = not self.local_sessions and not self.running_commands
        return not self._workspace_pending() and clean

    async def shutdown_async(self, control, authoritative=True):
        if self.metrics_link is not None:
            await asyncio.get_running_loop().run_in_executor(None, self.metrics_link.close)
        self._shutting_down = True
        reason = control.get("reason", "node_removed")
        allocation_id = control.get("allocation_id") or self.allocation_id
        self._log_lifecycle(
            "{}: reason={} allocation={}; "
            "cleaning local runtime state".format(
                "authoritative removal received" if authoritative else "local shutdown requested",
                reason, allocation_id or "unknown",
            )
        )
        loop = asyncio.get_running_loop()
        errors = await loop.run_in_executor(
            None, self._terminate_running_processes
        )
        pending = [future for future in list(self._async_tasks) if not future.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await loop.run_in_executor(None, self.join, 30.0)
        for thread in list(self._workspace_operation_threads):
            await loop.run_in_executor(None, thread.join)
        if self._workspace_probe_thread is not None:
            await loop.run_in_executor(None, self._workspace_probe_thread.join)
        with self._state_lock:
            if any(thread.is_alive() for thread in self._threads):
                raise WorkspaceError("workers remain active; workspace cleanup is deferred")
            self.running_processes.clear()
            self.running_commands.clear()
        await loop.run_in_executor(None, self._stop_all_local_sessions)
        with self._state_lock:
            self.running_processes.clear()
            self.running_commands.clear()
            self._save_state()
        await self._flush_workspace_updates_async()
        for error in errors:
            self._log_lifecycle(error)
        self._log_lifecycle(
            "local cleanup complete; node enroll exits {} after {}".format(
                0 if authoritative else 1, reason
            )
        )
        return dict(control)

    def join(self, timeout=None):
        with self._state_lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)

    def collect_status(self):
        with self._state_lock:
            managers = list(self.workspace_managers.items())
            maintaining = set(self._workspace_maintenance_sessions)
        for execution_session_id, manager in managers:
            if execution_session_id in maintaining:
                continue
            for handle in list(manager.handles.values()):
                guard = manager._guard(handle.manifest["workspace_id"])
                if not guard.acquire(blocking=False):
                    continue
                try:
                    manager.verify(handle)
                except Exception as error:
                    self._log_lifecycle("workspace health failure for {}: {}".format(execution_session_id, error))
                    self.stop_session(execution_session_id)
                finally:
                    guard.release()
        self._start_workspace_sampler()
        with self._state_lock:
            pending_stops = list(self._pending_workspace_stops)
        for execution_session_id in pending_stops:
            self.stop_session(execution_session_id)
        with self._state_lock:
            local_sessions = sorted(self.local_sessions)
            running_commands = list(self.running_commands.items())
            command_ids = sorted(set(self.running_commands) | self._pending_command_reports)
        sessions = []
        for execution_session_id in local_sessions:
            sessions.append({"execution_session_id": execution_session_id, "state": "CACHED_IDLE"})
        for command_id, assignment in running_commands:
            sessions.append({
                "execution_session_id": assignment.get("execution_session_id"),
                "state": "RUNNING_COMMAND",
                "active_request_id": assignment.get("request_id"),
            })
        status = {"authoritative": True, "sessions": sessions, "command_ids": command_ids,
                  "workspace_config": dict(self.workspace_config_status),
                  "workspace_updates": self._workspace_pending()}
        with self._state_lock:
            status["reclaim_ready"] = (not self._shutting_down and not command_ids and not self._workspace_operations_running
                                       and not (self._workspace_probe_thread and self._workspace_probe_thread.is_alive()))
        status["suspended_sessions"] = {key: value["suspend_version"]
                                        for key, value in getattr(self, "suspended_contexts", {}).items()}
        status["purged_executions"] = sorted(getattr(self, "purged_executions", ()))
        gpu_metrics = self.metrics_collector.collect_gpu_metrics() if self.metrics_collector is not None else []
        if gpu_metrics:
            status["gpu_metrics"] = gpu_metrics
        if self.metrics_link is not None:
            metrics = self.metrics_link.status()
            status["metrics"] = metrics
        return status

    def _start_workspace_sampler(self):
        policy = (self.workspace_config_status.get("config") or {}).get("maintenance") or {}
        with self._state_lock:
            if not policy.get("auto_grow") or self._shutting_down or (self._workspace_sample_thread and self._workspace_sample_thread.is_alive()):
                return
            def sample():
                with self._state_lock:
                    snapshots = [(session, manager, self.session_contexts.get(session, {})) for session, manager in self.workspace_managers.items()
                                 if session not in self._workspace_maintenance_sessions]
                for session, manager, context in snapshots:
                    workspace = context.get("session_spec", {}).get("workspace")
                    if not workspace:
                        continue
                    guard = manager._guard(workspace["id"])
                    if not guard.acquire(blocking=False):
                        continue
                    try:
                        self._queue_workspace(workspace, session, manager)
                    except Exception as error:
                        self._log_lifecycle("workspace sampling deferred: " + str(error))
                    finally:
                        guard.release()
            self._workspace_sample_thread = threading.Thread(target=sample, daemon=True)
            self._workspace_sample_thread.start()

    def execute_assignment(self, assignment):
        command_id = assignment["command_id"]
        stdout = ""
        stderr = ""
        exit_code = 127
        log_path = None
        executions = []
        failure_kind = "workload"
        runtime_name = assignment.get("session_spec", {}).get(
            "runtime_adapter"
        )
        session_spec = None
        workspace_manager = None
        workspace_handle = None
        try:
            log_path = self._log_path(command_id)
            with self._state_lock:
                self.running_commands[command_id] = assignment
                self._save_state()
            session_spec = self._session_spec_from_dict(assignment["session_spec"])
            command_spec = dict(assignment["command_spec"])
            command_env = dict(session_spec.env)
            command_env.update(command_spec.get("env", {}))
            command_env["CUDA_VISIBLE_DEVICES"] = leased_gpu_visibility(
                session_spec.gpu_uuids, command_env
            )
            session_spec.env["CUDA_VISIBLE_DEVICES"] = command_env["CUDA_VISIBLE_DEVICES"]
            runtime_name = assignment["session_spec"].get("runtime_adapter")
            if not runtime_name and len(self.runtime_capabilities) == 1:
                runtime_name = next(iter(self.runtime_capabilities))
            if not runtime_name:
                raise RuntimeError(
                    "multi-runtime assignment is missing session_spec.runtime_adapter"
                )
            runtime = self.runtime_adapters.get(runtime_name)
            if runtime is None:
                raise RuntimeError("node agent has no runtime adapter: " + runtime_name)
            with self._state_lock:
                retained = self._retained_resume_context(assignment)
                previous_context = retained or self.session_contexts.get(session_spec.execution_session_id)
                if previous_context is not None:
                    if previous_context["runtime_adapter"] != runtime_name:
                        raise WorkspaceError("retained execution runtime does not match")
                    for field in ("overlay_path", "overlay_upper", "overlay_work", "state_layer_id"):
                        setattr(session_spec, field, previous_context["session_spec"].get(field))
                if retained is not None:
                    self._closed_workspace_sessions.discard(session_spec.execution_session_id)
                if command_id in self._cancelled_commands or self._shutting_down or assignment.get("execution_session_id") in self._closed_workspace_sessions:
                    raise WorkspaceError("command cancelled before workspace preparation")
            workspace_manager, workspace_handle = self._prepare_workspace(
                assignment, runtime_name, session_spec, command_spec, command_env,
            )
            with self._state_lock:
                if command_id in self._cancelled_commands or self._shutting_down:
                    raise WorkspaceError("command cancelled before runtime preparation")
            image_ref = session_spec.image
            if hasattr(runtime, "prepare_image"):
                prepared = runtime.prepare_image(image_ref, self.runtime_cache_config)
                if isinstance(prepared, RuntimeResult):
                    executions.extend(prepared.executions)
                    if prepared.exit_code != 0:
                        result = prepared
                        exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
                        failure_kind = "infrastructure"
                        return
                elif prepared:
                    session_spec.image = prepared
            if previous_context is not None and hasattr(runtime, "retention_fingerprint"):
                if previous_context.get("retention_fingerprint") != runtime.retention_fingerprint(session_spec):
                    raise WorkspaceError("retained execution environment changed; explicit rebuild required")
            runtime_session_known = not hasattr(runtime, "sessions") or session_spec.execution_session_id in getattr(runtime, "sessions", {})
            with self._state_lock:
                local_session_known = session_spec.execution_session_id in self.local_sessions
            if (not local_session_known or not runtime_session_known) and hasattr(runtime, "start_session"):
                if previous_context is not None:
                    if previous_context["session_spec"].get("image") != session_spec.image:
                        raise WorkspaceError("retained base image changed; explicit rebuild required")
                    start_result = runtime.resume_session(session_spec)
                else:
                    if assignment["session_spec"].get("resume_retained") and not local_session_known:
                        raise WorkspaceError("retained execution metadata is missing; explicit rebuild required")
                    start_result = runtime.start_session(session_spec)
                if isinstance(start_result, RuntimeResult):
                    executions.extend(start_result.executions)
                    if start_result.exit_code != 0:
                        exit_code, stdout, stderr = start_result.exit_code, start_result.stdout, start_result.stderr
                        failure_kind = "infrastructure"
                        return
                with self._state_lock:
                    if retained is not None:
                        self.resumed_versions[session_spec.execution_session_id] = retained["suspend_version"]
                        self.suspended_contexts.pop(session_spec.execution_session_id, None)
                    self.local_sessions.add(session_spec.execution_session_id)
                    self.session_contexts[session_spec.execution_session_id] = (
                        self._persisted_session_context(
                            assignment, runtime_name, session_spec
                        )
                    )
                    self._save_state()
            else:
                if previous_context is not None and hasattr(runtime, "restore_session"):
                    if previous_context["session_spec"].get("image") != session_spec.image:
                        raise WorkspaceError("cached base image changed; explicit rebuild required")
                    if assignment["session_spec"].get("resume_retained"):
                        checked = runtime.resume_session(session_spec)
                        if checked.exit_code:
                            raise WorkspaceError(checked.stderr)
                    else:
                        runtime.restore_session(session_spec)
                with self._state_lock:
                    self.session_contexts[session_spec.execution_session_id] = (
                        self._persisted_session_context(
                            assignment, runtime_name, session_spec
                        )
                    )
            command_tuple = tuple(command_spec.get("command", ()))
            with self._state_lock:
                if command_id in self._cancelled_commands or self._shutting_down:
                    raise WorkspaceError("command cancelled before runtime launch")
                if retained is not None:
                    self.resumed_versions[session_spec.execution_session_id] = retained["suspend_version"]
                    self.suspended_contexts.pop(session_spec.execution_session_id, None)
                self.local_sessions.add(session_spec.execution_session_id)
                self._save_state()
            if workspace_manager is not None:
                workspace_manager.verify(workspace_handle)
            if hasattr(runtime, "start_command_process"):
                handle = runtime.start_command_process(
                    session_spec.execution_session_id,
                    command_tuple,
                    cwd=command_spec.get("cwd"),
                    env=command_env,
                    command_id=command_id,
                )
                with self._state_lock:
                    self.running_processes[command_id] = handle
                    cancelled = command_id in self._cancelled_commands or self._shutting_down
                if cancelled:
                    handle.terminate_tree()
                result = handle.wait(reap_process_group=True)
            elif hasattr(runtime, "start_process"):
                handle = runtime.start_process(command_tuple, cwd=command_spec.get("cwd"), env=command_env)
                with self._state_lock:
                    self.running_processes[command_id] = handle
                    cancelled = command_id in self._cancelled_commands or self._shutting_down
                if cancelled:
                    handle.terminate_tree()
                result = handle.wait(reap_process_group=True)
            else:
                result = runtime.run_command(
                    session_spec.execution_session_id,
                    command_tuple,
                    cwd=command_spec.get("cwd"),
                    env=command_env,
                )
            exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
            failure_kind = result.failure_kind or "workload"
            executions.extend(result.executions)
            if workspace_manager is not None:
                workspace_manager.verify(workspace_handle)
        except GPUVisibilityError as exc:
            exit_code, stdout, stderr = 127, stdout, str(exc)
            failure_kind = "workload"
        except Exception as exc:
            exit_code, stdout, stderr = 127, stdout, str(exc)
            failure_kind = "infrastructure"
        finally:
            try:
                workspace = assignment.get("session_spec", {}).get("workspace")
                manager = workspace_manager or self.workspace_managers.get(assignment.get("execution_session_id"))
                self._queue_workspace(workspace, assignment.get("execution_session_id"), manager,
                                      error=stderr if failure_kind == "infrastructure" else None)
            except Exception as error:
                exit_code, failure_kind = 127, "infrastructure"
                stderr += "\nfailed to persist workspace status: " + str(error)
            try:
                with self._state_lock:
                    self._pending_command_reports.add(command_id)
                    self.running_processes.pop(command_id, None)
                    self.running_commands.pop(command_id, None)
                    self._cancelled_commands.discard(command_id)
                    self._save_state()
            except Exception as exc:
                exit_code = 127
                failure_kind = "infrastructure"
                if stderr and not stderr.endswith("\n"):
                    stderr += "\n"
                stderr += "failed to persist node-agent state: " + str(exc)
            if log_path is not None:
                try:
                    self._write_log(
                        log_path,
                        stdout,
                        stderr,
                        header=self._command_log_header(
                            assignment,
                            runtime_name,
                            session_spec,
                            executions,
                            exit_code,
                        ),
                    )
                except Exception as exc:
                    exit_code = 127
                    failure_kind = "infrastructure"
                    if stderr and not stderr.endswith("\n"):
                        stderr += "\n"
                    stderr += "failed to write node-agent log: " + str(exc)
            try:
                self._report(
                    command_id, exit_code, stdout, stderr, log_path, failure_kind,
                )
            finally:
                with self._state_lock:
                    self._pending_command_reports.discard(command_id)


    def cancel_assignment(self, assignment):
        command_id = assignment["command_id"]
        with self._state_lock:
            self._cancelled_commands.add(command_id)
            handle = self.running_processes.get(command_id)
        if handle is not None and hasattr(handle, "terminate_tree"):
            handle.terminate_tree()
        elif assignment.get("execution_session_id"):
            self.stop_session(assignment["execution_session_id"])

    def stop_session(self, execution_session_id, runtime_name=None, session_spec=None, suspend_version=None, notify=False):
        with self._state_lock:
            if suspend_version is not None:
                if type(suspend_version) is not int or suspend_version < 1:
                    raise WorkspaceError("invalid suspension version")
                if execution_session_id in self.resumed_versions and suspend_version <= self.resumed_versions[execution_session_id]:
                    return
                if self.suspended_contexts.get(execution_session_id, {}).get("suspend_version", -1) >= suspend_version:
                    return
                if notify:
                    self._suspension_wakeups[execution_session_id] = suspend_version
                if execution_session_id in self.suspended_contexts:
                    self.suspended_contexts[execution_session_id]["suspend_version"] = suspend_version
                    self._save_state()
                    if self._suspension_wakeups.get(execution_session_id) == suspend_version:
                        self._suspension_wakeups.pop(execution_session_id)
                        self._notify_retention_change()
                    return
                self.pending_suspensions[execution_session_id] = suspend_version
            suspend_version = getattr(self, "pending_suspensions", {}).get(execution_session_id)
            if execution_session_id in self._closed_workspace_sessions and execution_session_id not in self.session_contexts and execution_session_id not in self.local_sessions:
                return
            self._closed_workspace_sessions.add(execution_session_id)
            self._pending_workspace_stops.add(execution_session_id)
            if execution_session_id in self._workspace_maintenance_sessions:
                return
            running = [identity for identity, assignment in self.running_commands.items()
                       if assignment.get("execution_session_id") == execution_session_id]
            self._cancelled_commands.update(running)
            handles = [self.running_processes.get(identity) for identity in running]
            context = dict(self.session_contexts.get(execution_session_id, {}))
        if running:
            for handle in handles:
                if handle is not None and hasattr(handle, "terminate_tree"):
                    handle.terminate_tree()
            return
        runtime_name = runtime_name or context.get("runtime_adapter")
        cleanup_spec = session_spec or context.get("session_spec")
        if cleanup_spec:
            cleanup_spec = self._cleanup_session_spec(cleanup_spec)
            context["session_spec"] = cleanup_spec
            context["isolation_policy"] = cleanup_spec.get(
                "isolation_policy", context.get("isolation_policy")
            )
        if runtime_name:
            context["runtime_adapter"] = runtime_name
        if runtime_name in self.runtime_adapters:
            runtimes = ((runtime_name, self.runtime_adapters[runtime_name]),)
        else:
            runtimes = tuple(self.runtime_adapters.items())
        executions = []
        errors = []
        restored_spec = (
            self._session_spec_from_dict(cleanup_spec)
            if cleanup_spec
            else None
        )
        for candidate_name, runtime in runtimes:
            if hasattr(runtime, "stop_session"):
                try:
                    runtime_sessions = getattr(runtime, "sessions", None)
                    runtime_session_known = (
                        runtime_sessions is None
                        or execution_session_id in runtime_sessions
                    )
                    if not runtime_session_known and restored_spec is not None:
                        if hasattr(runtime, "restore_session"):
                            runtime.restore_session(restored_spec)
                        else:
                            runtime_sessions[execution_session_id] = restored_spec
                    result = (runtime.suspend_session(execution_session_id) if suspend_version is not None
                              else runtime.stop_session(execution_session_id))
                    if isinstance(result, RuntimeResult):
                        executions.extend(result.executions)
                        if result.exit_code != 0:
                            errors.append(result.stderr or "runtime stop failed")
                    if not runtime_name:
                        runtime_name = candidate_name
                except Exception as exc:
                    errors.append(str(exc))
        manager = self.workspace_managers.get(execution_session_id)
        workspace = (cleanup_spec or {}).get("workspace")
        if workspace is not None and manager is None and not errors:
            try:
                manager = WorkspaceManager(WorkspaceConfig.from_dict(workspace["config"]), self.node_id, self.pool)
                checked = manager.check(workspace["id"], workspace["user_id"])
                self._queue_workspace(workspace, execution_session_id, manager)
                if checked["exit_code"]:
                    raise WorkspaceError("recovered workspace requires offline maintenance")
            except Exception as error:
                errors.append("workspace ownership requires reconciliation after agent restart: " + str(error))
        if manager is not None and workspace is not None and not errors:
            try:
                if workspace["id"] in manager.handles:
                    manager.release(workspace["id"])
                else:
                    manager.reconcile_released(workspace["id"], workspace["user_id"])
                self._queue_workspace(workspace, execution_session_id, manager)
            except Exception as error:
                errors.append(str(error))
                self._queue_workspace(workspace, execution_session_id, manager, error=error)
        context["cleanup_status"] = "failed" if errors else "ok"
        try:
            self._write_log(
                self._session_log_path(execution_session_id),
                "",
                "\n".join(errors),
                header=self._session_log_header(
                    execution_session_id,
                    runtime_name,
                    context,
                    executions,
                ),
            )
        except Exception:
            pass
        with self._state_lock:
            if errors:
                self.session_contexts[execution_session_id] = context
            else:
                if cleanup_spec is not None:
                    context["suspend_version"] = suspend_version or 0
                    self.suspended_contexts[execution_session_id] = context
                self.pending_suspensions.pop(execution_session_id, None)
                self.local_sessions.discard(execution_session_id)
                self.session_contexts.pop(execution_session_id, None)
                self.workspace_managers.pop(execution_session_id, None)
                self._pending_workspace_stops.discard(execution_session_id)
            self._save_state()
            if not errors and cleanup_spec is not None and suspend_version is not None and self._suspension_wakeups.get(execution_session_id) == suspend_version:
                self._suspension_wakeups.pop(execution_session_id)
                self._notify_retention_change()

    def _persisted_session_context(
        self, assignment, runtime_name, session_spec
    ):
        cleanup = self._cleanup_session_spec(assignment.get("session_spec", {}))
        cleanup["runtime_adapter"] = runtime_name
        cleanup["mounts"] = [{"source": mount.source, "target": mount.target, "read_only": mount.read_only} for mount in session_spec.mounts]
        cleanup["cwd"] = session_spec.cwd
        for field in ("overlay_path", "overlay_upper", "overlay_work", "state_layer_id", "image"):
            cleanup[field] = getattr(session_spec, field)
        return {
            "runtime_adapter": runtime_name,
            "isolation_policy": session_spec.isolation_policy,
            "request_id": assignment.get("request_id"),
            "session_spec": cleanup,
            "workspace_path": session_spec.env.get("CAPE_WORKSPACE_DIR"),
            "retention_fingerprint": self.runtime_adapters[runtime_name].retention_fingerprint(session_spec)
                if hasattr(self.runtime_adapters[runtime_name], "retention_fingerprint") else None,
        }

    def purge_session(self, command):
        identity = command["execution_session_id"]
        with self._state_lock:
            if identity in self.purged_executions:
                return
            if identity in self.local_sessions or any(a.get("execution_session_id") == identity for a in self.running_commands.values()):
                raise WorkspaceError("cannot purge a busy execution")
            context = self.suspended_contexts.get(identity)
            if context is None or context["suspend_version"] != command.get("suspend_version"):
                raise WorkspaceError("purge requires matching retained execution metadata")
            runtime = self.runtime_adapters[context["runtime_adapter"]]
            spec = self._session_spec_from_dict(context["session_spec"])
            result = None
            if context["runtime_adapter"] != "local-process":
                runtime.restore_session(spec)
                result = runtime.purge_session(identity)
            if isinstance(result, RuntimeResult) and result.exit_code:
                raise WorkspaceError(result.stderr or "runtime purge failed")
            if any(path and os.path.exists(path) for path in (spec.overlay_path, spec.overlay_upper, spec.overlay_work)):
                raise WorkspaceError("runtime purge left state paths; retry required")
            self.suspended_contexts.pop(identity)
            self.purged_executions.add(identity)
            self._save_state()
            self._notify_retention_change()

    @staticmethod
    def _cleanup_session_spec(data):
        allowed = (
            "execution_session_id",
            "runtime_adapter",
            "image",
            "mounts",
            "cwd",
            "privileged",
            "cache_dir",
            "tmp_dir",
            "runtime_dir",
            "oci_bundle",
            "isolation_policy",
            "state_layer_id",
            "overlay_path",
            "overlay_upper",
            "overlay_work",
            "workspace",
        )
        return {key: data.get(key) for key in allowed if key in data}

    def _report(
        self,
        command_id,
        exit_code,
        stdout,
        stderr,
        log_path,
        failure_kind="workload",
    ):
        payload = {
            "node_id": self.node_id,
            "agent_epoch": self.agent_epoch,
            "command_id": command_id,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "failure_kind": failure_kind,
            "log_path": log_path,
            "accept_commands": True,
            "max_commands": 8,
            "workspace_updates": self._workspace_pending(),
        }
        try:
            response = self._post(
                "/v1/agent/report-command", payload, token=self.heartbeat_token
            )
        except REPORT_DELIVERY_ERRORS as exc:
            if isinstance(exc, AgentHTTPError) and not (500 <= exc.status < 600 or exc.status == 413):
                self._record_report_abandonment(command_id, exc, log_path, "permanent_http")
                return {}
            # The node log already contains the complete command result. Never
            # replace it with a partial successful response to fit a rejection.
            payload.update(
                exit_code=127,
                stdout="",
                stderr=json.dumps({
                    "error": "command_report_delivery_failed",
                    "original_exit_code": exit_code,
                    "report_error": type(exc).__name__,
                    "log_reference": "node-local; not remotely retrievable",
                }, sort_keys=True),
                failure_kind="infrastructure",
            )
            for attempt in range(REPORT_FALLBACK_ATTEMPTS):
                try:
                    response = self._post(
                        "/v1/agent/report-command", payload,
                        token=self.heartbeat_token,
                    )
                    break
                except REPORT_DELIVERY_ERRORS as retry_error:
                    if isinstance(retry_error, AgentHTTPError) and not 500 <= retry_error.status < 600:
                        self._record_report_abandonment(command_id, retry_error, log_path, "permanent_http")
                        return {}
                    if attempt + 1 == REPORT_FALLBACK_ATTEMPTS:
                        self._report_retries_exhausted.set()
                        self._record_report_abandonment(command_id, retry_error, log_path, "retry_exhausted_agent_stopping")
                        return {}
                    self._log_lifecycle(
                        "failure report pending for {}: {}".format(
                            command_id, type(retry_error).__name__
                        )
                    )
                    if self._shutting_down:
                        return {}
                    threading.Event().wait(1.0)
        self._record_shutdown_control(response)
        self._prepare_received_response(response)
        with self._state_lock:
            self._pending_command_reports.discard(command_id)
        self._dispatch_response_commands(response, background=True)
        return response

    def _record_report_abandonment(self, command_id, error, log_path, reason):
        with self._state_lock:
            self._pending_command_reports.discard(command_id)
        status = error.status if isinstance(error, AgentHTTPError) else type(error).__name__
        self._log_lifecycle(
            "report abandoned command_id={} reason={} status={} log_path={}".format(
                command_id, reason, status, log_path
            )
        )

    def _post(self, path, payload, token=None):
        return _run_sync(self._post_async(path, payload, token=token))

    async def _post_async(self, path, payload, token=None, session=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if session is not None:
            return await self._post_with_session(session, path, payload, headers)
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as owned_session:
            return await self._post_with_session(owned_session, path, payload, headers)

    async def _post_with_session(self, session, path, payload, headers):
        async with session.post(self.controller_url + path, json=payload, headers=headers) as response:
            raw = await response.text()
            try:
                result = json.loads(raw) if raw else {}
            except Exception:
                result = {"error": raw}
            if response.status >= 400:
                raise AgentHTTPError(response.status, result)
            return result

    def _resources_dict(self, resources):
        return {"cpu_cores": resources.cpu_cores, "memory_gb": resources.memory_gb, "gpu_uuids": list(resources.gpu_uuids)}

    def _capabilities_dict(self, caps):
        return {
            "adapter": caps.adapter,
            "image_formats": list(caps.image_formats),
            "isolation_policies": list(caps.isolation_policies),
            "nested_container": caps.nested_container,
            "can_bind_mounts": caps.can_bind_mounts,
            "can_select_gpus": caps.can_select_gpus,
            "gpu_enforcement": caps.gpu_enforcement.value if hasattr(caps.gpu_enforcement, "value") else caps.gpu_enforcement,
            "can_enforce_cpu_memory": caps.can_enforce_cpu_memory,
            "node_local_cache": caps.node_local_cache,
            "supports_warm_sessions": caps.supports_warm_sessions,
            "privileged_allowed": caps.privileged_allowed,
            "gpu_isolation_policies": list(caps.gpu_isolation_policies),
            "reduced_isolation_policies": list(
                caps.reduced_isolation_policies
            ),
        }

    def _session_spec_from_dict(self, data):
        mounts = []
        for item in data.get("mounts", ()): 
            mounts.append(MountSpec(item["source"], item["target"], item.get("read_only", False)))
        return ContainerSessionSpec(
            execution_session_id=data["execution_session_id"],
            image=data.get("image"),
            mounts=tuple(mounts),
            cwd=data.get("cwd"),
            env=data.get("env", {}),
            gpu_uuids=tuple(data.get("gpu_uuids", ())),
            privileged=data.get("privileged", False),
            cache_dir=data.get("cache_dir"),
            tmp_dir=data.get("tmp_dir"),
            runtime_dir=data.get("runtime_dir"),
            oci_bundle=data.get("oci_bundle"),
            isolation_policy=data.get("isolation_policy", "default"),
            state_layer_id=data.get("state_layer_id"),
            overlay_path=data.get("overlay_path"),
            overlay_upper=data.get("overlay_upper"),
            overlay_work=data.get("overlay_work"),
        )

    def _log_path(self, command_id):
        os.makedirs(self.log_dir, exist_ok=True)
        return os.path.join(self.log_dir, command_id + ".log")

    def _session_log_path(self, execution_session_id):
        os.makedirs(self.log_dir, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", execution_session_id)
        return os.path.join(self.log_dir, safe_id + ".log")

    def _write_log(self, path, stdout, stderr, header=None):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as fh:
            if header:
                fh.write(header)
                if not header.endswith("\n"):
                    fh.write("\n")
                fh.write("=== stdout/stderr ===\n")
            if stdout:
                fh.write(stdout)
            if stderr:
                if stdout and not stdout.endswith("\n"):
                    fh.write("\n")
                fh.write(stderr)

    def _command_log_header(
        self,
        assignment,
        runtime_name,
        session_spec,
        executions,
        exit_code,
    ):
        command_id = assignment.get("command_id") or "unknown"
        request_id = assignment.get("request_id") or "unknown"
        execution_session_id = assignment.get("execution_session_id") or "unknown"
        session_data = assignment.get("session_spec", {})
        command_data = assignment.get("command_spec", {})
        isolation_policy = session_data.get("isolation_policy", "default")
        cwd = command_data.get("cwd") or session_data.get("cwd") or "<inherited>"
        lines = [
            "=== CAPE command {} (request {}, session {}) ===".format(
                command_id, request_id, execution_session_id
            ),
            "runtime-adapter: {}".format(runtime_name or "unknown"),
            "isolation-policy: {}".format(isolation_policy),
            "cwd: {}".format(cwd),
        ]
        lines.extend(self._execution_log_lines(executions))
        lines.append("exit-code: {}".format(exit_code))
        return "\n".join(lines) + "\n"

    def _session_log_header(
        self, execution_session_id, runtime_name, context, executions
    ):
        lines = [
            "=== CAPE session {} lifecycle ===".format(execution_session_id),
            "runtime-adapter: {}".format(runtime_name or "unknown"),
            "isolation-policy: {}".format(
                context.get("isolation_policy", "unknown")
            ),
        ]
        if context.get("request_id"):
            lines.append("last-request: " + context["request_id"])
        if context.get("cleanup_status"):
            lines.append("cleanup-status: " + context["cleanup_status"])
        lines.extend(self._execution_log_lines(executions))
        return "\n".join(lines) + "\n"

    def _execution_log_lines(self, executions):
        lines = []
        for execution in executions:
            argv = self._redacted_argv(execution.argv)
            env = self._redacted_env(execution.env)
            lines.append("phase: " + execution.phase)
            lines.append(
                "runtime-cwd: {}".format(execution.cwd or "<inherited>")
            )
            lines.append(
                "env: "
                + " ".join(
                    "{}={}".format(key, env[key]) for key in sorted(env)
                )
            )
            lines.append(
                "exec: " + " ".join(shlex.quote(part) for part in argv)
            )
            lines.append("argv-json: " + json.dumps(argv))
            lines.append(
                "phase-exit-code: {}".format(
                    execution.exit_code
                    if execution.exit_code is not None
                    else "unknown"
                )
            )
        return lines

    def _redacted_env(self, env):
        return {
            str(key): (
                "***" if self._sensitive_key(key) else str(value)
            )
            for key, value in dict(env or {}).items()
        }

    def _redacted_argv(self, argv):
        redacted = [str(part) for part in argv]
        index = 0
        while index < len(redacted):
            part = redacted[index]
            if part == "--setenv" and index + 2 < len(redacted):
                if self._sensitive_key(redacted[index + 1]):
                    redacted[index + 2] = "***"
                index += 3
                continue
            if part in ("--env", "-e") and index + 1 < len(redacted):
                redacted[index + 1] = self._redacted_assignment(
                    redacted[index + 1]
                )
                index += 2
                continue
            if part.startswith("--env="):
                redacted[index] = "--env=" + self._redacted_assignment(
                    part[len("--env=") :]
                )
            elif part.startswith("--") and "=" in part:
                option, value = part.split("=", 1)
                if self._sensitive_key(option.lstrip("-").replace("-", "_")):
                    redacted[index] = option + "=***"
            elif part.startswith("--") and self._sensitive_key(
                part.lstrip("-").replace("-", "_")
            ):
                if index + 1 < len(redacted):
                    redacted[index + 1] = "***"
                    index += 2
                    continue
            else:
                redacted[index] = self._redacted_assignment(part)
            index += 1
        return redacted

    def _redacted_assignment(self, value):
        key, separator, _raw_value = value.partition("=")
        if separator and self._sensitive_key(key):
            return key + "=***"
        return value

    def _sensitive_key(self, key):
        normalized = str(key).strip().upper().replace("-", "_")
        if normalized in self.redact_env_keys:
            return True
        return any(
            marker in normalized
            for marker in (
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "PASSWD",
                "PRIVATE_KEY",
                "API_KEY",
                "ACCESS_KEY",
                "KEY",
                "CREDENTIAL",
                "AUTHORIZATION",
            )
        )

    def _load_state(self):
        with self._state_lock:
            self._unreconciled_workspace_commands = False
            try:
                with open(self.state_path) as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    raise ValueError("agent state must be an object")
                if data.get("version") != 2:
                    raise ValueError("unsupported agent state schema; use a fresh state path")
                for name in ("local_sessions", "running_commands", "closed_workspace_sessions", "purged_executions"):
                    if name in data and (not isinstance(data[name], list) or any(not isinstance(item, str) for item in data[name])):
                        raise ValueError("invalid agent state field: " + name)
                for name in ("session_contexts", "workspace_updates", "workspace_bindings", "workspace_acknowledged", "workspace_operation_receipts",
                             "suspended_contexts", "resumed_versions", "pending_suspensions"):
                    if name in data and not isinstance(data[name], dict):
                        raise ValueError("invalid agent state field: " + name)
                for name in ("resumed_versions", "pending_suspensions"):
                    if any(type(value) is not int or not 0 <= value < 2 ** 63 for value in data.get(name, {}).values()):
                        raise ValueError("invalid execution generation: " + name)
                for context in data.get("suspended_contexts", {}).values():
                    if (not isinstance(context, dict) or not isinstance(context.get("session_spec"), dict)
                            or type(context.get("suspend_version")) is not int or not 0 <= context["suspend_version"] < 2 ** 63):
                        raise ValueError("invalid suspended execution metadata")
                if not data.get("running_commands") and any(identity not in data.get("session_contexts", {})
                        for identity in data.get("local_sessions", [])):
                    raise ValueError("cached execution metadata is missing; explicit recovery required")
                self.local_sessions = set(data.get("local_sessions", ()))
                self.session_contexts = dict(
                    data.get("session_contexts", {})
                )
                self.suspended_contexts = dict(data.get("suspended_contexts", {}))
                self.resumed_versions = dict(data.get("resumed_versions", {}))
                self.pending_suspensions = dict(data.get("pending_suspensions", {}))
                self.purged_executions = set(data.get("purged_executions", []))
                self.workspace_updates = dict(data.get("workspace_updates", {}))
                self.workspace_operation_receipts = dict(data.get("workspace_operation_receipts", {}))
                for identity, receipt in self.workspace_operation_receipts.items():
                    if (not isinstance(identity, str) or re.fullmatch(r"ws-[0-9a-f]{32}", identity) is None
                            or not isinstance(receipt, dict) or type(receipt.get("sequence")) is not int
                            or not 1 <= receipt["sequence"] < 2 ** 63 or not isinstance(receipt.get("id"), str)
                            or re.fullmatch(r"[0-9a-f]{32}", receipt["id"]) is None or not isinstance(receipt.get("update"), dict)):
                        raise ValueError("invalid workspace operation receipt")
                self.workspace_bindings = dict(data.get("workspace_bindings", {}))
                self.workspace_acknowledged = dict(data.get("workspace_acknowledged", {}))
                self._closed_workspace_sessions = set(data.get("closed_workspace_sessions", ()))
                self._unreconciled_workspace_commands = bool(data.get("running_commands")) and any(
                    context.get("session_spec", {}).get("workspace") for context in self.session_contexts.values())
            except FileNotFoundError:
                self.local_sessions = set()
                self.session_contexts = {}
                self.workspace_updates = {}
                self.workspace_operation_receipts = {}
                self.workspace_bindings = {}
                self.workspace_acknowledged = {}
                self._closed_workspace_sessions = set()
            except Exception as error:
                raise WorkspaceError("cannot safely restore agent state; preserve it for reconciliation: " + str(error))

    def _save_state(self):
        with self._state_lock:
            parent = os.path.dirname(self.state_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            payload = {
                "version": 2,
                "local_sessions": sorted(self.local_sessions),
                "running_commands": sorted(self.running_commands),
                "session_contexts": self.session_contexts,
                "suspended_contexts": getattr(self, "suspended_contexts", {}),
                "resumed_versions": getattr(self, "resumed_versions", {}),
                "pending_suspensions": getattr(self, "pending_suspensions", {}),
                "purged_executions": sorted(getattr(self, "purged_executions", ())),
                "workspace_updates": self.workspace_updates,
                "workspace_operation_receipts": self.workspace_operation_receipts,
                "workspace_bindings": self.workspace_bindings,
                "workspace_acknowledged": self.workspace_acknowledged,
                "closed_workspace_sessions": sorted(self._closed_workspace_sessions),
            }
            temporary_path = self.state_path + ".tmp." + uuid.uuid4().hex
            try:
                fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "w") as fh:
                    json.dump(payload, fh, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temporary_path, self.state_path)
                directory = os.open(parent or ".", os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass


def local_node_resources(cpu_cores=1, memory_gb=1, gpu_uuids=()):
    return NodeResources(cpu_cores=cpu_cores, memory_gb=memory_gb, gpu_uuids=tuple(gpu_uuids))


def local_runtime_capabilities(adapter="local-process"):
    return RuntimeCapabilities(
        adapter=adapter,
        image_formats=("default",),
        isolation_policies=("default", "none"),
        nested_container=True,
        can_bind_mounts=True,
        can_select_gpus=True,
        gpu_enforcement=EnforcementMode.NONE,
        can_enforce_cpu_memory=False,
        node_local_cache=True,
        supports_warm_sessions=True,
        privileged_allowed=False,
    )
