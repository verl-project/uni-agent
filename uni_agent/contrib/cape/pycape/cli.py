# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time

import aiohttp

from .agent import NodeAgentClient, local_node_resources, local_runtime_capabilities
from .credential_file import prepare_registration, read_registration
from .instance import Instance
from .managed_auth import ManagedAuthority
from .identity_cli import add_user_parser, add_admin_parsers, add_node_action_parsers, run_user, run_admin
from .controller import CAPEController, SchedulerPolicy
from .journal import SQLiteJournal
from .workspace import WorkspaceConfig
from .workspace_fs import WorkspaceError
from .models import (
    GPU_CONTENTION_POLICIES,
    POOL_MODES,
    POOL_PLACEMENTS,
    aggregate_runtime_capabilities,
)
from .observability import request_status_message
from .profiles import ProfileRegistry
from .runtime import (
    ApptainerRuntimeAdapter,
    BubbleWrapRuntimeAdapter,
    LocalProcessRuntimeAdapter,
    PodmanRuntimeAdapter,
)
from .service import CAPEControllerService


DEFAULT_CONTROLLER_URL = "http://127.0.0.1:8765"


class CapeHTTPError(Exception):
    def __init__(self, status, payload):
        Exception.__init__(self, payload)
        self.status = status
        self.payload = payload


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

    thread = threading.Thread(target=runner, name="cape-aiohttp-cli")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _controller_url(value=None):
    return (value or os.environ.get("CAPE_CONTROLLER_URL") or DEFAULT_CONTROLLER_URL).rstrip("/")


def _request(method, controller_url, path, payload=None, token=None, query=None, timeout=60):
    return _run_sync(
        _request_async(
            method,
            controller_url,
            path,
            payload=payload,
            token=token,
            query=query,
            timeout=timeout,
        )
    )


async def _request_async(method, controller_url, path, payload=None, token=None, query=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request_kwargs = {"headers": headers}
    if payload is not None:
        request_kwargs["json"] = payload
    if query:
        request_kwargs["params"] = query
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.request(
            method,
            _controller_url(controller_url) + path,
            **request_kwargs
        ) as response:
            raw = await response.text()
            try:
                result = json.loads(raw) if raw else {}
            except Exception:
                result = {"error": raw}
            if response.status >= 400:
                raise CapeHTTPError(response.status, result)
            return result


def _print_json(payload, stdout=None):
    out = stdout or sys.stdout
    out.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_error(exc, stderr=None):
    err = stderr or sys.stderr
    payload = {"status": exc.status, "error": exc.payload}
    err.write(json.dumps(payload, sort_keys=True) + "\n")


def _strip_command_separator(command):
    command = list(command or [])
    if command and command[0] == "--":
        command = command[1:]
    return command


def _parse_bind(value):
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("bind must be src:dst[:ro|rw]")
    read_only = False
    if len(parts) == 3:
        mode = parts[2].lower()
        if mode not in ("ro", "rw"):
            raise ValueError("bind mode must be ro or rw")
        read_only = mode == "ro"
    if not parts[0] or not parts[1]:
        raise ValueError("bind source and target are required")
    return {"source": parts[0], "target": parts[1], "read_only": read_only}


def _parse_env(values):
    env = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("env entries must be KEY=VALUE")
        key, val = value.split("=", 1)
        if not key:
            raise ValueError("env key is required")
        env[key] = val
    return env


def _build_run_payload(args):
    command = _strip_command_separator(args.workload_command)
    if not command:
        raise ValueError("command is required")
    resources = {
        "gpus": args.gpus,
        "cpu_cores": args.cpu_cores,
        "memory_gb": args.memory_gb,
        "max_duration_seconds": (
            args.max_duration_seconds
            if getattr(args, "max_duration_seconds", None) is not None
            else 3600
        ),
        "command_timeout_seconds": getattr(
            args, "command_timeout_seconds", None
        ),
    }
    if getattr(args, "runtime_adapter", None):
        resources["runtime_adapter"] = args.runtime_adapter
    if getattr(args, "hard_gpu_enforcement", False):
        resources["hard_gpu_enforcement"] = True
    if getattr(args, "port", None):
        resources["ports"] = list(args.port)
    if getattr(args, "locality", None):
        resources["locality"] = args.locality
    if getattr(args, "isolation_policy", None):
        resources["isolation_policy"] = args.isolation_policy
    payload = {
        "pool": args.pool,
        "user_id": args.user,
        "session_id": args.session_id,
        "workspace": {"id": getattr(args, "workspace_id", None), "target": getattr(args, "workspace_target", "/workspace")},
        "command": command,
        "cwd": args.cwd,
        "env": _parse_env(getattr(args, "env", [])),
        "image": args.image,
        "profile": args.profile,
        "keep_session_on_failure": not args.suspend_on_failure,
        "resources": resources,
        "mounts": [_parse_bind(value) for value in getattr(args, "bind", [])],
        "log_path": getattr(args, "log_path", None),
    }
    artifact_paths = getattr(args, "artifact_path", None)
    if artifact_paths:
        payload["artifact_paths"] = list(artifact_paths)
    return payload


def _run_from_args(args, stdout=None, no_wait_output="json"):
    from .session_identity import SessionSelection
    identity = _request("GET", args.controller_url, "/v1/identity/me", token=args.token)
    with SessionSelection(identity, args.session_id) as selection:
        args.session_id = selection.session_id
        status = _request("POST", args.controller_url, "/v1/submit", payload=_build_run_payload(args), token=args.token)
        selection.accepted(status)
    print("session_id=" + args.session_id, file=sys.stderr)
    if args.wait:
        wait_query = {"timeout": args.timeout}
        if args.running_timeout is not None:
            wait_query["running_timeout"] = args.running_timeout
        status = _request(
            "GET",
            args.controller_url,
            "/v1/requests/{}/wait".format(status["request_id"]),
            token=args.token,
            query=wait_query,
            timeout=max(60.0, args.timeout + 5.0),
        )
        if status.get("wait_timed_out") and args.cancel_on_timeout:
            timeout_kind = status.get("wait_timeout_kind")
            status = _request(
                "POST",
                args.controller_url,
                "/v1/requests/{}/cancel".format(status["request_id"]),
                payload={"reason": args.cancel_reason},
                token=args.token,
            )
            status["wait_timed_out"] = True
            status["wait_timeout_kind"] = timeout_kind
        _print_json(status, stdout=stdout)
        if status.get("exit_code") is not None:
            try:
                return int(status.get("exit_code") or 0)
            except Exception:
                return 1
        return 0
    if no_wait_output == "id":
        out = stdout or sys.stdout
        out.write(status["request_id"] + "\n")
    else:
        _print_json(status, stdout=stdout)
    return 0


def _status_from_args(args, stdout=None, stderr=None):
    status = _request(
        "GET",
        args.controller_url,
        "/v1/requests/{}".format(args.request_id),
        token=args.token,
    )
    status["status_message"] = status.get("status_message") or request_status_message(
        status.get("state"), status.get("reason")
    )
    (stderr or sys.stderr).write(status["status_message"] + "\n")
    _print_json(status, stdout=stdout)
    return 0


def _wait_from_args(args, stdout=None):
    wait_query = {"timeout": args.timeout}
    if args.running_timeout is not None:
        wait_query["running_timeout"] = args.running_timeout
    status = _request(
        "GET",
        args.controller_url,
        "/v1/requests/{}/wait".format(args.request_id),
        token=args.token,
        query=wait_query,
        timeout=max(60.0, args.timeout + 5.0),
    )
    _print_json(status, stdout=stdout)
    if status.get("exit_code") is not None:
        try:
            return int(status.get("exit_code") or 0)
        except Exception:
            return 1
    return 0


def _cancel_from_args(args, stdout=None):
    payload = _request(
        "POST",
        args.controller_url,
        "/v1/requests/{}/cancel".format(args.request_id),
        payload={"reason": args.reason},
        token=args.token,
    )
    _print_json(payload, stdout=stdout)
    return 0


def _logs_from_args(args, stdout=None, stderr=None):
    payload = _request("GET", args.controller_url, "/v1/requests/{}/logs".format(args.request_id), token=args.token)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    out.write(payload.get("stdout") or "")
    err.write(payload.get("stderr") or "")
    return 0


def _utilization_from_args(args, stdout=None):
    _print_json(_request("GET", args.controller_url, "/v1/pools/{}/utilization".format(args.pool), token=args.token), stdout=stdout)
    return 0


def _adapter_runtime_path(explicit_path, common_root, adapter):
    if explicit_path:
        return explicit_path
    if common_root:
        return os.path.join(common_root, adapter)
    return None


def _runtime_config(args):
    selected = []
    for runtime_name in args.runtime_adapter or ():
        if runtime_name not in selected:
            selected.append(runtime_name)
    explicit_filters = set(selected)
    if not selected:
        selected = ["apptainer", "bubblewrap", "podman"]

    runtime_adapters = {}
    runtime_capabilities = {}

    for runtime_name in selected:
        if runtime_name == "local-process":
            runtime_adapters[runtime_name] = LocalProcessRuntimeAdapter()
            runtime_capabilities[runtime_name] = local_runtime_capabilities(
                adapter=runtime_name
            )
            continue
        if runtime_name == "apptainer":
            runtime = ApptainerRuntimeAdapter(
                binary=args.apptainer_binary,
                contain=not args.no_contain,
                unsquash=args.apptainer_unsquash,
                cache_dir=_adapter_runtime_path(
                    args.apptainer_cache, args.runtime_cache, "apptainer"
                ),
                tmp_dir=_adapter_runtime_path(
                    args.apptainer_tmp, args.runtime_tmp, "apptainer"
                ),
                runtime_dir=_adapter_runtime_path(
                    args.apptainer_runtime_dir,
                    args.runtime_dir,
                    "apptainer",
                ),
                stateful_sessions=not args.no_warm_sessions,
                probe_image=args.apptainer_probe_image,
            )
        elif runtime_name == "bubblewrap":
            runtime = BubbleWrapRuntimeAdapter(
                binary=args.bubblewrap_binary,
                cache_dir=_adapter_runtime_path(
                    args.bubblewrap_cache, args.runtime_cache, "bubblewrap"
                ),
                tmp_dir=_adapter_runtime_path(
                    args.bubblewrap_tmp, args.runtime_tmp, "bubblewrap"
                ),
                runtime_dir=_adapter_runtime_path(
                    args.bubblewrap_runtime_dir,
                    args.runtime_dir,
                    "bubblewrap",
                ),
                stateful_sessions=not args.no_warm_sessions,
                node_gpu_uuids=tuple(args.gpu_uuid),
                system_paths=args.bubblewrap_system_path,
                read_only_paths=args.bubblewrap_read_only_path,
            )
        elif runtime_name == "podman":
            runtime = PodmanRuntimeAdapter(
                binary=args.podman_binary,
                graphroot=_adapter_runtime_path(
                    args.podman_graphroot, args.runtime_cache, "podman"
                ),
                tmpdir=_adapter_runtime_path(
                    args.podman_tmpdir, args.runtime_tmp, "podman"
                ),
                runroot=_adapter_runtime_path(
                    args.podman_runroot, args.runtime_dir, "podman"
                ),
                probe_image=args.podman_probe_image,
                warm_sessions=not args.no_warm_sessions,
            )
        else:
            raise ValueError("unsupported runtime adapter: " + runtime_name)

        runtime.execution_state_dir = getattr(args, "execution_state_dir", None) or os.path.join(
            os.path.dirname(os.path.abspath(args.state_path)) if getattr(args, "state_path", None)
            else os.path.abspath(getattr(args, "log_dir", None) or ".pycape-agent-logs"), "execution-state")
        runtime.execution_state_min_free_bytes = getattr(args, "execution_state_min_free_bytes", 512 * 1024 ** 2)
        runtime.execution_state_max_bytes = getattr(args, "execution_state_max_bytes", 0)
        try:
            capabilities = runtime.probe(execute=True)
        except Exception as exc:
            if runtime_name in explicit_filters:
                raise RuntimeError(
                    "requested runtime adapter probe raised an error: {}: {}".format(
                        runtime_name, exc
                    )
                )
            print("runtime probe skipped {}: {}".format(runtime_name, exc), file=sys.stderr)
            continue
        if capabilities.nested_container and capabilities.isolation_policies:
            runtime_adapters[runtime_name] = runtime
            runtime_capabilities[runtime_name] = capabilities
        elif runtime_name in explicit_filters:
            raise RuntimeError(
                "requested runtime adapter failed its execution probe: "
                + runtime_name
            )
        else:
            print("runtime probe skipped {}: unavailable or execution probe failed".format(runtime_name), file=sys.stderr)

    if not runtime_adapters:
        raise RuntimeError("no runtime adapter passed its execution probe")

    return (
        aggregate_runtime_capabilities(runtime_capabilities),
        runtime_adapters,
        runtime_capabilities,
    )


def _runtime_cache_config_from_args(args):
    runtime_cache_config = args.runtime_cache_config
    if runtime_cache_config == "default" and args.runtime_cache:
        runtime_cache_config = args.runtime_cache
    return runtime_cache_config


def _node_agent_from_args(args, agent_epoch=None, workspace_config=None):
    caps, runtime_adapters, runtime_capabilities = _runtime_config(args)
    return NodeAgentClient(
        _controller_url(args.controller_url),
        node_id=args.node_id,
        pool=args.pool,
        resources=local_node_resources(args.cpu_cores, args.memory_gb, tuple(args.gpu_uuid)),
        capabilities=caps,
        runtime_adapters=runtime_adapters,
        runtime_capabilities=runtime_capabilities,
        join_token=args.join_token,
        heartbeat_token=args.token,
        allocation_id=args.allocation_id,
        allocation_deadline=args.allocation_deadline,
        drain_before_sec=args.drain_before_sec,
        cleanup_buffer_sec=args.cleanup_buffer_sec,
        runtime_cache_config=_runtime_cache_config_from_args(args),
        cached_images=tuple(args.cached_image),
        log_dir=args.log_dir,
        state_path=args.state_path,
        redact_env_keys=tuple(args.log_redact_env_key),
        agent_epoch=agent_epoch,
        workspace_config=workspace_config,
        exit_on_removal=not args.stay_after_removal,
        pool_mode=args.pool_mode,
        gpu_slots=args.gpu_slots,
        placement=args.pool_placement,
        metrics_enabled=getattr(args, "metrics", False),
    )


def _node_announcement_payload(args, agent_epoch):
    resources = local_node_resources(
        args.cpu_cores, args.memory_gb, tuple(args.gpu_uuid)
    )
    return {
        "node_id": args.node_id,
        "pool": args.pool,
        "pool_mode": args.pool_mode,
        "gpu_slots": args.gpu_slots,
        "placement": args.pool_placement,
        "resources": {
            "cpu_cores": resources.cpu_cores,
            "memory_gb": resources.memory_gb,
            "gpu_uuids": list(resources.gpu_uuids),
        },
        "allocation_id": args.allocation_id,
        "allocation_deadline": args.allocation_deadline,
        "drain_before_sec": args.drain_before_sec,
        "cleanup_buffer_sec": args.cleanup_buffer_sec,
        "runtime_cache_config": _runtime_cache_config_from_args(args),
        "cached_images": list(args.cached_image),
        "join_token": args.join_token,
        "agent_epoch": agent_epoch,
    }


def _probe_heartbeat_loop(args, agent_epoch, stop):
    interval = max(0.01, float(args.heartbeat_interval))
    while not stop.wait(interval):
        try:
            _request(
                "POST",
                args.controller_url,
                "/v1/agent/heartbeat",
                payload={
                    "node_id": args.node_id,
                    "agent_epoch": agent_epoch,
                    "state": "REGISTERING",
                    "status": {
                        "authoritative": False,
                        "phase": "runtime_probe",
                    },
                    "max_commands": 0,
                },
                token=args.token,
            )
        except Exception:
            continue


def _node_enroll_from_args(args, stdout=None, stderr=None):
    try:
        return _node_enroll_impl(args, stdout, stderr)
    finally:
        metrics_link = getattr(args, "_metrics_link", None)
        if metrics_link is not None:
            metrics_link.close()
        guard = getattr(args, "_enrollment_lock", None)
        if guard is not None:
            guard.close()


def _node_enroll_impl(args, stdout=None, stderr=None):
    from .node_enrollment import enroll, EnrollmentUncertain
    try:
        agent_epoch = enroll(args, _request, _controller_url(args.controller_url))
    except EnrollmentUncertain as error:
        _print_json({"outcome": "unknown", "operation_id": error.operation_id, "token_file": error.path,
                     "recovery": "node enroll --resume"}, stdout=stderr or sys.stderr)
        return 3
    announcement = _request(
        "POST",
        args.controller_url,
        "/v1/agent/announce",
        payload=_node_announcement_payload(args, agent_epoch),
        token=args.token,
    )
    probe_stop = threading.Event()
    probe_heartbeat = threading.Thread(
        target=_probe_heartbeat_loop,
        args=(args, agent_epoch, probe_stop),
        name="cape-runtime-probe-heartbeat",
    )
    probe_heartbeat.daemon = True
    probe_heartbeat.start()
    try:
        agent = _node_agent_from_args(args, agent_epoch=agent_epoch,
                                      workspace_config=announcement.get("workspace_config", WorkspaceConfig().to_dict()))
        args._metrics_link = agent.metrics_link
    except BaseException:
        try:
            _request(
                "POST",
                args.controller_url,
                "/v1/agent/heartbeat",
                payload={
                    "node_id": args.node_id,
                    "agent_epoch": agent_epoch,
                    "state": "UNHEALTHY",
                    "status": {
                        "authoritative": False,
                        "phase": "runtime_probe_failed",
                    },
                    "max_commands": 0,
                },
                token=args.token,
            )
        except Exception:
            pass
        raise
    finally:
        probe_stop.set()
        probe_heartbeat.join(max(1.0, float(args.heartbeat_interval) + 1.0))
    if args.once:
        payload = agent.register()
        try:
            payload = agent.heartbeat_once(background=False)
        finally:
            agent.join(timeout=30.0)
            agent._stop_all_local_sessions()
            if not agent._flush_workspace_updates():
                raise WorkspaceError("workspace cleanup status could not be acknowledged; state is preserved for retry")
    else:
        payload = agent.register_until_available(
            interval=args.heartbeat_interval
        )
        _print_json(payload, stdout=stdout)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: agent.request_drain(),
        )
        try:
            agent.run_forever(interval=args.heartbeat_interval)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        return 0
    _print_json(payload, stdout=stdout)
    return 0


def _admin_from_args(args, stdout=None):
    if args.admin_command in ("get-workspace-config", "set-workspace-config"):
        if args.admin_command == "get-workspace-config":
            query = {"pool": args.pool}
            if args.node_id is not None:
                query["node_id"] = args.node_id
            payload = _request("GET", args.controller_url, "/v1/admin/workspace-config", query=query, token=args.token)
        else:
            changes = {}
            for option, field in (("image_size_gib", "image_size_bytes"), ("max_image_size_gib", "max_image_size_bytes")):
                value = getattr(args, option)
                if value is not None:
                    if value < 1 or value * 1024 ** 3 > 2 ** 63 - 1:
                        raise ValueError(option + " is out of range")
                    changes[field] = value * 1024 ** 3
            for field in ("image_root", "bytes_per_inode", "mount_timeout_seconds"):
                value = getattr(args, field)
                if value is not None:
                    changes[field] = value
            maintenance = {}
            for field in ("auto_grow", "reclaim_on_release"):
                value = getattr(args, field, None)
                if value is not None:
                    maintenance[field] = value == "on"
            for field in ("grow_threshold_percent", "grow_target_percent", "grow_cooldown_seconds", "reclaim_cooldown_seconds",
                          "reclaim_idle_seconds", "reclaim_start_interval_seconds", "reclaim_max_candidates",
                          "reclaim_retry_backoff_seconds", "reclaim_low_yield_backoff_seconds", "reclaim_nice"):
                value = getattr(args, field, None)
                if value is not None:
                    maintenance[field] = value
            for option, field, scale in (("grow_min_mib", "grow_min_bytes", 1024 ** 2),
                                         ("reclaim_min_mib", "reclaim_min_allocated_bytes", 1024 ** 2),
                                         ("backing_reserve_mib", "backing_reserve_bytes", 1024 ** 2),
                                         ("maintenance_timeout_seconds", "timeout_seconds", 1)):
                value = getattr(args, option, None)
                if value is not None:
                    maintenance[field] = value * scale
            if maintenance:
                changes["maintenance"] = maintenance
            payload = _request("POST", args.controller_url, "/v1/admin/workspace-config", token=args.token,
                               payload={"pool": args.pool, "node_id": args.node_id, "changes": changes})
    else:
        raise ValueError("unknown admin command")
    _print_json(payload, stdout=stdout)
    return 0


def _controller_start_from_args(args, stdout=None, stderr=None):
    instance = None
    journal = None
    service = None
    metrics_supervisor = None
    metrics_state = None
    previous_sigterm = None
    try:
        if threading.current_thread() is threading.main_thread():
            def terminate(_signum, _frame):
                raise KeyboardInterrupt()
            previous_sigterm = signal.signal(signal.SIGTERM, terminate)
        token = read_registration(args.token_file)["secret"] if args.token_file else args.token
        instance = Instance.open(args.state_dir, token)
        from .telemetry_control import set_config
        if args.metrics:
            set_config(instance.store, token, {"enabled": True})
        if instance.store.snapshot().get("telemetry", {}).get("default", {}).get("enabled", False):
            from .telemetry_supervisor import PrometheusSupervisor
            metrics_supervisor = PrometheusSupervisor(args.state_dir, instance.store.snapshot()["scheduler_id"],
                                                      os.environ.get("CAPE_PROMETHEUS_BIN", "prometheus"))
            try:
                metrics_state = metrics_supervisor.start()
            except (OSError, ValueError, RuntimeError):
                metrics_state = {"state": "unavailable", "reason": "prometheus_start_failed"}
        token = None
        args.token = None
        auth = ManagedAuthority(instance.store)
        journal = SQLiteJournal(instance.database)
        overrides = {}
        for gpu_uuid, override_policy, override_ratio in (
            args.gpu_share_override or []
        ):
            if gpu_uuid in overrides:
                raise ValueError(
                    "duplicate GPU share override: " + gpu_uuid
                )
            overrides[gpu_uuid] = {
                "policy": override_policy,
                "ratio": (
                    args.gpu_shared_ratio
                    if override_ratio is None
                    else override_ratio
                ),
            }
        for name in (
            "max_active_gpu_sessions_per_user",
            "max_queued_requests_per_user",
            "max_active_sessions_per_pool",
        ):
            value = getattr(args, name)
            if value is not None and value < 0:
                raise ValueError(name + " must be non-negative")
        if args.idle_session_timeout_seconds < 0:
            raise ValueError("idle_session_timeout_seconds must be non-negative")
        if args.reconcile_missing_grace_seconds < 0:
            raise ValueError("reconcile_missing_grace_seconds must be non-negative")
        if args.reconcile_missing_heartbeats < 1:
            raise ValueError("reconcile_missing_heartbeats must be at least 1")
        profile_registry = (
            ProfileRegistry.from_json_file(args.profiles_file)
            if args.profiles_file
            else ProfileRegistry()
        )
        policy = SchedulerPolicy(
            max_active_gpu_sessions_per_user=args.max_active_gpu_sessions_per_user,
            max_queued_requests_per_user=args.max_queued_requests_per_user,
            max_active_sessions_per_pool=args.max_active_sessions_per_pool,
            heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
            lost_timeout_seconds=args.lost_timeout_seconds,
            idle_session_timeout_seconds=args.idle_session_timeout_seconds,
            reconcile_missing_grace_seconds=args.reconcile_missing_grace_seconds,
            reconcile_missing_heartbeats=args.reconcile_missing_heartbeats,
            profile_registry=profile_registry,
            gpu_contention_policy=args.gpu_contention_policy,
            gpu_shared_ratio=args.gpu_shared_ratio,
            gpu_share_overrides=overrides,
        )
        if journal is None:
            controller = CAPEController(auth=auth, policy=policy)
        else:
            controller = CAPEController.reconcile_from_journal(
                journal, auth=auth, policy=policy
            )
        if metrics_state and metrics_state["state"] == "ready":
            controller.telemetry_query_url = metrics_supervisor.query_url
            controller.telemetry_receiver_url = metrics_supervisor.receiver_url
        service = CAPEControllerService(
            controller=controller,
            host=args.host,
            port=args.port,
            schedule_interval=args.schedule_interval,
        ).start()
        out = stdout or sys.stdout
        ready = {"url": service.url}
        if metrics_state is not None:
            ready["metrics"] = metrics_state
        out.write(json.dumps(ready, sort_keys=True) + "\n")
        out.flush()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if service is not None:
                service.stop()
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            try:
                if metrics_supervisor is not None and metrics_supervisor._record is not None:
                    metrics_supervisor.stop()
            finally:
                try:
                    if journal is not None:
                        journal.close()
                finally:
                    if instance is not None:
                        instance.close()
    return 0


def _add_run_args(parser):
    parser.add_argument("--controller-url")
    parser.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
    parser.add_argument("--pool", required=True)
    parser.add_argument("--user", default="", help="optional identity consistency check; ownership comes from the token")
    parser.add_argument("--session-id", help="logical session; defaults to this controller/user/project's last accepted session")
    parser.add_argument("--image")
    parser.add_argument("--profile")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--memory-gb", type=int, default=1)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        help="per-command admission bound; defaults to max-duration-seconds when omitted",
    )
    parser.add_argument("--runtime-adapter")
    parser.add_argument("--hard-gpu-enforcement", action="store_true")
    parser.add_argument("--port", type=int, action="append", default=[])
    parser.add_argument("--locality")
    parser.add_argument("--isolation-policy", default="default")
    parser.add_argument("--cwd")
    parser.add_argument("--workspace-id", help="reuse a retained managed image workspace; does not replace --session-id")
    parser.add_argument("--workspace-target", default="/workspace", help="container bind target for the managed workspace")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument("--log-path")
    parser.add_argument("--artifact-path", action="append", default=[])
    parser.add_argument(
        "--suspend-on-failure",
        action="store_true",
        help="release cached execution resources after workload failure, retaining filesystem state",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--running-timeout", type=float)
    parser.add_argument("--cancel-on-timeout", action="store_true")
    parser.add_argument("--cancel-reason", default="wait_timeout")
    parser.add_argument("workload_command", nargs=argparse.REMAINDER)


def _add_node_enroll_args(parser):
    parser.add_argument("--controller-url")
    parser.add_argument("--node-id")
    parser.add_argument("--pool", required=True)
    parser.add_argument(
        "--pool-mode",
        choices=POOL_MODES,
        default="exclusive",
        help="pool GPU ownership mode (default: exclusive)",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=1,
        help="slots per GPU; shared pools require a positive value",
    )
    parser.add_argument(
        "--pool-placement",
        choices=POOL_PLACEMENTS,
        default="pack",
        help="shared-slot placement policy (default: pack)",
    )
    parser.set_defaults(join_token=None)
    parser.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--memory-gb", type=int, default=1)
    parser.add_argument("--gpu-uuid", action="append", default=[])
    parser.add_argument("--allocation-id")
    parser.add_argument("--allocation-deadline", type=float)
    parser.add_argument("--drain-before-sec", type=float, default=0)
    parser.add_argument("--cleanup-buffer-sec", type=float, default=0)
    parser.add_argument(
        "--runtime-adapter",
        action="append",
        choices=("local-process", "apptainer", "bubblewrap", "podman"),
        help="repeatable runtime filter; default: auto-probe apptainer, bubblewrap and podman; local-process is explicit-only",
    )
    parser.add_argument("--runtime-cache-config", default="default")
    parser.add_argument("--runtime-cache")
    parser.add_argument("--runtime-tmp")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--apptainer-binary")
    parser.add_argument("--apptainer-cache")
    parser.add_argument("--apptainer-tmp")
    parser.add_argument("--apptainer-runtime-dir")
    parser.add_argument("--bubblewrap-binary")
    parser.add_argument("--bubblewrap-cache")
    parser.add_argument("--bubblewrap-tmp")
    parser.add_argument("--bubblewrap-runtime-dir")
    parser.add_argument("--execution-state-dir", help="durable retained-layer root; defaults beside agent state")
    parser.add_argument("--execution-state-min-free-bytes", type=int, default=512 * 1024 ** 2)
    parser.add_argument("--execution-state-max-bytes", type=int, default=0, help="allocated-byte admission cap; 0 is unlimited")
    parser.add_argument(
        "--bubblewrap-system-path",
        action="append",
        help=(
            "repeatable node-local system/toolchain directory exposed by "
            "confined Bubblewrap host policies; default: /usr, /etc, /opt"
        ),
    )
    parser.add_argument(
        "--bubblewrap-read-only-path", action="append", default=[],
        help="repeatable optional virtual-kernel subpath bound read-only; roots and symlink escapes are refused; absent paths are skipped",
    )
    parser.add_argument("--podman-binary")
    parser.add_argument("--podman-graphroot")
    parser.add_argument("--podman-runroot")
    parser.add_argument("--podman-tmpdir")
    parser.add_argument("--apptainer-probe-image", dest="apptainer_probe_image",
                        default="docker://docker.io/library/ubuntu:24.04",
                        help="probe image (default: docker://docker.io/library/ubuntu:24.04); independent of workload image")
    parser.add_argument("--podman-probe-image", default="docker://docker.io/library/ubuntu:24.04",
                        help="probe image (default: docker://docker.io/library/ubuntu:24.04); independent of workload image")
    parser.add_argument("--apptainer-unsquash", action="store_true")
    parser.add_argument("--no-contain", action="store_true")
    parser.add_argument(
        "--no-warm-sessions",
        action="store_true",
        help="disable stateful writable-layer support",
    )
    parser.add_argument("--cached-image", action="append", default=[])
    parser.add_argument("--log-dir")
    parser.add_argument("--state-path")
    parser.add_argument(
        "--log-redact-env-key",
        action="append",
        default=[],
        help="additional environment key whose runtime-log value is masked",
    )
    parser.add_argument("--heartbeat-interval", type=float, default=1.0)
    parser.add_argument(
        "--stay-after-removal",
        action="store_true",
        help=(
            "keep heartbeating after authoritative removal instead of "
            "cleaning local sessions and exiting 0"
        ),
    )
    parser.add_argument("--once", action="store_true")


def _parse_gpu_share_override(value):
    gpu_uuid, separator, policy_ratio = value.partition("=")
    if not separator or not gpu_uuid or not policy_ratio:
        raise argparse.ArgumentTypeError(
            "expected GPU_UUID=POLICY[:RATIO]"
        )
    policy_name, ratio_separator, ratio_text = policy_ratio.partition(":")
    if policy_name not in GPU_CONTENTION_POLICIES:
        raise argparse.ArgumentTypeError(
            "policy must be one of: "
            + ", ".join(GPU_CONTENTION_POLICIES)
        )
    ratio = None
    if ratio_separator:
        try:
            ratio = int(ratio_text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "override ratio must be a positive integer"
            )
        if ratio <= 0:
            raise argparse.ArgumentTypeError(
                "override ratio must be a positive integer"
            )
    return gpu_uuid, policy_name, ratio


def _build_cape_parser():
    parser = argparse.ArgumentParser(prog="pycape")
    parser.add_argument("--version", action="version", version="pycape 0.1.0")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run")
    _add_run_args(run)

    sessions = sub.add_parser("session").add_subparsers(dest="session_command", required=True)
    for verb in ("list", "status", "suspend", "purge"):
        child = sessions.add_parser(verb)
        child.add_argument("--controller-url")
        child.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
        if verb == "status":
            child.add_argument("session_id")
        if verb in ("suspend", "purge"):
            child.add_argument("execution_id")
        if verb == "suspend":
            child.add_argument("--force", action="store_true")

    for name in ("status", "logs", "wait", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--controller-url")
        p.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
        if name == "wait":
            p.add_argument("--timeout", type=float, default=30.0)
            p.add_argument("--running-timeout", type=float)
        if name == "cancel":
            p.add_argument("--reason", default="cancelled")

    util = sub.add_parser("utilization")
    util.add_argument("--controller-url")
    util.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
    util.add_argument("--pool", required=True)

    node = sub.add_parser("node")
    node_sub = node.add_subparsers(dest="node_command")
    add_node_action_parsers(node_sub)
    enroll = node_sub.add_parser("enroll")
    enroll.add_argument("--metrics", action="store_true", help="attach optional host/scheduler shared DCGM metrics service")
    _add_node_enroll_args(enroll)
    enroll.add_argument("--node-token-out", required=True)
    enroll.add_argument("--token-file")
    enroll.add_argument("--resume", action="store_true")

    workspace = sub.add_parser("workspace")
    workspace.add_argument("workspace_action", choices=("status", "set", "release", "check", "delete", "grow", "reclaim", "shrink", "recover"))
    workspace.add_argument("--auto-reclaim", choices=("off", "inherit"))
    workspace.add_argument("workspace_id")
    workspace.add_argument("--size-gib", type=int)
    workspace.add_argument("--operation-id", help="retry/query the current server-issued maintenance operation")
    workspace.add_argument("--controller-url")
    workspace.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
    admin = sub.add_parser("admin")
    admin_sub = admin.add_subparsers(dest="admin_command")
    add_admin_parsers(admin_sub)
    for name in ("get-workspace-config", "set-workspace-config"):
        workspace = admin_sub.add_parser(name)
        workspace.add_argument("--controller-url")
        workspace.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
        workspace.add_argument("--pool", required=True)
        workspace.add_argument("--node-id")
        if name == "set-workspace-config":
            workspace.add_argument("--image-size-gib", type=int)
            workspace.add_argument("--max-image-size-gib", type=int)
            workspace.add_argument("--image-root")
            workspace.add_argument("--bytes-per-inode", type=int)
            workspace.add_argument("--mount-timeout-seconds", type=int)
            for flag in ("--auto-grow", "--reclaim-on-release"):
                workspace.add_argument(flag, choices=("on", "off"))
            for flag in ("--grow-threshold-percent", "--grow-target-percent", "--grow-min-mib", "--grow-cooldown-seconds",
                         "--reclaim-min-mib", "--reclaim-cooldown-seconds", "--maintenance-timeout-seconds", "--backing-reserve-mib",
                         "--reclaim-idle-seconds", "--reclaim-start-interval-seconds", "--reclaim-max-candidates",
                         "--reclaim-retry-backoff-seconds", "--reclaim-low-yield-backoff-seconds", "--reclaim-nice"):
                workspace.add_argument(flag, type=int)
    controller = sub.add_parser("controller")
    add_user_parser(sub)
    metric_commands = sub.add_parser("metric").add_subparsers(dest="metric_command", required=True)
    for action in ("status", "latest", "history"):
        metric_leaf = metric_commands.add_parser(action)
        metric_leaf.add_argument("--controller-url")
        metric_leaf.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
        metric_leaf.add_argument("--token-file")
        metric_leaf.add_argument("--node-id")
        if action != "history":
            metric_leaf.add_argument("--local-url", help="query a shared local service without credentials")
        if action != "status":
            metric_leaf.add_argument("--metric", required=True)
            metric_leaf.add_argument("--pool")
            metric_leaf.add_argument("--gpu-uuid")
        if action == "history":
            for field in ("start", "end", "step"):
                metric_leaf.add_argument("--" + field, type=float, required=True)
    controller_sub = controller.add_subparsers(dest="controller_command")
    initialize = controller_sub.add_parser("init")
    initialize.add_argument("--state-dir", required=True)
    initialize.add_argument("--user", required=True)
    initialize.add_argument("--token-out", required=True)
    initialize.add_argument("--metrics", action="store_true", help="enable telemetry control plane and start instance Prometheus")
    start = controller_sub.add_parser("start")
    start.add_argument("--metrics", action="store_true", help="enable telemetry; enabled instance configuration is otherwise restored")
    start.add_argument("--state-dir", required=True)
    credentials = start.add_mutually_exclusive_group()
    credentials.add_argument("--token")
    credentials.add_argument("--token-file")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)
    start.add_argument("--schedule-interval", type=float, default=0.05)
    start.add_argument("--heartbeat-timeout-seconds", type=float, default=30.0)
    start.add_argument("--lost-timeout-seconds", type=float, default=90.0)
    start.add_argument("--profiles-file")
    start.add_argument("--max-active-gpu-sessions-per-user", type=int)
    start.add_argument("--max-queued-requests-per-user", type=int)
    start.add_argument("--max-active-sessions-per-pool", type=int)
    start.add_argument("--idle-session-timeout-seconds", type=float, default=3600.0)
    start.add_argument("--reconcile-missing-grace-seconds", type=float, default=5.0)
    start.add_argument("--reconcile-missing-heartbeats", type=int, default=2)
    start.add_argument(
        "--gpu-contention-policy",
        choices=GPU_CONTENTION_POLICIES,
        default="barrier-ratio",
        help=(
            "dual-membership GPU arbitration policy "
            "(default: barrier-ratio)"
        ),
    )
    start.add_argument(
        "--gpu-shared-ratio",
        type=int,
        default=4,
        help=(
            "shared admissions per exclusive turn under contention "
            "(default: 4)"
        ),
    )
    start.add_argument(
        "--gpu-share-override",
        action="append",
        type=_parse_gpu_share_override,
        default=[],
        metavar="GPU_UUID=POLICY[:RATIO]",
        help="repeatable per-physical-GPU policy/ratio override",
    )

    _add_credential_file_options(parser)
    return parser


def _add_credential_file_options(parser):
    actions = {action.dest: action for action in parser._actions}
    if "token" in actions:
        file_action = actions.get("token_file") or parser.add_argument("--token-file")
        token_action = actions["token"]
        if not any(token_action in group._group_actions and file_action in group._group_actions for group in parser._mutually_exclusive_groups):
            group = parser.add_mutually_exclusive_group()
            group._group_actions.extend((token_action, file_action))
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _add_credential_file_options(child)


def cape_main(argv=None, stdout=None, stderr=None):
    parser = _build_cape_parser()
    args = parser.parse_args(argv)
    if args.command == "controller" and args.controller_command == "start" and args.state_dir and not args.token and not args.token_file:
        args.token = os.environ.get("CAPE_ADMIN_TOKEN") or os.environ.get("CAPE_TOKEN")
    # Credentials must not reach workloads that inherit this process environment.
    # Scrub retired aliases too, so an old shell configuration cannot leak secrets.
    for name in (
        "CAPE_TOKEN", "CAPE_JOIN_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_PROBE_NODE_TOKEN", "CAPE_NODE_TOKEN",
        "CAPE_USER_TOKENS_INLINE", "CAPE_SUBMIT_TOKEN", "CAPE_SUBMIT_USER_TOKENS_INLINE",
        "CAPE_STATUS_TOKEN", "CAPE_RETIRE_TOKEN", "CAPE_CONCURRENCY_STATUS_TOKEN",
    ):
        os.environ.pop(name, None)
    if not args.command:
        parser.error("subcommand required")
    try:
        if getattr(args, "token_file", None) and args.command != "controller":
            args.token = read_registration(args.token_file)["secret"]
        if args.command == "user":
            return run_user(args, _request, _controller_url, CapeHTTPError, stdout or sys.stdout, stderr or sys.stderr)
        if args.command == "run":
            return _run_from_args(args, stdout=stdout, no_wait_output="id")
        if args.command == "session":
            if args.session_command in ("list", "status"):
                result = _request("GET", args.controller_url, "/v1/sessions", token=args.token,
                                  query={"session_id": args.session_id} if args.session_command == "status" else {})
            else:
                result = _request("POST", args.controller_url, "/v1/executions/{}/{}".format(args.execution_id, args.session_command),
                                  token=args.token, payload={"force": getattr(args, "force", False)})
            _print_json(result, stdout=stdout)
            return 0
        if args.command == "status":
            return _status_from_args(args, stdout=stdout, stderr=stderr)
        if args.command == "logs":
            return _logs_from_args(args, stdout=stdout, stderr=stderr)
        if args.command == "wait":
            return _wait_from_args(args, stdout=stdout)
        if args.command == "cancel":
            return _cancel_from_args(args, stdout=stdout)
        if args.command == "metric":
            if getattr(args, "local_url", None):
                from .telemetry_control import _location_url
                _location_url(args.local_url)
                query = {key: getattr(args, key) for key in ("node_id", "pool", "gpu_uuid", "metric") if getattr(args, key, None) is not None}
                _print_json(_request("GET", args.local_url, "/snapshot", token=None, query=query), stdout=stdout)
                return 0
            query = {"node_id": args.node_id} if args.node_id else {}
            path = "/v1/metrics/status"
            if args.metric_command != "status":
                path = "/v1/metrics/query"
                query.update(mode=args.metric_command, metric=args.metric)
                query.update({key: getattr(args, key) for key in ("pool", "gpu_uuid", "start", "end", "step") if getattr(args, key, None) is not None})
            result = _request("GET", args.controller_url, path, token=args.token, query=query)
            _print_json(result, stdout=stdout)
            return 0
        if args.command == "utilization":
            return _utilization_from_args(args, stdout=stdout)
        if args.command == "node":
            if args.node_command == "edit":
                changes = {k: v for k, v in (("cpu_cores", args.cpu_cores), ("memory_gb", args.memory_gb)) if v is not None}
                if args.gpu_uuid is not None or args.no_gpus:
                    changes["gpu_uuids"] = [] if args.no_gpus else args.gpu_uuid
                result = _request("POST", args.controller_url, "/v1/identity/nodes/{}/action".format(args.node_id),
                                  token=args.token, payload={"action": "edit", "changes": changes})
                _print_json(result, stdout=stdout)
                return 0
            if args.node_command in ("drain", "exit"):
                result = _request("POST", args.controller_url, "/v1/identity/nodes/{}/action".format(args.node_id),
                                  token=args.token, payload={"action": args.node_command, "force": getattr(args, "force", False)})
                _print_json(result, stdout=stdout)
                return 0
            if args.node_command != "enroll":
                parser.error("node subcommand required: join")
            return _node_enroll_from_args(args, stdout=stdout, stderr=stderr)
        if args.command == "admin":
            if not args.admin_command:
                parser.error("admin subcommand required")
            if args.admin_command in ("pool", "user", "audit", "node", "metric"):
                return run_admin(args, _request, stdout or sys.stdout)
            return _admin_from_args(args, stdout=stdout)
        if args.command == "workspace":
            body = {}
            if args.workspace_action == "set":
                if args.auto_reclaim is None or args.operation_id is not None:
                    raise ValueError("set requires --auto-reclaim and does not accept --operation-id")
                body["auto_reclaim"] = args.auto_reclaim
            elif args.auto_reclaim is not None:
                raise ValueError("--auto-reclaim applies only to set")
            if args.workspace_action in ("grow", "shrink"):
                if args.size_gib is None or not 1 <= args.size_gib <= (2 ** 63 - 1) // 1024 ** 3:
                    raise ValueError("grow/shrink require a positive bounded --size-gib")
                body["target_bytes"] = args.size_gib * 1024 ** 3
            elif args.size_gib is not None:
                raise ValueError("--size-gib applies only to grow/shrink")
            if args.operation_id:
                body["operation_id"] = args.operation_id
            path = "/v1/workspaces/" + args.workspace_id
            if args.workspace_action != "status":
                path += "/" + args.workspace_action
            options = {"token": args.token}
            if body:
                options["payload"] = body
            payload = _request("GET" if args.workspace_action == "status" else "POST",
                               args.controller_url, path, **options)
            print(json.dumps(payload, sort_keys=True), file=stdout)
            return 0
        if args.command == "controller":
            if args.controller_command == "init":
                if os.path.lexists(args.state_dir):
                    raise ValueError("instance directory already exists")
                envelope = prepare_registration(args.token_out, "local:controller-init", args.user, "bootstrap")
                scheduler_id = Instance.initialize(args.state_dir, args.user, envelope["secret"])
                result = {"scheduler_id": scheduler_id, "state_dir": args.state_dir, "token_file": args.token_out}
                if args.metrics:
                    from .identity import IdentityStore
                    from .telemetry_control import set_config
                    from .telemetry_supervisor import PrometheusSupervisor
                    store = IdentityStore(os.path.join(args.state_dir, "state.sqlite3"))
                    try:
                        set_config(store, envelope["secret"], {"enabled": True})
                    finally:
                        store.close()
                    try:
                        result["metrics"] = PrometheusSupervisor(args.state_dir, scheduler_id,
                            os.environ.get("CAPE_PROMETHEUS_BIN", "prometheus")).start()
                    except (OSError, ValueError, RuntimeError):
                        result["metrics"] = {"state": "unavailable", "reason": "prometheus_start_failed"}
                _print_json(result, stdout=stdout)
                return 0
            if args.controller_command != "start":
                parser.error("controller subcommand required: start")
            return _controller_start_from_args(args, stdout=stdout, stderr=stderr)
        parser.error("unknown subcommand")
    except CapeHTTPError as exc:
        _write_error(exc, stderr=stderr)
        return 1
    except WorkspaceError as exc:
        (stderr or sys.stderr).write(str(exc) + "\n")
        return 1
    except (ValueError, PermissionError, OSError) as exc:
        (stderr or sys.stderr).write(str(exc) + "\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cape_main())
