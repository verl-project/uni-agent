# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import functools
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

from .controller import CAPEController, PoolConfigConflictError
from .models import EnforcementMode, MountSpec, NodeResources, NodeState, ResourceRequest, RuntimeCapabilities, WorkloadSpec


TERMINAL_REQUEST_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "EXPIRED",
    "LOST",
    "INFEASIBLE",
}


@web.middleware
async def _json_error_middleware(request, handler):
    try:
        return await handler(request)
    except web.HTTPException as exc:
        return web.json_response({"error": exc.reason}, status=exc.status)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except PermissionError as exc:
        return web.json_response({"error": str(exc)}, status=403)
    except PoolConfigConflictError as exc:
        return web.json_response(exc.payload, status=400)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


class CAPEControllerService(object):
    def __init__(self, controller=None, host="127.0.0.1", port=0, schedule_interval=0.05):
        self.controller = controller or CAPEController()
        self.host = host
        self.port = port
        self.schedule_interval = schedule_interval
        self.app = self._build_app()
        self._bound_host = host
        self._bound_port = port
        self._loop = None
        self._loop_thread = None
        self._runner = None
        self._site = None
        self._scheduler_task = None
        self._scheduler_error = None
        self._executor = None
        self._ready = threading.Event()
        self._start_error = None
        self._closing = False
        self._known_request_states = {}
        self._request_waiters = {}
        self._registration_attempts = deque()

    def _build_app(self):
        app = web.Application(middlewares=[_json_error_middleware])
        app.add_routes([
            web.get("/health", self._http_health),
            web.get("/v1/management/capabilities", self._http_management_capabilities),
            web.post("/v1/management/{action}", self._http_management),
            web.post("/v1/identity/tokens", self._http_identity_register),
            web.get("/v1/identity/me", self._http_identity_me),
            web.get("/v1/identity/pools", self._http_identity_pools),
            web.get("/v1/identity/tokens", self._http_identity_tokens),
            web.post("/v1/identity/tokens/{token_id}/revoke", self._http_identity_revoke),
            web.post("/v1/identity/admin", self._http_identity_admin),
            web.get("/v1/metrics/status", self._http_metric_status),
            web.get("/v1/metrics/query", self._http_metric_query),
            web.get("/v1/metrics/service-config", self._http_metric_service_config),
            web.post("/v1/agent/metrics/control", self._http_metric_control),
            web.post("/v1/identity/nodes/enroll", self._http_identity_enroll),
            web.get("/v1/identity/nodes", self._http_identity_nodes),
            web.post("/v1/identity/nodes/{node_id}/action", self._http_identity_node_action),
            web.post("/v1/submit", self._http_submit),
            web.get("/v1/requests/{request_id}", self._http_request_status),
            web.get("/v1/requests/{request_id}/logs", self._http_request_logs),
            web.get("/v1/requests/{request_id}/wait", self._http_request_wait),
            web.post("/v1/requests/{request_id}/cancel", self._http_request_cancel),
            web.get("/v1/pools/{pool}/utilization", self._http_pool_utilization),
            web.post("/v1/agent/announce", self._http_agent_announce),
            web.post("/v1/agent/register", self._http_agent_register),
            web.post("/v1/agent/heartbeat", self._http_agent_heartbeat),
            web.post("/v1/agent/report-command", self._http_agent_report_command),
            web.post("/v1/admin/drain-node", self._http_admin_drain_node),
            web.post("/v1/admin/drain-allocation", self._http_admin_drain_allocation),
            web.post("/v1/admin/remove-allocation", self._http_admin_remove_allocation),
            web.post("/v1/admin/set-pool-config", self._http_admin_set_pool_config),
            web.get("/v1/admin/workspace-config", self._http_admin_get_workspace_config),
            web.post("/v1/admin/workspace-config", self._http_admin_set_workspace_config),
            web.get("/v1/workspaces/{workspace_id}", self._http_workspace_status),
            web.post("/v1/workspaces/{workspace_id}/{action}", self._http_workspace_operation),
            web.get("/v1/sessions", self._http_logical_sessions),
            web.post("/v1/executions/{execution_id}/{action}", self._http_execution_action),
        ])
        return app

    @property
    def url(self):
        return "http://{}:{}".format(self._bound_host, self._bound_port)

    def start(self):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return self
        self._closing = False
        self._ready.clear()
        self._start_error = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cape-controller")
        self._loop_thread = threading.Thread(target=self._run_event_loop, name="cape-aiohttp")
        self._loop_thread.daemon = True
        self._loop_thread.start()
        if not self._ready.wait(10.0):
            raise RuntimeError("timed out starting CAPE aiohttp service")
        if self._start_error is not None:
            error = self._start_error
            self._shutdown_executor()
            raise RuntimeError("failed to start CAPE aiohttp service: {}".format(error))
        return self

    def stop(self):
        loop = self._loop
        if loop is None:
            self._shutdown_executor()
            return
        try:
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._async_stop(), loop)
                future.result(timeout=10.0)
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(10.0)
            self._shutdown_executor()
            self._loop = None

    def _shutdown_executor(self):
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _run_event_loop(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
        else:
            self._ready.set()
            loop.run_forever()
        finally:
            pending = list(asyncio.all_tasks(loop))
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    async def _async_start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        addresses = list(self._runner.addresses)
        if not addresses:
            raise RuntimeError("aiohttp site did not expose a listening address")
        self._bound_host = addresses[0][0]
        self._bound_port = addresses[0][1]
        self._known_request_states = await self._run_blocking(self._snapshot_request_states)
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def _async_stop(self):
        if self._closing:
            return
        self._closing = True
        self._notify_waiters()
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
        if self._runner is not None:
            await self._runner.cleanup()
        await self._run_blocking(self._journal_flush)

    async def _scheduler_loop(self):
        while True:
            try:
                states = await self._run_blocking(self._scheduler_tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._scheduler_error = exc
            else:
                self._scheduler_error = None
                self._publish_request_states(states)
            await asyncio.sleep(max(0.0, self.schedule_interval))

    def _scheduler_tick(self):
        with self._journal_batch():
            self.controller.schedule_once()
            self.controller.check_heartbeats()
            self.controller.expire_idle_sessions()
            return self._snapshot_request_states()

    def _snapshot_request_states(self):
        states = {}
        for request_id, record in self.controller.requests.items():
            state = record.state.value if hasattr(record.state, "value") else str(record.state)
            states[request_id] = state
        return states

    async def _run_blocking(self, function, *args, **kwargs):
        if self._executor is None:
            raise RuntimeError("CAPE controller executor is not running")
        call = functools.partial(function, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    async def _mutate(self, function):
        def invoke():
            with self._journal_batch():
                result = function()
                return result, self._snapshot_request_states()

        result, states = await self._run_blocking(invoke)
        self._publish_request_states(states)
        return result

    def _journal_batch(self):
        batch = getattr(self.controller.journal, "batch", None)
        if callable(batch):
            return batch()
        return contextlib.nullcontext()

    def _journal_flush(self):
        flush = getattr(self.controller.journal, "flush", None)
        if callable(flush):
            flush()

    def _publish_request_states(self, states):
        previous = self._known_request_states
        self._known_request_states = dict(states)
        changed = set(previous).union(states)
        for request_id in changed:
            if previous.get(request_id) != states.get(request_id):
                self._notify_waiters(request_id)

    def _notify_waiters(self, request_id=None):
        if request_id is None:
            groups = list(self._request_waiters.values())
        else:
            groups = [self._request_waiters.get(request_id, ())]
        for group in groups:
            for event in list(group):
                event.set()

    async def _wait_for_request(
        self, request_id, timeout, token, running_timeout=None
    ):
        if running_timeout is not None and running_timeout < 0:
            raise ValueError("running_timeout must be non-negative")
        event = asyncio.Event()
        waiters = self._request_waiters.setdefault(request_id, set())
        waiters.add(event)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        running_deadline = None
        try:
            while True:
                status = await self._run_blocking(self.controller.request_status, request_id, token=token)
                if status["state"] in TERMINAL_REQUEST_STATES or self._closing:
                    return status
                if (
                    running_timeout is not None
                    and running_deadline is None
                    and status.get("state") == "RUNNING"
                    and status.get("started_at") is not None
                ):
                    elapsed = max(
                        0.0,
                        self.controller.clock()
                        - float(status["started_at"]),
                    )
                    running_deadline = loop.time() + max(
                        0.0, running_timeout - elapsed
                    )
                effective_deadline = deadline
                timeout_kind = "overall"
                if (
                    running_deadline is not None
                    and running_deadline <= effective_deadline
                ):
                    effective_deadline = running_deadline
                    timeout_kind = "running"
                remaining = effective_deadline - loop.time()
                if remaining <= 0:
                    payload = dict(status)
                    payload["wait_timed_out"] = True
                    payload["wait_timeout_kind"] = timeout_kind
                    return payload
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
                event.clear()
        finally:
            waiters.discard(event)
            if not waiters:
                self._request_waiters.pop(request_id, None)

    async def _http_health(self, request):
        return web.json_response({"ok": True})

    async def _http_management_capabilities(self, request):
        from .management import capabilities
        return web.json_response(capabilities(self.controller))

    async def _http_management(self, request):
        from .management import call
        body = await _read_json_body(request)
        return web.json_response(await self._mutate(lambda: call(
            self.controller, request.match_info["action"], _extract_token(request.headers), body)))

    def _identity_store(self):
        if not getattr(self.controller.auth, "managed", False):
            raise ValueError("persistent identity is not configured")
        return self.controller.auth.store

    async def _http_metric_query(self, request):
        from .telemetry_query import prepare_query, fetch_query
        def prepare():
            store = self._identity_store()
            store.authenticate(_extract_token(request.headers))
            return prepare_query(store.snapshot(), dict(request.query), time.time())
        plan = await self._run_blocking(prepare)
        # Network I/O never occupies the single-writer mutation executor.
        return web.json_response(await fetch_query(plan, getattr(self.controller, "telemetry_query_url", None)))

    async def _http_metric_service_config(self, request):
        from .telemetry_control import public_config
        if set(request.query) != {"service_id"}:
            raise ValueError("service_id required")
        result = await self._run_blocking(lambda: public_config(self._identity_store(), request.query["service_id"]))
        result["receiver_url"] = getattr(self.controller, "telemetry_receiver_url", None)
        return web.json_response(result)

    async def _http_metric_status(self, request):
        from .telemetry_control import get_config
        if set(request.query) - {"node_id"}:
            raise ValueError("unknown metrics status parameter")
        return web.json_response(await self._run_blocking(lambda: get_config(
            self._identity_store(), _extract_token(request.headers), request.query.get("node_id"))))

    async def _http_metric_control(self, request):
        from .telemetry_control import poll_control
        body = await _read_json_body(request)
        result = await self._run_blocking(lambda: poll_control(self._identity_store(), _extract_token(request.headers), body))
        result["receiver_url"] = getattr(self.controller, "telemetry_receiver_url", None)
        return web.json_response(result)

    async def _http_identity_register(self, request):
        body = await _read_json_body(request)
        allowed = {"operation_id", "verifier", "name", "username", "expires_at", "user_id"}
        if set(body) - allowed:
            raise ValueError("unknown registration parameter")
        def register():
            if _extract_token(request.headers) is None:
                now = time.monotonic()
                while self._registration_attempts and self._registration_attempts[0] <= now - 60:
                    self._registration_attempts.popleft()
                if len(self._registration_attempts) >= 64:
                    raise web.HTTPTooManyRequests(reason="anonymous registration rate limit exceeded")
                self._registration_attempts.append(now)
            return self._identity_store().register_token(
                body["operation_id"], body["verifier"], body["name"], body.get("username"),
                bearer=_extract_token(request.headers), expires_at=body.get("expires_at"), target_user_id=body.get("user_id"))
        return web.json_response(await self._run_blocking(register))

    async def _http_identity_me(self, request):
        def identify():
            store = self._identity_store()
            actor = store.authenticate(_extract_token(request.headers))
            state = store.snapshot()
            return dict(actor, scheduler_id=state["scheduler_id"],
                        permissions={g: state["groups"][g] for g in actor["groups"] if g in state["groups"]})
        return web.json_response(await self._run_blocking(identify))

    async def _http_identity_enroll(self, request):
        from .identity_nodes import enroll
        body = await _read_json_body(request)
        return web.json_response(await self._run_blocking(
            lambda: enroll(self._identity_store(), _extract_token(request.headers), body)))

    async def _http_identity_node_action(self, request):
        from .managed_lifecycle import node_action, edit_node
        body = await _read_json_body(request)
        if not isinstance(body, dict) or set(body) - {"action", "force", "changes"}:
            raise ValueError("invalid node action")
        self._identity_store()
        if body.get("action") == "edit":
            if body.get("force", False):
                raise ValueError("force applies to drain, not edit")
            return web.json_response(await self._mutate(lambda: edit_node(
                self.controller, _extract_token(request.headers), request.match_info["node_id"], body.get("changes"))))
        return web.json_response(await self._mutate(lambda: node_action(
            self.controller, _extract_token(request.headers), request.match_info["node_id"], body.get("action"), body.get("force", False))))

    async def _http_identity_nodes(self, request):
        def inventory():
            store = self._identity_store()
            store.authenticate(_extract_token(request.headers))
            pool_id = self.controller.auth.pool_id(request.query["pool"]) if request.query.get("pool") else None
            result = []
            for key, declared in store.snapshot().get("nodes", {}).items():
                if declared["desired"] == "removed" or (pool_id and declared["pool"] != pool_id):
                    continue
                node = self.controller.nodes.get(key)
                allowed = {"index", "uuid", "name", "timestamp", "gpu_utilization_percent", "memory_utilization_percent",
                           "memory_total_bytes", "memory_used_bytes", "memory_free_bytes", "temperature_c"}
                metrics = getattr(node, "last_status", {}).get("gpu_metrics", []) if node else []
                metrics = [{k: v for k, v in metric.items() if k in allowed} for metric in metrics
                           if isinstance(metric, dict) and metric.get("uuid") in declared.get("effective_resources", declared["resources"])["gpu_uuids"]] if isinstance(metrics, list) else []
                result.append({"node_id": key, "pool_id": declared["pool"], "resources": declared.get("effective_resources", declared["resources"]),
                               "state": node.state.value if node else "enrolled", "gpu_metrics": metrics})
            return result
        return web.json_response(await self._run_blocking(inventory))

    async def _http_identity_tokens(self, request):
        return web.json_response(await self._run_blocking(
            lambda: self._identity_store().list_tokens(_extract_token(request.headers))))

    async def _http_identity_pools(self, request):
        def inventory():
            store = self._identity_store()
            store.authenticate(_extract_token(request.headers))
            return [{"pool_id": key, "name": pool["name"], "config": pool.get("config", {"pool_mode": "exclusive", "gpu_slots": 1, "placement": "pack"}),
                     "max_nodes": pool.get("max_nodes"),
                     "state": "draining" if pool["draining"] else ("available" if any(
                         node.pool == key and node.state == NodeState.ACTIVE for node in self.controller.nodes.values()) else "unavailable")}
                    for key, pool in store.snapshot()["pools"].items()]
        return web.json_response(await self._run_blocking(inventory))

    async def _http_identity_revoke(self, request):
        def revoke():
            self._identity_store().revoke_token(_extract_token(request.headers), request.match_info["token_id"])
            return {"revoked": True}
        return web.json_response(await self._run_blocking(revoke))

    async def _http_identity_admin(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)
        def manage():
            store = self._identity_store()
            store.require_admin(token)
            action = body.get("action")
            if action in ("metric-get", "metric-set"):
                from .telemetry_control import get_config, set_config
                allowed = {"action", "node_id"} | ({"changes", "expected_version"} if action == "metric-set" else set())
                if set(body) - allowed:
                    raise ValueError("unknown telemetry parameter")
                if action == "metric-get":
                    return get_config(store, token, body.get("node_id"), admin=True)
                return set_config(store, token, body.get("changes"), body.get("node_id"), body.get("expected_version"))
            if action == "node-release-quarantine":
                if set(body) != {"action", "node_id", "reason", "confirm_stopped"} or not isinstance(body["node_id"], str):
                    raise ValueError("invalid quarantine release parameters")
                from .node_purge import release_quarantine
                return release_quarantine(self.controller, token, body["node_id"], body["reason"], body["confirm_stopped"])
            if action == "node-purge":
                if set(body) != {"action", "node_id", "reason"} or not isinstance(body["node_id"], str):
                    raise ValueError("invalid node purge parameters")
                from .node_purge import purge
                return purge(self.controller, token, body["node_id"], body["reason"])
            if action == "audit":
                return store.audit_events(token, body.get("limit", 100))
            if action == "user-list":
                return [dict(user_id=key, **value) for key, value in store.snapshot()["users"].items()]
            if action == "group-list":
                return store.snapshot()["groups"]
            if action in ("pool-drain", "pool-remove"):
                from .managed_lifecycle import pool_action
                return pool_action(self.controller, token, self.controller.auth.pool_id(body["pool"]),
                                   action.split("-")[1], body.get("force", False))
            if action == "pool-create":
                return {"pool_id": store.create_pool(token, body["name"], body.get("config"), body.get("max_nodes"))}
            if action == "pool-edit":
                from .managed_lifecycle import edit_pool
                return edit_pool(self.controller, token, self.controller.auth.pool_id(body["pool"]), body["changes"])
            if action == "group-set":
                pools = body.get("pools", [])
                if not isinstance(pools, list) or any(not isinstance(pool, str) for pool in pools):
                    raise ValueError("invalid pool IDs")
                store.set_group(token, body["name"], body.get("all_pools", False), [self.controller.auth.pool_id(pool) for pool in pools])
            elif action == "group-delete":
                store.delete_group(token, body["name"])
            elif action == "user-edit":
                store.edit_user(token, body["user_id"], body.get("groups"), body.get("disabled"))
            else:
                raise ValueError("unknown identity management action")
            return {"updated": True}
        return web.json_response(await self._mutate(manage))

    async def _http_submit(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def submit():
            if getattr(self.controller.auth, "managed", False):
                actor = self._identity_store().authenticate(token)
                if body.get("user_id") not in (None, "", actor["user_id"], actor["name"]):
                    raise PermissionError("cannot submit for another identity")
                body["user_id"] = actor["user_id"]
                body["pool"] = self.controller.auth.pool_id(body["pool"])
            record = self.controller.submit_workload(workload_spec_from_dict(body), token=token)
            return self.controller.request_status(record.request_id, token=token)

        return web.json_response(await self._mutate(submit), status=202)

    async def _http_request_status(self, request):
        payload = await self._run_blocking(
            self.controller.request_status,
            request.match_info["request_id"],
            token=_extract_token(request.headers),
        )
        return web.json_response(payload)

    async def _http_request_logs(self, request):
        payload = await self._run_blocking(
            self.controller.logs_for_request,
            request.match_info["request_id"],
            token=_extract_token(request.headers),
        )
        return web.json_response(payload)

    async def _http_request_wait(self, request):
        timeout = float(request.query.get("timeout", 30))
        running_timeout = request.query.get("running_timeout")
        if running_timeout is not None:
            running_timeout = float(running_timeout)
        payload = await self._wait_for_request(
            request.match_info["request_id"],
            timeout,
            _extract_token(request.headers),
            running_timeout=running_timeout,
        )
        return web.json_response(payload)

    async def _http_request_cancel(self, request):
        body = await _read_json_body(request)
        request_id = request.match_info["request_id"]
        token = _extract_token(request.headers)

        def cancel():
            record = self.controller.cancel_request(
                request_id,
                reason=body.get("reason", "cancelled"),
                token=token,
            )
            return self.controller.request_status(record.request_id, token=token)

        return web.json_response(await self._mutate(cancel))

    async def _http_pool_utilization(self, request):
        if getattr(self.controller.auth, "managed", False):
            await self._run_blocking(lambda: self._identity_store().authenticate(_extract_token(request.headers)))
        payload = await self._run_blocking(
            self.controller.pool_utilization,
            pool=request.match_info["pool"],
        )
        if getattr(self.controller.auth, "managed", False):
            for node in payload.get("node_states", []):
                node.pop("active_sessions", None)
        return web.json_response(payload)

    async def _http_agent_register(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def register():
            legacy_capabilities = body.get("capabilities")
            runtime_capabilities = {
                adapter: runtime_capabilities_from_dict(capabilities)
                for adapter, capabilities in body.get(
                    "runtime_capabilities", {}
                ).items()
            }
            node = self.controller.register_node(
                node_id=body["node_id"],
                pool=body["pool"],
                resources=node_resources_from_dict(body.get("resources", {})),
                capabilities=(
                    runtime_capabilities_from_dict(legacy_capabilities)
                    if legacy_capabilities is not None
                    else None
                ),
                runtime_capabilities=runtime_capabilities,
                allocation_id=body.get("allocation_id"),
                allocation_deadline=body.get("allocation_deadline"),
                drain_before_sec=body.get("drain_before_sec", 0),
                cleanup_buffer_sec=body.get("cleanup_buffer_sec", 0),
                runtime_cache_config=body.get("runtime_cache_config", "default"),
                cached_images=tuple(body.get("cached_images", ())),
                join_token=body.get("join_token"),
                agent_token=token,
                agent_epoch=body.get("agent_epoch"),
                recovering=bool(body.get("recovering", False)),
                workspace_status=body.get("workspace_config") or {"protocol": 0, "state": "unavailable", "applied_version": None},
                pool_mode=body.get("pool_mode", "exclusive"),
                gpu_slots=body.get("gpu_slots", 1),
                placement=body.get("placement", "pack"),
            )
            payload = self.controller.node_status(node.node_id, token=token)
            control = self.controller.agent_control(node.node_id)
            if control is not None:
                payload["agent_control"] = control
            return payload

        return web.json_response(await self._mutate(register))

    async def _http_agent_announce(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def announce():
            node = self.controller.register_node(
                node_id=body["node_id"],
                pool=body["pool"],
                resources=node_resources_from_dict(
                    body.get("resources", {})
                ),
                allocation_id=body.get("allocation_id"),
                allocation_deadline=body.get("allocation_deadline"),
                drain_before_sec=body.get("drain_before_sec", 0),
                cleanup_buffer_sec=body.get("cleanup_buffer_sec", 0),
                runtime_cache_config=body.get(
                    "runtime_cache_config", "default"
                ),
                cached_images=tuple(body.get("cached_images", ())),
                join_token=body.get("join_token"),
                agent_token=token,
                agent_epoch=body.get("agent_epoch"),
                state=NodeState.REGISTERING,
                workspace_status={"protocol": body.get("workspace_protocol", 0), "state": "pending", "applied_version": None},
                pool_mode=body.get("pool_mode", "exclusive"),
                gpu_slots=body.get("gpu_slots", 1),
                placement=body.get("placement", "pack"),
            )
            payload = self.controller.node_status(node.node_id, token=token)
            control = self.controller.agent_control(node.node_id)
            if control is not None:
                payload["agent_control"] = control
            return payload

        return web.json_response(await self._mutate(announce))

    async def _http_agent_heartbeat(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def heartbeat():
            payload = self.controller.node_heartbeat(
                body["node_id"],
                state=NodeState(body["state"]) if body.get("state") else None,
                status=body.get("status", {}),
                token=token,
                max_commands=body.get("max_commands", 8),
                agent_epoch=body.get("agent_epoch"),
            )
            metrics = body.get("status", {}).get("metrics")
            if metrics is not None and getattr(self.controller.auth, "managed", False):
                from .telemetry_control import poll_control
                try:
                    control = poll_control(self._identity_store(), token, {
                        "node_id": body["node_id"], "agent_epoch": body.get("agent_epoch"),
                        "service_id": metrics["service_id"], "applied_version": metrics.get("applied_version", 0),
                        "state": metrics.get("state", "pending")})
                    control["receiver_url"] = getattr(self.controller, "telemetry_receiver_url", None)
                    payload["metrics_control"] = control
                except (ValueError, PermissionError, TypeError, KeyError):
                    payload["metrics_control"] = {"state": "error", "reason": "invalid_metrics_control"}
            return payload

        payload = await self._mutate(heartbeat)
        if getattr(self.controller.auth, "managed", False) or payload.get("workspace_operations") or body.get("status", {}).get("workspace_updates") or any(item.get("session_spec", {}).get("workspace", {}).get("create") for item in payload.get("commands", ()) if item.get("action") == "run"):
            await self._run_blocking(self._journal_flush)
        return web.json_response(payload)

    async def _http_agent_report_command(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def report():
            payload = self.controller.report_command_result(
                body["node_id"],
                body["command_id"],
                exit_code=body.get("exit_code", 0),
                stdout=body.get("stdout", ""),
                stderr=body.get("stderr", ""),
                failure_kind=body.get("failure_kind", "workload"),
                log_path=body.get("log_path"),
                artifact_paths=tuple(body.get("artifact_paths", ())),
                token=token,
                agent_epoch=body.get("agent_epoch"),
            )
            payload["workspace_acks"] = self.controller.apply_workspace_updates(
                body["node_id"], body.get("workspace_updates", []), token, body.get("agent_epoch"),
            )
            node = self.controller.nodes[body["node_id"]]
            payload["workspace_config"] = self.controller.workspace_configs.get(node.pool, node.node_id).to_dict()
            self.controller.schedule_once()
            control = self.controller.agent_control(body["node_id"])
            payload["commands"] = []
            if body.get("accept_commands") is True:
                node = self.controller.nodes[body["node_id"]]
                payload["commands"] = (
                    self.controller.dequeue_node_response_commands(
                        body["node_id"],
                        max_commands=body.get("max_commands", 8),
                        allow_assignments=(
                            control is None
                            and node.state == NodeState.ACTIVE
                        ),
                    )
                )
            if control is not None:
                payload["agent_control"] = control
            return payload

        payload = await self._mutate(report)
        if getattr(self.controller.auth, "managed", False) or payload.get("workspace_operations") or body.get("workspace_updates") or any(item.get("session_spec", {}).get("workspace", {}).get("create") for item in payload.get("commands", ()) if item.get("action") == "run"):
            await self._run_blocking(self._journal_flush)
        return web.json_response(payload)

    async def _http_workspace_status(self, request):
        payload = await self._run_blocking(self.controller.workspace_status, request.match_info["workspace_id"],
                                           token=_extract_token(request.headers))
        if payload.get("operation"):
            await self._run_blocking(self._journal_flush)
        return web.json_response(payload)

    async def _http_logical_sessions(self, request):
        from .session_api import list_sessions
        result = await self._run_blocking(list_sessions, self.controller, _extract_token(request.headers),
                                          request.query.get("session_id"))
        return web.json_response(result)

    async def _http_execution_action(self, request):
        from .session_api import execution_action
        body = await _read_json_body(request) if request.can_read_body else {}
        if not isinstance(body, dict) or set(body) - {"force"} or type(body.get("force", False)) is not bool:
            raise ValueError("invalid execution action body")
        result = await self._mutate(lambda: execution_action(self.controller, request.match_info["execution_id"],
            request.match_info["action"], _extract_token(request.headers), body.get("force", False)))
        await self._run_blocking(self._journal_flush)
        return web.json_response(result)

    async def _http_workspace_operation(self, request):
        body = await _read_json_body(request) if request.can_read_body else {}
        allowed = {"auto_reclaim"} if request.match_info["action"] == "set" else {"target_bytes", "operation_id"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ValueError("invalid workspace operation body")
        payload = await self._mutate(lambda: self.controller.workspace_operation(
            request.match_info["workspace_id"], request.match_info["action"], token=_extract_token(request.headers),
            target_bytes=body.get("target_bytes"), operation_id=body.get("operation_id"), auto_reclaim=body.get("auto_reclaim")))
        await self._run_blocking(self._journal_flush)
        return web.json_response(payload)

    async def _http_admin_drain_node(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def drain():
            node = self.controller.drain_node(
                body["node_id"],
                reason=body.get("reason", "operator_drain"),
                token=token,
            )
            if getattr(self.controller.auth, "managed", False):
                return {"node_id": node.node_id, "state": node.state.value}
            return self.controller.node_status(node.node_id, token=token)

        return web.json_response(await self._mutate(drain))

    async def _http_admin_drain_allocation(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def drain():
            allocation = self.controller.drain_allocation(
                body["allocation_id"],
                reason=body.get("reason", "allocation_drain"),
                token=token,
            )
            return self.controller.allocation_status(allocation.allocation_id)

        return web.json_response(await self._mutate(drain))

    async def _http_admin_remove_allocation(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def remove():
            allocation = self.controller.remove_allocation(
                body["allocation_id"],
                force=body.get("force", False),
                token=token,
            )
            return self.controller.allocation_status(allocation.allocation_id)

        return web.json_response(await self._mutate(remove))

    async def _http_admin_get_workspace_config(self, request):
        payload = await self._run_blocking(
            self.controller.get_workspace_config, request.query.get("pool", ""),
            node_id=request.query.get("node_id"), token=_extract_token(request.headers),
        )
        return web.json_response(payload)

    async def _http_admin_set_workspace_config(self, request):
        body = await _read_json_body(request)
        if not isinstance(body, dict):
            raise ValueError("workspace configuration request must be an object")
        payload = await self._mutate(lambda: self.controller.set_workspace_config(
            body.get("pool", ""), body.get("changes"), node_id=body.get("node_id"),
            token=_extract_token(request.headers),
        ))
        await self._run_blocking(self._journal_flush)
        return web.json_response(payload)

    async def _http_admin_set_pool_config(self, request):
        body = await _read_json_body(request)
        token = _extract_token(request.headers)

        def set_config():
            self.controller.set_pool_config(
                body["pool"],
                pool_mode=body["pool_mode"],
                gpu_slots=body["gpu_slots"],
                placement=body["placement"],
                token=token,
            )
            return self.controller.pool_utilization(body["pool"])

        return web.json_response(await self._mutate(set_config))


async def _read_json_body(request):
    raw = await request.text()
    return json.loads(raw) if raw else {}


def _extract_token(headers):
    auth = headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1]
    return headers.get("X-CAPE-Token")


def workload_spec_from_dict(data):
    resources = data.get("resources", {})
    if "gpu_mode" in resources:
        raise ValueError(
            "request resource gpu_mode is no longer supported; "
            "select an exclusive or shared pool"
        )
    session_id = data.get("session_id")
    if any(key in data for key in ("task_id", "session_key", "workspace_id")):
        raise ValueError("obsolete workload identity fields; use session_id and workspace.id")
    mounts = []
    for item in data.get("mounts", ()):
        mounts.append(MountSpec(item["source"], item["target"], item.get("read_only", False)))
    return WorkloadSpec(
        pool=data["pool"],
        user_id=data["user_id"],
        session_id=session_id,
        command=tuple(data["command"]),
        resources=ResourceRequest(
            gpus=resources.get("gpus", 0),
            cpu_cores=resources.get("cpu_cores", 1),
            memory_gb=resources.get("memory_gb", 1),
            max_duration_seconds=resources.get("max_duration_seconds", 3600),
            command_timeout_seconds=resources.get("command_timeout_seconds"),
            hard_gpu_enforcement=resources.get("hard_gpu_enforcement", False),
            runtime_adapter=resources.get("runtime_adapter"),
            ports=tuple(resources.get("ports", ())),
            locality=resources.get("locality"),
            isolation_policy=resources.get("isolation_policy", "default"),
        ),
        cwd=data.get("cwd"),
        env=data.get("env", {}),
        image=data.get("image"),
        profile=data.get("profile"),
        mounts=tuple(mounts),
        reuse=data.get("reuse", True),
        keep_session_on_failure=data.get("keep_session_on_failure", True),
        priority=data.get("priority", 0),
        privileged=data.get("privileged", False),
        log_path=data.get("log_path"),
        artifact_paths=tuple(data.get("artifact_paths", ())),
        workspace=data.get("workspace"),
    )


def node_resources_from_dict(data):
    return NodeResources(data.get("cpu_cores", 0), data.get("memory_gb", 0), tuple(data.get("gpu_uuids", ())))


def runtime_capabilities_from_dict(data):
    mode = data.get("gpu_enforcement", EnforcementMode.HARD.value)
    return RuntimeCapabilities(
        adapter=data.get("adapter", "fake"),
        image_formats=tuple(data.get("image_formats", ("default", "docker", "sif", "oci"))),
        isolation_policies=tuple(data.get("isolation_policies", ("default",))),
        nested_container=data.get("nested_container", True),
        can_bind_mounts=data.get("can_bind_mounts", True),
        can_select_gpus=data.get("can_select_gpus", True),
        gpu_enforcement=EnforcementMode(mode),
        can_enforce_cpu_memory=data.get("can_enforce_cpu_memory", True),
        node_local_cache=data.get("node_local_cache", True),
        supports_warm_sessions=data.get("supports_warm_sessions", True),
        privileged_allowed=data.get("privileged_allowed", False),
        gpu_isolation_policies=(
            tuple(data["gpu_isolation_policies"])
            if "gpu_isolation_policies" in data
            else None
        ),
        reduced_isolation_policies=tuple(
            data.get("reduced_isolation_policies", ())
        ),
    )
