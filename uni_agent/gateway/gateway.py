"""Thin FastAPI/Ray actor layer for routing and session ownership.

The actor owns routing, capability gates, and per-session ``GatewaySession``
instances. Provider adapters own wire-to-internal translation and the response
or SSE envelopes returned to clients.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uni_agent.events import (
    GATEWAY_GLOBAL_FORWARDER_SUBSCRIPTION,
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    GATEWAY_SESSION_STATE_SCOPE,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    DirectAckStatus,
    DirectRayEventBridge,
    DirectSyncStatus,
    EventContext,
    EventPublisher,
    GlobalBusForwarder,
    LocalEventBus,
    Scope,
    StateSnapshot,
    SubscriptionSpec,
)
from uni_agent.gateway.adapters.anthropic import (
    anthropic_build_response,
    anthropic_error_body,
    anthropic_stream_response,
    anthropic_to_internal,
)
from uni_agent.gateway.adapters.openai import (
    openai_build_response,
    openai_error_body,
    openai_stream_response,
    openai_to_internal,
)
from uni_agent.gateway.adapters.types import AnthropicRequest, MalformedRequestError, OpenAIChatCompletionRequest
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import (
    GatewaySession,
    MessageCodec,
    SessionFinalizationResult,
    SessionHandle,
    Trajectory,
)
from uni_agent.metrics import GatewayTaskMetricsProjector
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.utils import run_uvicorn

DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS = frozenset({"max_tokens", "stop"})

logger = logging.getLogger(__name__)


def _validate_sampling_params(sampling_params: dict[str, Any]) -> None:
    max_tokens = sampling_params.get("max_tokens")
    if "max_tokens" in sampling_params and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise MalformedRequestError("max_tokens must be a positive integer")


class _GatewayActor:
    """Ray actor implementation exposed as ``GatewayActor = ray.remote(...)``.

    Runtime and manager callers invoke public methods with
    ``actor.method.remote(...)``. The actor owns FastAPI routing, provider
    capability gates, response envelopes, and per-session ``GatewaySession``
    instances.
    """

    def __init__(
        self,
        config: GatewayActorConfig,
        backend,
        direct_event_target=None,
        global_event_target=None,
        event_run_id: str | None = None,
        event_source_id: str | None = None,
    ):
        """Create an actor with model codec configuration and backend client."""
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._backend = backend
        self._codec = MessageCodec(
            tokenizer=config.tokenizer,
            processor=config.processor,
            vision_info_extractor=config.vision_info_extractor,
            vision_info_extractor_kwargs=config.vision_info_extractor_kwargs,
            tool_parser_name=config.tool_parser_name,
            rollout_backend=config.rollout_backend,
            enable_tool_parser_cache=config.enable_tool_parser_cache,
            hf_model_type=config.hf_model_type,
            apply_chat_template_kwargs=config.apply_chat_template_kwargs,
            mm_processor_kwargs=config.mm_processor_kwargs,
        )
        self._allowed_request_sampling_param_keys = frozenset(DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS).union(
            config.allowed_request_sampling_param_keys or ()
        )
        self._warned_discarded_request_sampling_param_keys: set[str] = set()
        self._prompt_length = config.prompt_length
        self._response_length = config.response_length
        self._enable_last_assistant_rollback = config.enable_last_assistant_rollback
        self._coalesce_reserved_exact_requests = config.coalesce_reserved_exact_requests
        self._sessions: dict[str, GatewaySession] = {}
        self._event_bus: LocalEventBus | None = None
        self._event_source_instance: str | None = None
        self._event_publisher: EventPublisher | None = None
        self._task_metrics_projector: GatewayTaskMetricsProjector | None = None
        self._direct_bridge: DirectRayEventBridge | None = None
        self._direct_resync_task: asyncio.Task | None = None
        self._direct_current_sessions: dict[str, dict[str, Any]] | None = None
        self._direct_terminal_sessions: OrderedDict[str, dict[str, Any]] | None = None
        self._direct_max_terminal_entities = config.direct_state_sync_max_terminal_entities
        self._direct_state_revision = 0
        self._global_forwarder: GlobalBusForwarder | None = None
        if config.task_metrics_mode != "off" or config.direct_state_sync_enabled or config.global_telemetry_enabled:
            self._event_bus = LocalEventBus()
            self._event_source_instance = event_source_id or f"gateway-{uuid4().hex}"
            self._event_publisher = EventPublisher(
                self._event_bus,
                run_id=event_run_id or f"gateway-run-{uuid4().hex}",
                producer_id=self._event_source_instance,
            )
        if config.task_metrics_mode != "off":
            assert self._event_bus is not None
            assert self._event_source_instance is not None
            self._task_metrics_projector = GatewayTaskMetricsProjector(
                self._event_bus,
                source_instance=self._event_source_instance,
            )
        if config.direct_state_sync_enabled:
            if direct_event_target is None:
                raise ValueError("direct_event_target is required when Direct state sync is enabled")
            assert self._event_bus is not None
            assert self._event_publisher is not None
            self._direct_current_sessions = {}
            self._direct_terminal_sessions = OrderedDict()

            def _send_batch(batch: dict[str, Any]) -> dict[str, Any]:
                return ray.get(direct_event_target.receive_direct_events.remote(batch))

            def _install_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
                return ray.get(direct_event_target.install_direct_snapshot.remote(snapshot))

            self._direct_bridge = DirectRayEventBridge(
                self._event_bus,
                SubscriptionSpec(
                    subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
                    event_types=(SESSION_OPENED, SESSION_CLOSED),
                    scope=Scope.DIRECT,
                    delivery=DeliveryMode.STATE_SYNC,
                    max_queue_events=config.direct_state_sync_max_queue_events,
                    max_queue_bytes=config.direct_state_sync_max_queue_bytes,
                ),
                run_id=self._event_publisher.run_id,
                source_epoch=self._event_publisher.producer_epoch,
                send_batch=_send_batch,
                install_snapshot=_install_snapshot,
                max_retries=config.direct_state_sync_max_retries,
                retry_backoff_s=config.direct_state_sync_retry_backoff_s,
            )
        if config.global_telemetry_enabled:
            if global_event_target is None:
                raise ValueError("global_event_target is required when Global telemetry is enabled")
            assert self._event_bus is not None
            assert self._event_publisher is not None

            def _publish_global_batch(batch: dict[str, Any]) -> dict[str, Any]:
                return global_event_target.publish_global_batch.remote(batch).future().result()

            self._global_forwarder = GlobalBusForwarder(
                self._event_bus,
                SubscriptionSpec(
                    subscription_id=GATEWAY_GLOBAL_FORWARDER_SUBSCRIPTION,
                    event_types=config.global_telemetry_event_types,
                    scope=Scope.GLOBAL,
                    delivery=DeliveryMode.INLINE,
                    max_queue_events=config.global_telemetry_max_queue_events,
                    max_queue_bytes=config.global_telemetry_max_queue_bytes,
                ),
                run_id=self._event_publisher.run_id,
                source_id=self._event_source_instance,
                source_epoch=self._event_publisher.producer_epoch,
                send_batch=_publish_global_batch,
                max_batch_events=config.global_telemetry_max_batch_events,
                max_batch_bytes=config.global_telemetry_max_batch_bytes,
                flush_interval_s=config.global_telemetry_flush_interval_s,
                max_retries=config.global_telemetry_max_retries,
                retry_backoff_s=config.global_telemetry_retry_backoff_s,
            )
        self._event_runtime_closed = False
        self._app = FastAPI()
        self._server_port: int | None = None
        self._server_task: asyncio.Task | None = None
        self._server_base_url: str | None = None
        self._register_routes()

    def _register_routes(self) -> None:
        """Register provider HTTP handlers."""

        def _error_body_for_path(path: str, status_code: int, message: str) -> dict[str, Any]:
            if path.endswith("/v1/messages"):
                return anthropic_error_body(status_code, message)
            return openai_error_body(status_code, message)

        @self._app.exception_handler(HTTPException)
        async def _http_exception_handler(request: Request, exc: HTTPException):
            if isinstance(exc.detail, str):
                message = exc.detail
            elif isinstance(exc.detail, dict) and "message" in exc.detail:
                message = str(exc.detail["message"])
            else:
                message = str(exc.detail)
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body_for_path(request.url.path, exc.status_code, message),
            )

        @self._app.exception_handler(Exception)
        async def _unhandled_exception_handler(request: Request, exc: Exception):
            # Any exception that escapes the route handlers (programmer bugs,
            # ungraceful third-party failures, etc.) reaches here. We log the
            # full traceback for diagnosis but return a provider-shaped error
            # body so client SDKs can still parse it.
            logger.exception("Unhandled exception in gateway route %s", request.url.path)
            return JSONResponse(
                status_code=500,
                content=_error_body_for_path(request.url.path, 500, "Internal server error"),
            )

        @self._app.post("/sessions/{session_id}/v1/chat/completions")
        async def _openai_chat_completions(session_id: str, request: Request):
            try:
                payload = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
            return await self._handle_openai_chat_completions(session_id=session_id, payload=payload)

        @self._app.post("/sessions/{session_id}/v1/messages")
        async def _anthropic_messages(session_id: str, request: Request):
            try:
                payload = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
            return await self._handle_anthropic_messages(session_id=session_id, payload=payload)

    def _require_started(self) -> None:
        """Raise if the HTTP server has not been started."""
        if self._server_base_url is None:
            raise RuntimeError("GatewayActor.start() must be called before session creation")

    def _get_session(self, session_id: str) -> GatewaySession:
        """Return a live session or raise for an unknown session id."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    @staticmethod
    def _build_event_context(session_id: str, metadata: dict[str, Any] | None) -> EventContext:
        raw_context = (metadata or {}).get("_event_context")
        if not isinstance(raw_context, dict):
            raw_context = {}
        raw_global_step = raw_context.get("global_step")
        global_step = (
            raw_global_step if isinstance(raw_global_step, int) and not isinstance(raw_global_step, bool) else None
        )
        runner_name = raw_context.get("runner_name")
        return EventContext(
            episode_id=str(raw_context.get("episode_id") or session_id),
            session_id=session_id,
            runner_name=str(runner_name) if runner_name is not None else None,
            global_step=global_step,
        )

    def _publish_session_event(
        self,
        event_type: str,
        context: EventContext,
        *,
        revision: int,
        close_reason: str | None = None,
    ) -> None:
        source_revision = self._record_direct_session_state(context, revision=revision, close_reason=close_reason)
        if self._event_publisher is None:
            return
        payload: dict[str, Any] = {"revision": revision}
        if source_revision is not None:
            payload["source_revision"] = source_revision
        if close_reason is not None:
            payload["close_reason"] = close_reason
        try:
            self._event_publisher.publish(event_type, payload, context=context)
        except Exception:
            logger.exception("Failed to publish Gateway event %s for session %s", event_type, context.session_id)

    def _record_direct_session_state(
        self,
        context: EventContext,
        *,
        revision: int,
        close_reason: str | None,
    ) -> int | None:
        if self._direct_current_sessions is None or self._direct_terminal_sessions is None:
            return None
        session_id = context.session_id
        if session_id is None:
            return None
        self._direct_state_revision += 1
        state = {
            "episode_id": context.episode_id,
            "revision": revision,
            "status": "open" if close_reason is None else "closed",
            "close_reason": close_reason,
        }
        if close_reason is None:
            self._direct_terminal_sessions.pop(session_id, None)
            self._direct_current_sessions[session_id] = state
            return self._direct_state_revision
        self._direct_current_sessions.pop(session_id, None)
        self._direct_terminal_sessions[session_id] = state
        self._direct_terminal_sessions.move_to_end(session_id)
        while len(self._direct_terminal_sessions) > self._direct_max_terminal_entities:
            self._direct_terminal_sessions.popitem(last=False)
        return self._direct_state_revision

    def _build_direct_snapshot(self, watermark: int) -> StateSnapshot:
        assert self._event_source_instance is not None
        assert self._event_publisher is not None
        assert self._direct_current_sessions is not None
        assert self._direct_terminal_sessions is not None
        return StateSnapshot(
            owner=self._event_source_instance,
            source_epoch=self._event_publisher.producer_epoch,
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=self._direct_state_revision,
            watermark=watermark,
            source_health="healthy",
            current_entities=self._direct_current_sessions,
            terminal_entities=self._direct_terminal_sessions,
        )

    async def _resync_direct_state(self) -> None:
        bridge = self._direct_bridge
        if bridge is None:
            return
        _stream_epoch, watermark = bridge.begin_resync()
        snapshot = self._build_direct_snapshot(watermark)
        ack = await asyncio.to_thread(bridge.install_authoritative_snapshot, snapshot)
        if ack.status is not DirectAckStatus.OK:
            raise RuntimeError(f"Direct snapshot was not applied: {ack.status.value}: {ack.reason}")

    async def _maintain_direct_state(self) -> None:
        while self._direct_bridge is not None:
            try:
                if self._direct_bridge.health.status is DirectSyncStatus.STALE:
                    await self._resync_direct_state()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Gateway Direct state resynchronization failed")
            await asyncio.sleep(0.05)

    async def _handle_openai_chat_completions(
        self,
        session_id: str,
        payload: OpenAIChatCompletionRequest,
    ) -> JSONResponse | StreamingResponse:
        """Validate an OpenAI Chat Completions payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

        try:
            internal = openai_to_internal(
                payload,
                base_sampling_params=dict(session.sampling_params),
                allowed_sampling_keys=self._allowed_request_sampling_param_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
            discarded_keys = (
                session.sampling_params.keys()
                & payload.keys()
                - self._allowed_request_sampling_param_keys
                - self._warned_discarded_request_sampling_param_keys
            )
            if discarded_keys:
                logger.warning(
                    "Ignoring request sampling parameters not enabled by allowed_request_sampling_param_keys: %s",
                    ", ".join(sorted(discarded_keys)),
                )
                self._warned_discarded_request_sampling_param_keys.update(discarded_keys)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        outcome = await session.run_generation(internal, self._backend)
        model = str(payload.get("model") or "unknown")
        if payload.get("stream") is True:
            return openai_stream_response(outcome, model=model)
        return JSONResponse(openai_build_response(outcome, model=model))

    async def _handle_anthropic_messages(
        self,
        session_id: str,
        payload: AnthropicRequest,
    ) -> JSONResponse | StreamingResponse:
        """Validate an Anthropic Messages payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

        try:
            internal = anthropic_to_internal(
                payload,
                base_sampling_params=dict(session.sampling_params),
                allowed_sampling_keys=self._allowed_request_sampling_param_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
            discarded_keys = (
                session.sampling_params.keys()
                & payload.keys()
                - self._allowed_request_sampling_param_keys
                - self._warned_discarded_request_sampling_param_keys
            )
            if discarded_keys:
                logger.warning(
                    "Ignoring request sampling parameters not enabled by allowed_request_sampling_param_keys: %s",
                    ", ".join(sorted(discarded_keys)),
                )
                self._warned_discarded_request_sampling_param_keys.update(discarded_keys)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        outcome = await session.run_generation(internal, self._backend)
        model = str(payload.get("model") or "unknown")
        if payload.get("stream") is True:
            return anthropic_stream_response(outcome, model=model)
        return JSONResponse(anthropic_build_response(outcome, model=model))

    async def start(self) -> None:
        """Start the FastAPI server backing this gateway actor."""
        if self._server_task is not None:
            return
        if self._direct_bridge is not None:
            await self._resync_direct_state()
        self._server_port, self._server_task = await run_uvicorn(self._app, None, self._server_address)
        host = f"[{self._server_address}]" if is_valid_ipv6_address(self._server_address) else self._server_address
        self._server_base_url = f"http://{host}:{self._server_port}"
        if self._direct_bridge is not None:
            self._direct_resync_task = asyncio.create_task(self._maintain_direct_state())

    async def shutdown(self) -> None:
        """Stop the FastAPI server backing this gateway actor."""
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
            self._server_port = None
            self._server_base_url = None
        if not self._event_runtime_closed:
            if self._direct_resync_task is not None:
                self._direct_resync_task.cancel()
                try:
                    await self._direct_resync_task
                except asyncio.CancelledError:
                    pass
                self._direct_resync_task = None
            if self._direct_bridge is not None:
                await asyncio.to_thread(self._direct_bridge.close)
                self._direct_bridge = None
            if self._global_forwarder is not None:
                await asyncio.to_thread(self._global_forwarder.close)
                self._global_forwarder = None
            if self._task_metrics_projector is not None:
                self._task_metrics_projector.close()
            if self._event_bus is not None:
                self._event_bus.close()
            self._event_runtime_closed = True

    async def get_direct_sync_status(self) -> dict[str, Any]:
        """Expose source-side Direct health for readiness and diagnostics."""
        if self._direct_bridge is None:
            return {"enabled": False}
        health = self._direct_bridge.health
        return {
            "enabled": True,
            "status": health.status.value,
            "stream_epoch": health.stream_epoch,
            "contiguous_cursor": health.contiguous_cursor,
            "last_assigned_seq": health.last_assigned_seq,
            "pending_events": health.pending_events,
            "pending_bytes": health.pending_bytes,
            "retries": health.retries,
            "resyncs": health.resyncs,
            "overflow_count": health.overflow_count,
            "stale_reason": health.stale_reason,
        }

    async def get_global_telemetry_status(self) -> dict[str, Any]:
        """Expose source-side Global queue, retry, and loss state."""
        if self._global_forwarder is None:
            return {"enabled": False}
        health = self._global_forwarder.health
        return {
            "enabled": True,
            "status": health.status.value,
            "pending_events": health.pending_events,
            "pending_bytes": health.pending_bytes,
            "pending_calls": health.pending_calls,
            "accepted_batches": health.accepted_batches,
            "accepted_events": health.accepted_events,
            "retries": health.retries,
            "dropped_events": health.dropped_events,
            "incomplete_batches": health.incomplete_batches,
            "last_error": health.last_error,
        }

    async def create_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> SessionHandle:
        """Create an actor-owned session and return its provider-compatible handle."""
        self._require_started()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")

        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        event_context = self._build_event_context(session_id, metadata)
        self._sessions[session_id] = GatewaySession(
            handle=handle,
            codec=self._codec,
            prompt_length=self._prompt_length,
            response_length=self._response_length,
            sampling_params=sampling_params,
            enable_last_assistant_rollback=self._enable_last_assistant_rollback,
            coalesce_reserved_exact_requests=self._coalesce_reserved_exact_requests,
            metadata=metadata,
            event_publisher=self._event_publisher,
            event_context=event_context,
        )
        self._publish_session_event(SESSION_OPENED, event_context, revision=1)
        return handle

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        """Finalize a session, remove it from the actor, and return its trajectories."""
        return (await self.finalize_session_result(session_id)).trajectories

    async def finalize_session_result(self, session_id: str) -> SessionFinalizationResult:
        """Finalize a session and return trajectories plus its metrics fragment."""
        session = self._get_session(session_id)
        trajectories = await session.finalize()
        self._publish_session_event(SESSION_CLOSED, session.event_context, revision=2, close_reason="finalized")
        metrics_fragment = (
            self._task_metrics_projector.finalize_session(session_id)
            if self._task_metrics_projector is not None
            else None
        )
        self._sessions.pop(session_id, None)
        return SessionFinalizationResult(
            trajectories=trajectories,
            metrics_fragment=metrics_fragment,
        )

    async def abort_session(self, session_id: str) -> None:
        """Abort a session and remove it from the actor if it still exists."""
        await self.abort_session_result(session_id)

    async def abort_session_result(self, session_id: str) -> SessionFinalizationResult:
        """Abort a session while preserving any metrics observed before failure."""
        session = self._sessions.get(session_id)
        if session is None:
            return SessionFinalizationResult(trajectories=[])
        await session.abort()
        self._publish_session_event(SESSION_CLOSED, session.event_context, revision=2, close_reason="aborted")
        metrics_fragment = (
            self._task_metrics_projector.finalize_session(session_id)
            if self._task_metrics_projector is not None
            else None
        )
        self._sessions.pop(session_id, None)
        return SessionFinalizationResult(trajectories=[], metrics_fragment=metrics_fragment)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Return a snapshot of a live session's state."""
        session = self._get_session(session_id)
        return session.snapshot_state()


GatewayActor = ray.remote(_GatewayActor)
