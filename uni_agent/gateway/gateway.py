"""Thin FastAPI/Ray actor layer for routing and session ownership.

The actor owns routing, capability gates, and per-session ``GatewaySession``
instances. Provider adapters own wire-to-internal translation and the response
or SSE envelopes returned to clients.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uni_agent.events import (
    SESSION_CLOSED,
    SESSION_OPENED,
    EventContext,
    EventPublisher,
    LocalEventBus,
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

    def __init__(self, config: GatewayActorConfig, backend):
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
        if config.task_metrics_mode != "off":
            self._event_bus = LocalEventBus()
            self._event_source_instance = f"gateway-{uuid4().hex}"
            self._event_publisher = EventPublisher(
                self._event_bus,
                run_id=f"gateway-run-{uuid4().hex}",
                producer_id=self._event_source_instance,
            )
            self._task_metrics_projector = GatewayTaskMetricsProjector(
                self._event_bus,
                source_instance=self._event_source_instance,
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
        if self._event_publisher is None:
            return
        payload: dict[str, Any] = {"revision": revision}
        if close_reason is not None:
            payload["close_reason"] = close_reason
        try:
            self._event_publisher.publish(event_type, payload, context=context)
        except Exception:
            logger.exception("Failed to publish Gateway event %s for session %s", event_type, context.session_id)

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
        self._server_port, self._server_task = await run_uvicorn(self._app, None, self._server_address)
        host = f"[{self._server_address}]" if is_valid_ipv6_address(self._server_address) else self._server_address
        self._server_base_url = f"http://{host}:{self._server_port}"

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
            if self._task_metrics_projector is not None:
                self._task_metrics_projector.close()
            if self._event_bus is not None:
                self._event_bus.close()
            self._event_runtime_closed = True

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
        session = self._sessions.get(session_id)
        if session is None:
            return  # Already finalized or aborted — treat as idempotent.
        await session.abort()
        self._publish_session_event(SESSION_CLOSED, session.event_context, revision=2, close_reason="aborted")
        if self._task_metrics_projector is not None:
            self._task_metrics_projector.discard_session(session_id)
        self._sessions.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Return a snapshot of a live session's state."""
        session = self._get_session(session_id)
        return session.snapshot_state()


GatewayActor = ray.remote(_GatewayActor)
