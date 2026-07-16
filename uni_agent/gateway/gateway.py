"""Thin FastAPI/Ray actor layer for routing and session ownership.

The actor owns routing, capability gates, and per-session ``GatewaySession``
instances. Provider adapters own wire-to-internal translation and the response
or SSE envelopes returned to clients.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    SessionHandle,
    Trajectory,
)
from verl.workers.rollout.utils import run_uvicorn

DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})

logger = logging.getLogger("gateway")


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
        # Same pattern as vllm_async_server.py / async_sglang_server.py:
        # use the node's routable IP for both bind and URL.
        self._server_address = ray.util.get_node_ip_address()
        self._backend = backend
        self._codec = MessageCodec(
            tokenizer=config.tokenizer,
            processor=config.processor,
            vision_info_extractor=config.vision_info_extractor,
            vision_info_extractor_kwargs=config.vision_info_extractor_kwargs,
            tool_parser_name=config.tool_parser_name,
            apply_chat_template_kwargs=config.apply_chat_template_kwargs,
        )
        self._base_sampling_params = dict(config.base_sampling_params or {})
        self._allowed_request_sampling_param_keys = (
            DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS
            if config.allowed_request_sampling_param_keys is None
            else frozenset(config.allowed_request_sampling_param_keys)
        )
        self._prompt_length = config.prompt_length
        self._response_length = config.response_length
        self._sessions: dict[str, GatewaySession] = {}
        self._app = FastAPI()
        self._server_port: int | None = None
        self._server_task: asyncio.Task | None = None
        self._server_base_url: str | None = None
        self._register_routes()

    def _register_routes(self) -> None:
        """Register provider HTTP handlers and reward metadata."""

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

        @self._app.post("/sessions/{session_id}/reward_info")
        async def _reward_info(session_id: str, request: Request):
            payload = await request.json()
            reward_info = payload.get("reward_info")
            try:
                await self.set_reward_info(session_id=session_id, reward_info=reward_info)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return JSONResponse({"status": "ok"})

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
                base_sampling_params={**self._base_sampling_params, **session.sampling_params},
                allowed_sampling_keys=self._allowed_request_sampling_param_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
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
                base_sampling_params={**self._base_sampling_params, **session.sampling_params},
                allowed_sampling_keys=self._allowed_request_sampling_param_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
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
        self._server_base_url = f"http://{self._server_address}:{self._server_port}"

    async def shutdown(self) -> None:
        """Stop the FastAPI server backing this gateway actor."""
        if self._server_task is None:
            return
        self._server_task.cancel()
        try:
            await self._server_task
        except asyncio.CancelledError:
            pass
        self._server_task = None
        self._server_port = None
        self._server_base_url = None

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
            reward_info_url=f"{self._server_base_url}/sessions/{session_id}/reward_info",
        )
        self._sessions[session_id] = GatewaySession(
            handle=handle,
            codec=self._codec,
            prompt_length=self._prompt_length,
            response_length=self._response_length,
            sampling_params=sampling_params,
        )
        return handle

    async def set_reward_info(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        """Attach optional reward metadata to a live session."""
        session = self._get_session(session_id)
        await session.set_reward_info(reward_info)

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        """Finalize a session, remove it from the actor, and return its trajectories."""
        session = self._get_session(session_id)
        trajectories = await session.finalize()
        self._sessions.pop(session_id, None)
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        """Abort a session and remove it from the actor if it still exists."""
        session = self._sessions.get(session_id)
        if session is None:
            return  # Already finalized or aborted — treat as idempotent.
        await session.abort()
        self._sessions.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Return a snapshot of a live session's state."""
        session = self._get_session(session_id)
        return session.snapshot_state()


GatewayActor = ray.remote(_GatewayActor)
