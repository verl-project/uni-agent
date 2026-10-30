"""Thin FastAPI/Ray actor layer for routing and session ownership.

The actor owns routing, capability gates, and per-session ``GatewaySession``
instances. Provider adapters own wire-to-internal translation and the response
or SSE envelopes returned to clients.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uni_agent.gateway.adapters.agent_context import agent_context_from_headers
from uni_agent.gateway.adapters.anthropic import (
    anthropic_build_response,
    anthropic_error_body,
    anthropic_stream_response,
    anthropic_to_internal,
)
from uni_agent.gateway.adapters.openai_chat import (
    openai_build_response,
    openai_error_body,
    openai_stream_response,
    openai_to_internal,
)
from uni_agent.gateway.adapters.openai_response import (
    responses_build_response,
    responses_error_body,
    responses_stream_response,
    responses_to_internal,
)
from uni_agent.gateway.adapters.types import AnthropicRequest, MalformedRequestError, OpenAIChatCompletionRequest
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import (
    AgentContext,
    GatewaySession,
    MessageCodec,
    SessionHandle,
    Trajectory,
)
from uni_agent.gateway.session.types import UnsupportedMediaError
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.utils import run_uvicorn

DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS = frozenset({"max_tokens", "stop"})

#: Reserved policy name used in single-policy mode.
SINGLE_POLICY_NAME = "__single__"

logger = logging.getLogger(__name__)


def _normalize_gateway_inputs(
    config: GatewayActorConfig | Mapping[str, GatewayActorConfig],
    backend: Any | Mapping[str, Any],
) -> tuple[Mapping[str, GatewayActorConfig], Mapping[str, Any]]:
    """Normalize single-policy and multi-policy Gateway inputs.

    Single-policy values are wrapped under ``SINGLE_POLICY_NAME``. Multi-policy
    mappings are reused without copying. ``config`` and ``backend`` must use
    the same representation and contain identical policy names.

    Returns:
        A pair of policy-to-config and policy-to-backend mappings.
    """
    configs_are_mapped = isinstance(config, Mapping)
    backends_are_mapped = isinstance(backend, Mapping)
    if configs_are_mapped != backends_are_mapped:
        raise ValueError("config and backend must both be policy mappings or both single values")
    if not configs_are_mapped:
        config = {SINGLE_POLICY_NAME: config}
        backend = {SINGLE_POLICY_NAME: backend}
    if not config or not backend:
        raise ValueError("config and backend must not be empty")
    if set(config) != set(backend):
        raise ValueError("config and backend must have identical policy names")
    return config, backend


def _build_codec(config: GatewayActorConfig) -> MessageCodec:
    """Build a :class:`MessageCodec` from one policy configuration."""
    return MessageCodec(
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


def _validate_sampling_params(sampling_params: dict[str, Any]) -> None:
    max_tokens = sampling_params.get("max_tokens")
    if "max_tokens" in sampling_params and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise MalformedRequestError("max_tokens must be a positive integer")


class _GatewayActor:
    """Gateway actor exposed as ``GatewayActor = ray.remote(_GatewayActor)``.

    Runtime and manager callers invoke public methods with
    ``actor.method.remote(...)``. The actor owns FastAPI routing, provider
    capability gates, response envelopes, and per-session ``GatewaySession``
    instances. A request's session id comes from the URL and its policy from the
    session's creation-time binding.
    """

    def __init__(
        self,
        config: GatewayActorConfig | Mapping[str, GatewayActorConfig],
        backend: Any | Mapping[str, Any],
    ):
        """Create an actor from single-policy values or policy mappings.

        ``config`` and ``backend`` may each be a single-policy value or a
        policy-name mapping. Single-policy values are normalized under
        ``SINGLE_POLICY_NAME``; mappings must contain identical policy names.
        The normalized mappings are retained as actor configuration.
        """
        self._policy_configs, self._policy_clients = _normalize_gateway_inputs(config, backend)
        self._policy_codecs: dict[str, MessageCodec] = {
            policy_name: _build_codec(policy_config) for policy_name, policy_config in self._policy_configs.items()
        }
        self._policy_allowed_request_sampling_param_keys: dict[str, frozenset[str]] = {
            policy_name: frozenset(DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS).union(
                config.allowed_request_sampling_param_keys or ()
            )
            for policy_name, config in self._policy_configs.items()
        }
        self._warned_discarded_request_sampling_param_keys: set[tuple[str, str]] = set()
        self._sessions: dict[str, GatewaySession] = {}
        self._session_policies: dict[str, str] = {}

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port: int | None = None
        self._server_task: asyncio.Task[Any] | None = None
        self._server_base_url: str | None = None
        self._app: FastAPI = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        """Register provider HTTP handlers."""

        def _error_body_for_path(path: str, status_code: int, message: str) -> dict[str, Any]:
            if path.endswith("/v1/messages"):
                return anthropic_error_body(status_code, message)
            if path.endswith("/v1/responses"):
                return responses_error_body(status_code, message)
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

        @self._app.exception_handler(MalformedRequestError)
        @self._app.exception_handler(UnsupportedMediaError)
        async def _bad_request_handler(request: Request, exc: ValueError):
            return JSONResponse(
                status_code=400,
                content=_error_body_for_path(request.url.path, 400, str(exc)),
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
            return await self._handle_openai_chat_completions(
                session_id=session_id, payload=payload, agent_context=agent_context_from_headers(request.headers)
            )

        @self._app.post("/sessions/{session_id}/v1/responses")
        async def _openai_responses(session_id: str, request: Request):
            try:
                payload = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
            return await self._handle_openai_responses(
                session_id=session_id, payload=payload, agent_context=agent_context_from_headers(request.headers)
            )

        @self._app.post("/sessions/{session_id}/v1/messages")
        async def _anthropic_messages(session_id: str, request: Request):
            try:
                payload = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
            return await self._handle_anthropic_messages(
                session_id=session_id, payload=payload, agent_context=agent_context_from_headers(request.headers)
            )

    def _require_started(self) -> None:
        """Raise if the HTTP server has not been started."""
        if self._server_base_url is None:
            raise RuntimeError("GatewayActor.start() must be called before session creation")

    def _get_policy_name(self, session_id: str) -> str:
        """Return a policy name or raise for an unknown session id."""
        policy_name = self._session_policies.get(session_id)
        if policy_name is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return policy_name

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
        agent_context: AgentContext | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Validate an OpenAI Chat Completions payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
        policy_name = self._get_policy_name(session_id)
        allowed_sampling_keys = self._policy_allowed_request_sampling_param_keys[policy_name]
        try:
            internal = openai_to_internal(
                payload,
                base_sampling_params=dict(session.sampling_params),
                allowed_sampling_keys=allowed_sampling_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
            discarded_keys = session.sampling_params.keys() & payload.keys() - allowed_sampling_keys - {
                key for policy, key in self._warned_discarded_request_sampling_param_keys if policy == policy_name
            }
            if discarded_keys:
                logger.warning(
                    "Ignoring request sampling parameters not enabled by "
                    "allowed_request_sampling_param_keys for policy %s: %s",
                    policy_name,
                    ", ".join(sorted(discarded_keys)),
                )
                self._warned_discarded_request_sampling_param_keys.update((policy_name, key) for key in discarded_keys)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if agent_context is not None:
            internal["agent_context"] = agent_context
        outcome = await session.run_generation(internal, self._policy_clients[policy_name])
        model = str(payload.get("model") or "unknown")
        if payload.get("stream") is True:
            return openai_stream_response(outcome, model=model)
        return JSONResponse(openai_build_response(outcome, model=model))

    async def _handle_openai_responses(
        self,
        session_id: str,
        payload: dict[str, Any],
        agent_context: AgentContext | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Validate a Responses API payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
        policy_name = self._get_policy_name(session_id)
        allowed_sampling_keys = self._policy_allowed_request_sampling_param_keys[policy_name]
        try:
            internal = responses_to_internal(
                payload,
                base_sampling_params=dict(session.sampling_params),
                allowed_sampling_keys=allowed_sampling_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if agent_context is not None:
            internal["agent_context"] = agent_context
        outcome = await session.run_generation(internal, self._policy_clients[policy_name])
        model = str(payload.get("model") or "unknown")
        try:
            if payload.get("stream") is True:
                return responses_stream_response(outcome, payload=payload, model=model)
            response = responses_build_response(outcome, payload=payload, model=model)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(response)

    async def _handle_anthropic_messages(
        self,
        session_id: str,
        payload: AnthropicRequest,
        agent_context: AgentContext | None = None,
    ) -> JSONResponse | StreamingResponse:
        """Validate an Anthropic Messages payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
        policy_name = self._get_policy_name(session_id)
        allowed_sampling_keys = self._policy_allowed_request_sampling_param_keys[policy_name]
        try:
            internal = anthropic_to_internal(
                payload,
                base_sampling_params=dict(session.sampling_params),
                allowed_sampling_keys=allowed_sampling_keys,
            )
            _validate_sampling_params(internal["sampling_params"])
            discarded_keys = session.sampling_params.keys() & payload.keys() - allowed_sampling_keys - {
                key for policy, key in self._warned_discarded_request_sampling_param_keys if policy == policy_name
            }
            if discarded_keys:
                logger.warning(
                    "Ignoring request sampling parameters not enabled by "
                    "allowed_request_sampling_param_keys for policy %s: %s",
                    policy_name,
                    ", ".join(sorted(discarded_keys)),
                )
                self._warned_discarded_request_sampling_param_keys.update((policy_name, key) for key in discarded_keys)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if agent_context is not None:
            internal["agent_context"] = agent_context
        outcome = await session.run_generation(internal, self._policy_clients[policy_name])
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
        policy_name: str | None = None,
    ) -> SessionHandle:
        """Create an actor-owned session and return its provider-compatible handle."""
        self._require_started()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")

        if policy_name is None:
            if len(self._policy_configs) != 1:
                raise ValueError("policy_name is required when the actor serves multiple policies")
            policy_name = next(iter(self._policy_configs))
        if policy_name not in self._policy_configs:
            raise ValueError(f"Unknown policy: {policy_name}")

        config = self._policy_configs[policy_name]
        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySession(
            handle=handle,
            codec=self._policy_codecs[policy_name],
            prompt_length=config.prompt_length,
            response_length=config.response_length,
            sampling_params=sampling_params,
            enable_last_assistant_rollback=config.enable_last_assistant_rollback,
            coalesce_reserved_exact_requests=config.coalesce_reserved_exact_requests,
            metadata=metadata,
        )
        self._session_policies[session_id] = policy_name
        return handle

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        """Finalize a session, remove it from the actor, and return its trajectories."""
        session = self._get_session(session_id)
        trajectories = await session.finalize()
        self._sessions.pop(session_id, None)
        self._session_policies.pop(session_id, None)
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        """Abort a session and remove it from the actor if it still exists."""
        session = self._sessions.get(session_id)
        if session is None:
            return  # Already finalized or aborted — treat as idempotent.
        await session.abort()
        self._sessions.pop(session_id, None)
        self._session_policies.pop(session_id, None)

    async def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Return a snapshot of a live session's state."""
        session = self._get_session(session_id)
        return session.snapshot_state()


GatewayActor = ray.remote(_GatewayActor)
