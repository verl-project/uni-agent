"""Thin FastAPI/Ray actor layer for gateway routing and JSON serialization."""

from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any
from uuid import uuid4

import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from uni_agent.gateway.codec import MessageCodec
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.protocol import ChatCompletionRequest, ChatCompletionResponse
from uni_agent.gateway.session import GatewaySession
from uni_agent.gateway.types import SessionHandle, Trajectory
from verl.workers.rollout.utils import run_uvicorn


class _GatewayActor:
    """Ray actor implementation exposed as ``GatewayActor = ray.remote(...)``.

    Runtime and manager callers invoke public methods with
    ``actor.method.remote(...)``. The actor owns FastAPI routing, OpenAI
    capability gates, JSON response envelopes, and per-session
    ``GatewaySession`` instances.
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
            base_sampling_params=config.base_sampling_params,
            allowed_request_sampling_param_keys=config.allowed_request_sampling_param_keys,
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
        """Register HTTP handlers for chat completions and session completion."""

        @self._app.exception_handler(HTTPException)
        async def _http_exception_handler(_request: Request, exc: HTTPException):
            if isinstance(exc.detail, str):
                message = exc.detail
            elif isinstance(exc.detail, dict) and "message" in exc.detail:
                message = str(exc.detail["message"])
            else:
                message = str(exc.detail)
            error_type = "invalid_request_error" if 400 <= exc.status_code < 500 else "internal_server_error"
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": message,
                        "type": error_type,
                        "code": None,
                        "param": None,
                    }
                },
            )

        @self._app.post("/sessions/{session_id}/v1/chat/completions")
        async def _chat_completions(session_id: str, request: Request):
            payload = await request.json()
            return await self._handle_chat_completions(session_id=session_id, payload=payload)

        @self._app.post("/sessions/{session_id}/complete")
        async def _complete(session_id: str, request: Request):
            payload = await request.json()
            reward_info = payload.get("reward_info")
            try:
                await self.complete_session(session_id=session_id, reward_info=reward_info)
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

    async def _handle_chat_completions(
        self,
        session_id: str,
        payload: ChatCompletionRequest,
    ) -> JSONResponse:
        """Validate a chat-completion payload and serialize the session outcome."""
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

        if payload.get("stream") is True:
            getLogger("gateway").warning(
                "session=%s stream=true requested; gateway returns non-streaming response",
                session_id,
            )
        n_value = payload.get("n", 1)
        if n_value != 1:
            raise HTTPException(status_code=400, detail=f"n={n_value} is not supported (only n=1)")
        if payload.get("response_format") is not None:
            raise HTTPException(status_code=400, detail="response_format is not supported")
        tool_choice_payload = payload.get("tool_choice")
        if isinstance(tool_choice_payload, dict):
            raise HTTPException(
                status_code=400,
                detail='tool_choice with a specific function is not supported (only "auto" / "none" are supported)',
            )
        if isinstance(tool_choice_payload, str) and tool_choice_payload.lower() == "required":
            raise HTTPException(
                status_code=400,
                detail='tool_choice="required" is not supported (only "auto" / "none" are supported)',
            )

        outcome = await session.run_generation(payload, self._backend)
        response: ChatCompletionResponse = {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model") or "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": outcome.assistant_msg,
                    "finish_reason": outcome.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
            },
        }
        return JSONResponse(response)

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

    async def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> SessionHandle:
        """Create an actor-owned session and return its OpenAI-compatible handle."""
        self._require_started()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} already exists")

        handle = SessionHandle(
            session_id=session_id,
            base_url=f"{self._server_base_url}/sessions/{session_id}/v1",
        )
        self._sessions[session_id] = GatewaySession(
            handle=handle,
            codec=self._codec,
            prompt_length=self._prompt_length,
            response_length=self._response_length,
        )
        return handle

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        """Mark a session complete and attach optional reward metadata."""
        session = self._get_session(session_id)
        await session.complete(reward_info)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        """Wait for a live session, treating already-removed sessions as done."""
        session = self._sessions.get(session_id)
        if session is None:
            # Already finalized or aborted by a concurrent caller — nothing to wait for.
            return
        await session.wait_for_completion(timeout=timeout)

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
