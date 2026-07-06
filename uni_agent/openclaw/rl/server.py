"""External OpenClaw proxy for the verl ``fully_async_policy`` online path."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from uni_agent.gateway.manager import GatewayManager
from uni_agent.gateway.session import TurnCapture
from uni_agent.openclaw import protocol, scorers

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)

# submit_fn(prompt_ids, response_ids, response_logprobs, score, session_id, turn, has_next_state) -> Awaitable
SubmitFn = Callable[..., Awaitable[None]]


class _PendingTurn:
    """A captured single main turn awaiting its next-state PRM signal."""

    __slots__ = ("turn", "prompt_ids", "response_ids", "response_logprobs", "response_text")

    def __init__(self, turn, prompt_ids, response_ids, response_logprobs, response_text):
        self.turn = turn
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_logprobs = response_logprobs
        self.response_text = response_text


class _SessionState:
    """Per-session bookkeeping: the turn counter + the single main turn awaiting next state."""

    def __init__(self):
        self.turn_count = 0
        self.pending: _PendingTurn | None = None


class OpenClawRLServer:
    """FastAPI proxy that turns live OpenClaw client traffic into per-turn samples."""

    def __init__(
        self,
        *,
        submit_fn: SubmitFn,
        is_paused: Callable[[], bool],
        scorer: Optional[scorers.GenericPRMScorer] = None,
        model_name: str = "",
        default_max_tokens: int = 1024,
        host: str = "0.0.0.0",
        port: int = 30000,
        api_key: str = "",
        gateway_manager: GatewayManager,
    ):
        self.submit_fn = submit_fn
        self.is_paused = is_paused
        self._scorer = scorer
        self.model_name = model_name
        self.default_max_tokens = int(default_max_tokens)
        self.host = host
        self.port = int(port)
        self.api_key = api_key
        self.gateway_manager = gateway_manager

        if self._scorer is None:
            logger.warning(
                "[OpenClawRL] no PRM judge configured; all turns score 0 (no learnable "
                "signal). Enable reward.reward_model (GenRM) to activate PRM scoring."
            )

        self._sessions: dict[str, _SessionState] = {}
        self._gateway_sessions: dict[str, SessionHandle] = {}
        self._sessions_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # ------------------------------------------------------------------ app
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw RL Proxy (verl fully_async)")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True, "accepting": not self.is_paused()}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: OpenClawRLServer = request.app.state.owner
            owner._check_auth(authorization)
            if owner.is_paused():
                raise HTTPException(status_code=503, detail="submission paused for weight update")
            if owner.gateway_manager is None:
                raise HTTPException(status_code=503, detail="gateway manager not ready")

            body = await request.json()
            session_id = protocol.parse_session_id(x_session_id, body)
            turn_type = protocol.parse_turn_type(x_turn_type, body)
            session_done = protocol.parse_session_done(x_session_done, body)

            result = await owner._handle_request(body, session_id, turn_type, session_done)
            if bool(body.get("stream", False)):
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        return app

    def _check_auth(self, authorization: Optional[str]) -> None:
        if not self.api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization.split(" ", 1)[1].strip() != self.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # -------------------------------------------------------- request handling
    async def _handle_request(
        self, body: dict[str, Any], session_id: str, turn_type: str, session_done: bool
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        if protocol.is_main_turn(turn_type):
            state = await self._get_or_create_session(session_id)

            # The latest inbound message is the next state for the previous main
            # turn (the user/tool reply that followed the assistant's output).
            if state.pending is not None:
                prev, next_state = state.pending, messages[-1]
                state.pending = None
                self._spawn(self._score_and_submit(session_id, prev, next_state))

            output, capture = await self._forward_main_turn(messages, body, session_id=session_id)
            assistant_msg = output.get("choices", [{}])[0].get("message", {})
            content = assistant_msg.get("content") or ""
            turn_index = state.turn_count
            state.turn_count += 1
            if capture is not None:
                prompt_ids, response_ids, response_logprobs = capture
                state.pending = _PendingTurn(turn_index, prompt_ids, response_ids, response_logprobs, content)
            logger.info(
                "[OpenClawRL] MAIN session=%s turn=%d prompt_msgs=%d resp_len=%d",
                session_id,
                turn_index,
                len(messages),
                len(content),
            )
        else:
            output = await self._forward_side(messages, body, session_id=session_id)
            logger.info("[OpenClawRL] SIDE session=%s -> not captured", session_id)

        if session_done:
            await self._on_session_done(session_id)

        output["session_id"] = session_id
        return {"response": output}

    async def _on_session_done(self, session_id: str) -> None:
        async with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return
        # The final main turn has no next state -> it stays unjudged (score 0, no gradient)
        if state.pending is not None:
            prev = state.pending
            state.pending = None
            self._spawn(self._score_and_submit(session_id, prev, None))
        await self._close_gateway_session(session_id)
        logger.info("[OpenClawRL] session=%s done", session_id)

    async def _score_and_submit(self, session_id: str, pending: _PendingTurn, next_state: dict | None) -> None:
        """PRM-score one main turn against its next state and submit it as a sample."""
        score = 0.0
        if next_state is not None and self._scorer is not None:
            ns_text, ns_role = protocol.next_state_text_role(next_state)
            try:
                result = await self._scorer.evaluate(
                    pending.response_text,
                    ns_text,
                    ns_role,
                    session_id=session_id,
                    turn_num=pending.turn,
                )
                score = float(result.get("score", 0.0))
            except Exception as e:
                logger.warning("[OpenClawRL] PRM scoring failed session=%s turn=%d: %s", session_id, pending.turn, e)

        try:
            await self.submit_fn(
                prompt_ids=pending.prompt_ids,
                response_ids=pending.response_ids,
                response_logprobs=pending.response_logprobs,
                score=score,
                session_id=session_id,
                turn=pending.turn,
                has_next_state=next_state is not None,
            )
        except Exception as e:
            logger.exception("[OpenClawRL] submit failed session=%s turn=%d: %s", session_id, pending.turn, e)
            return
        logger.info(
            "[OpenClawRL] submitted session=%s turn=%d score=%.1f",
            session_id,
            pending.turn,
            score,
        )

    async def _ensure_gateway_session(self, session_id: str) -> SessionHandle:
        if self.gateway_manager is None:
            raise RuntimeError("gateway_manager is not configured")
        handle = self._gateway_sessions.get(session_id)
        if handle is not None:
            return handle
        handle = await self.gateway_manager.create_session(session_id)
        self._gateway_sessions[session_id] = handle
        return handle

    async def _close_gateway_session(self, session_id: str) -> None:
        if self.gateway_manager is None:
            return
        handle = self._gateway_sessions.pop(session_id, None)
        if handle is None:
            return
        try:
            await self.gateway_manager.abort_session(session_id)
        except KeyError:
            pass

    async def _forward_via_gateway(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        capture_turn: bool,
    ) -> tuple[dict[str, Any], TurnCapture | None]:
        if self.gateway_manager is None:
            raise RuntimeError("gateway_manager is not configured")
        await self._ensure_gateway_session(session_id)
        forward_body = self._build_forward_body(messages, body)
        forward_body["openclaw_capture_turn"] = bool(capture_turn)
        output = await self.gateway_manager.chat_completions(session_id=session_id, payload=forward_body)
        if not capture_turn:
            return output, None
        captures = await self.gateway_manager.pop_turn_captures(session_id=session_id)
        if not captures:
            return output, None
        if len(captures) > 1:
            logger.warning("[OpenClawRL] session=%s has %d queued captures; using latest", session_id, len(captures))
        return output, captures[-1]

    # ------------------------------------------------------------ rollout I/O
    async def _get_or_create_session(self, session_id: str) -> _SessionState:
        async with self._sessions_lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            return state

    def _build_forward_body(self, messages: list[dict], body: dict[str, Any]) -> dict[str, Any]:
        forward = protocol.strip_non_standard_keys(body)
        forward["messages"] = messages
        forward["stream"] = False
        forward.pop("stream_options", None)
        if self.model_name:
            forward["model"] = self.model_name
        forward.setdefault("max_tokens", self.default_max_tokens)
        # Request per-token log-probs so we can build training samples.
        forward["logprobs"] = True
        forward.setdefault("top_logprobs", 0)
        return forward

    async def _forward_main_turn(self, messages: list[dict], body: dict[str, Any], *, session_id: str):
        """Forward one main turn and capture token-level data."""
        output, capture = await self._forward_via_gateway(
            session_id=session_id,
            messages=messages,
            body=body,
            capture_turn=True,
        )
        if capture is None:
            return output, None
        return output, (
            list(capture.prompt_ids),
            list(capture.response_ids),
            list(capture.response_logprobs),
        )

    async def _forward_side(self, messages: list[dict], body: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        output, _ = await self._forward_via_gateway(
            session_id=session_id,
            messages=messages,
            body=body,
            capture_turn=False,
        )
        return output

    # --------------------------------------------------------------- tasks
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClawRL] background task failed: %s", exc, exc_info=exc)

    # --------------------------------------------------------------- stream
    async def _stream_response(self, result: dict[str, Any]):
        import json

        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info("[OpenClawRL] proxy listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
