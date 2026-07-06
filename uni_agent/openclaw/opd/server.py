"""OpenClaw OPD proxy for the verl ``fully_async_policy`` online path."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException

from uni_agent.openclaw import protocol, scorers
from uni_agent.openclaw.rl.server import OpenClawRLServer, _PendingTurn

logger = logging.getLogger(__name__)

# submit_fn(*, prompt_ids, response_ids, response_logprobs, teacher_log_probs,
#           teacher_topk_log_probs, teacher_topk_indices, session_id, turn,
#           eval_score) -> Awaitable[None]
OPDSubmitFn = Callable[..., Awaitable[None]]


class _OPDPendingTurn(_PendingTurn):
    """A captured main turn awaiting its next-state hint judge + teacher query."""

    __slots__ = ("messages", "tools")

    def __init__(self, turn, prompt_ids, response_ids, response_logprobs, response_text, messages, tools):
        super().__init__(turn, prompt_ids, response_ids, response_logprobs, response_text)
        self.messages = messages
        self.tools = tools


class OpenClawOPDServer(OpenClawRLServer):
    """FastAPI proxy turning live OpenClaw traffic into per-turn OPD samples."""

    def __init__(
        self,
        *,
        opd_scorer: scorers.GenericOPDScorer,
        distill_topk: int = 0,
        **kwargs,
    ):
        # The base stores ``opd_scorer`` as ``self._scorer`` so the inherited
        # warning/None handling still applies; the OPD evaluate() signature
        # differs, so we keep a typed handle too.
        super().__init__(scorer=opd_scorer, **kwargs)
        self._opd_scorer = opd_scorer
        self.distill_topk = int(distill_topk or 0)

    async def _handle_request(
        self, body: dict[str, Any], session_id: str, turn_type: str, session_done: bool
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        if protocol.is_main_turn(turn_type):
            state = await self._get_or_create_session(session_id)

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
                state.pending = _OPDPendingTurn(
                    turn_index,
                    prompt_ids,
                    response_ids,
                    response_logprobs,
                    content,
                    messages=list(messages),
                    tools=body.get("tools"),
                )
            logger.info(
                "[OpenClawOPD] MAIN session=%s turn=%d prompt_msgs=%d resp_len=%d",
                session_id,
                turn_index,
                len(messages),
                len(content),
            )
        else:
            output = await self._forward_side(messages, body, session_id=session_id)
            logger.info("[OpenClawOPD] SIDE session=%s -> not captured", session_id)

        if session_done:
            await self._on_session_done(session_id)

        output["session_id"] = session_id
        return {"response": output}

    async def _score_and_submit(  # type: ignore[override]
        self, session_id: str, pending: _OPDPendingTurn, next_state: Optional[dict]
    ) -> None:
        """Judge the turn against its next state; submit an OPD sample if accepted."""
        if next_state is None:
            # Final/unjudged turn -- no hindsight available, drop.
            logger.info("[OpenClawOPD] session=%s turn=%d dropped (no next state)", session_id, pending.turn)
            return
        if self._opd_scorer is None:
            return

        ns_text, ns_role = protocol.next_state_text_role(next_state)
        turn_record = protocol.TurnRecord(
            session_id=session_id,
            turn_num=pending.turn,
            prompt_ids=pending.prompt_ids,
            response_ids=pending.response_ids,
            response_logprobs=pending.response_logprobs,
            response_text=pending.response_text,
            messages=pending.messages,
            tools=pending.tools,
            has_next_state=True,
        )
        try:
            result = await self._opd_scorer.evaluate(
                pending.response_text,
                ns_text,
                ns_role,
                turn=turn_record,
                session_id=session_id,
                turn_num=pending.turn,
            )
        except Exception as e:
            logger.warning(
                "[OpenClawOPD] OPD scoring failed session=%s turn=%d: %s", session_id, pending.turn, e, exc_info=True
            )
            return

        if not result.get("accepted"):
            logger.info("[OpenClawOPD] session=%s turn=%d dropped (no accepted hint)", session_id, pending.turn)
            return

        teacher = result.get("teacher_log_probs")
        # teacher_logprobs_fn returns a dict payload (see opd.teacher_channel).
        if isinstance(teacher, dict):
            teacher_lp = teacher.get("teacher_log_probs") or []
            topk_lp = teacher.get("teacher_topk_log_probs")
            topk_idx = teacher.get("teacher_topk_indices")
        else:
            teacher_lp = teacher or []
            topk_lp = None
            topk_idx = None

        try:
            await self.submit_fn(
                prompt_ids=pending.prompt_ids,
                response_ids=pending.response_ids,
                response_logprobs=pending.response_logprobs,
                teacher_log_probs=teacher_lp,
                teacher_topk_log_probs=topk_lp,
                teacher_topk_indices=topk_idx,
                session_id=session_id,
                turn=pending.turn,
                eval_score=result.get("eval_score"),
            )
        except Exception as e:
            logger.exception("[OpenClawOPD] submit failed session=%s turn=%d: %s", session_id, pending.turn, e)
            return
        logger.info(
            "[OpenClawOPD] submitted OPD sample session=%s turn=%d hint_len=%d",
            session_id,
            pending.turn,
            len(result.get("hint", "")),
        )
