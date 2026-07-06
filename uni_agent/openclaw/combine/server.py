"""OpenClaw Combine proxy (RL + OPD) for verl fully_async_policy.

Per captured main turn (once next state arrives), dispatch exactly one sample:
- hint accepted + eval in {+1,-1} -> combined sample (OPD + RL)
- hint accepted + eval invalid    -> OPD-only sample
- hint rejected + eval in {+1,-1} -> RL-only sample
- otherwise                       -> drop
"""

from __future__ import annotations

import logging
from typing import Optional

from uni_agent.openclaw import protocol, scorers
from uni_agent.openclaw.opd.server import OpenClawOPDServer, _OPDPendingTurn

logger = logging.getLogger(__name__)


class OpenClawCombineServer(OpenClawOPDServer):
    """FastAPI proxy producing combined OPD/RL samples from OpenClaw traffic."""

    def __init__(self, *, combine_scorer: scorers.GenericCombinedScorer, **kwargs):
        super().__init__(opd_scorer=combine_scorer, distill_topk=0, **kwargs)
        self._combine_scorer = combine_scorer

    async def _score_and_submit(  # type: ignore[override]
        self, session_id: str, pending: _OPDPendingTurn, next_state: Optional[dict]
    ) -> None:
        if next_state is None:
            logger.info("[OpenClawCombine] session=%s turn=%d dropped (no next state)", session_id, pending.turn)
            return
        if self._combine_scorer is None:
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
            result = await self._combine_scorer.evaluate(
                pending.response_text,
                ns_text,
                ns_role,
                turn=turn_record,
                session_id=session_id,
                turn_num=pending.turn,
            )
        except Exception as e:
            logger.warning(
                "[OpenClawCombine] scoring failed session=%s turn=%d: %s",
                session_id,
                pending.turn,
                e,
                exc_info=True,
            )
            return

        eval_score = result.get("eval_score")
        has_rl = scorers.is_valid_rl_score(eval_score)
        accepted = bool(result.get("accepted"))

        if not accepted and not has_rl:
            logger.info(
                "[OpenClawCombine] session=%s turn=%d dropped (no hint and invalid eval)",
                session_id,
                pending.turn,
            )
            return

        if accepted:
            teacher = result.get("teacher_log_probs")
            if isinstance(teacher, dict):
                teacher_lp = teacher.get("teacher_log_probs") or []
            else:
                teacher_lp = teacher or []
            score = float(eval_score) if has_rl else 0.0
            kind = "opd+rl" if has_rl else "opd_only"
        else:
            # RL-only fallback: set
            # teacher_log_probs to the rollout log-probs so the combine loss's
            # teacher_adv = teacher_logp - old_logp is only as large as the
            # rollout-vs-train log-prob drift (~0), keeping the reward branch
            # dominant. For an *exact* zero teacher_adv on RL-only rows, set
            # OPENCLAW_COMBINE_ZERO_TEACHER_ON_RL=1 -- the sample carries an
            # rl-only marker (sample_kind) that the loss gates on.
            teacher_lp = list(pending.response_logprobs)
            score = float(eval_score)
            kind = "rl_only"

        try:
            await self.submit_fn(
                prompt_ids=pending.prompt_ids,
                response_ids=pending.response_ids,
                response_logprobs=pending.response_logprobs,
                teacher_log_probs=teacher_lp,
                score=score,
                sample_kind=kind,
                session_id=session_id,
                turn=pending.turn,
            )
        except Exception as e:
            logger.exception(
                "[OpenClawCombine] submit failed session=%s turn=%d: %s",
                session_id,
                pending.turn,
                e,
            )
            return

        logger.info(
            "[OpenClawCombine] submitted %s sample session=%s turn=%d score=%.1f",
            kind,
            session_id,
            pending.turn,
            score,
        )
