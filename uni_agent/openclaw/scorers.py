"""Backend-agnostic PRM / OPD scorers for OpenClaw optimization.

This module contains the *pure* scoring logic:

- Prompt builders for the PRM eval judge and the hint (hindsight) judge.
- ``\\boxed{...}`` / ``[HINT_START]...[HINT_END]`` parsing + majority vote.
- Hint selection helpers.
- ``Generic{PRM,OPD,Combined}Scorer`` classes that take *injected* async model
  callables so the same algorithm runs on Tinker, Fireworks, or an
  OpenAI-compatible endpoint without duplication.

The injected callables are:

- ``generate_fn(messages: list[dict]) -> str``
      Run the judge/teacher model on a chat prompt and return decoded text.
- ``teacher_logprobs_fn(hint: str, turn: TurnRecord) -> list[float]``
      Return per-response-token teacher log-probs for the hint-augmented prompt
      (length aligned to ``len(turn.response_ids)``). Only needed for OPD.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import logging
import re
from typing import Awaitable, Callable, Optional

from uni_agent.openclaw.protocol import TurnRecord, flatten_content

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)

GenerateFn = Callable[[list[dict]], Awaitable[str]]
TeacherLogprobsFn = Callable[[str, TurnRecord], Awaitable[list[float]]]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_prm_eval_prompt(response_text: str, next_state_text: str, next_state_role: str = "user") -> list[dict]:
    """PRM eval prompt -- used by RL (primary), OPD (eval_mode) and Combine."""
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action -- "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action worked correctly "
        "and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected -- "
        "e.g. the user moves on, says thanks, the environment confirms success, "
        "or a tool returns a successful, non-error result.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
        "incomplete, or unwanted. **Key negative signals include:**\n"
        "  * The user asks the assistant to **redo, retry, or repeat** the same action "
        '("do it again", "try again", "one more time").\n'
        "  * The user requests a **correction or modification** to what the assistant just did "
        '("change X to Y", "no, I meant ...", "not that, ...", "please fix ...").\n'
        "  * The user **rephrases or restates** the same request, implying the assistant "
        "did not understand or execute it correctly.\n"
        "  * The environment returns an **error, failure, or unexpected result** caused "
        "by the assistant's action.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous -- e.g. the user gives an "
        "unrelated follow-up that neither confirms nor denies success, or there is "
        "insufficient information to judge.\n\n"
        "## Important\n"
        "A change request IS negative feedback -- it means the previous output did not "
        "meet the user's need. Do NOT treat it as a neutral new instruction.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_hint_judge_messages(response_text: str, next_state_text: str, next_state_role: str = "user") -> list[dict]:
    """Hint judge prompt -- used by OPD and Combine methods."""
    system = (
        "You are a process reward model used for hindsight hint extraction.\n"
        "You are given:\n"
        "1) The assistant response at turn t.\n"
        "2) The next state at turn t+1, along with its **role**.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user (follow-up, correction, new request, etc.).\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action -- "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output generally means the assistant's "
        "action was appropriate; do NOT treat it as information the assistant "
        "should have already known.\n\n"
        "Your goal is to decide whether the next state reveals useful hindsight information\n"
        "that could have helped improve the assistant response at turn t.\n\n"
        "Output format rules (strict):\n"
        "- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n"
        "- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n"
        "  wrapped between [HINT_START] and [HINT_END].\n"
        "- If decision is \\boxed{-1}, do not provide a hint block.\n"
        "- Hint must be concrete and actionable for improving the previous response."
    )
    user = (
        f"## Assistant response (turn t)\n{response_text}\n\n"
        f"## Next state (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n"
        "Now output your decision and (if positive) the hint in the required format."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_prm_eval_score(text: str) -> Optional[int]:
    """Extract ``\\boxed{N}`` score for PRM eval (N in {+1, -1, 0})."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    return val if val in (1, -1, 0) else None


def parse_judge_result(text: str) -> tuple[Optional[int], str]:
    """Extract ``(score, hint)`` from hint-judge output (score in {+1, -1})."""
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hint_matches = _HINT_RE.findall(text)
    hint = hint_matches[-1].strip() if hint_matches else ""
    return score, hint


def majority_vote(scores: list[Optional[int]]) -> float:
    """Majority-voted score; ties or all-None -> 0.0 (neutral)."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


def select_best_hint(votes: list[dict]) -> Optional[dict]:
    """Select the longest accepted hint (score==1, len>10) from voting results."""
    good = [v for v in votes if v.get("score") == 1 and isinstance(v.get("hint"), str) and len(v["hint"].strip()) > 10]
    return max(good, key=lambda v: len(v["hint"].strip())) if good else None


def select_candidate_hints(votes: list[dict], max_candidates: int = 3) -> list[str]:
    """Top-K hint selection: dedupe accepted hints, shortest first, cap to K.

    Mirrors OpenClaw combine-select's multi-candidate hint collection.
    """
    seen: set[str] = set()
    hints: list[str] = []
    for v in votes:
        if v.get("score") != 1:
            continue
        h = (v.get("hint") or "").strip()
        if len(h) <= 10 or h in seen:
            continue
        seen.add(h)
        hints.append(h)
    hints.sort(key=len)
    return hints[:max_candidates]


def append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    """Append a hindsight hint to the last user message."""
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[user's hint / instruction]\n{hint}"}]
    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1
    content = flatten_content(cloned[target_idx].get("content"))
    suffix = f"\n\n[user's hint / instruction]\n{hint.strip()}"
    cloned[target_idx]["content"] = (content + suffix).strip()
    return cloned


def is_valid_rl_score(score) -> bool:
    """True if a PRM eval score is a usable +/-1 RL signal."""
    return score in (1, -1, 1.0, -1.0)


# ===========================================================================
# Generic scorers (model calls injected)
# ===========================================================================


class GenericPRMScorer:
    """PRM scorer for Binary RL: majority vote over ``m`` judge generations."""

    def __init__(self, generate_fn: GenerateFn, prm_m: int = 3):
        self._generate = generate_fn
        self.m = prm_m

    async def evaluate(
        self,
        response_text: str,
        next_state_text: str,
        next_state_role: str = "user",
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict:
        msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)
        results = await asyncio.gather(*[self._query_once(msgs, i) for i in range(self.m)])
        scores = [r[0] for r in results]
        final = majority_vote(scores)

        representative = ""
        if final != 0.0:
            for s, text in results:
                if s is not None and s == int(final):
                    representative = text
                    break
        votes_display = [s if s is not None else "fail" for s in scores]
        logger.info("[PRM] session=%s turn=%d votes=%s -> score=%.1f", session_id, turn_num, votes_display, final)
        return {"score": final, "votes": votes_display, "representative": representative}

    async def _query_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await self._generate(messages)
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.exception("[PRM] query failed (vote %d): %s", vote_id, e)
            return None, ""


class GenericOPDScorer:
    """Hint judge (+ optional PRM eval) + teacher log-probs for OPD.

    ``force_hint`` is a **debug/smoke-test** affordance: when set, the hint judge
    is bypassed and every turn (with a next state) is accepted using this canned
    hint. The teacher log-prob query against the real GenRM still runs, so the
    full OPD data path (teacher channel -> sample -> distillation loss) is
    genuinely exercised without depending on a strong judge model. Leave empty
    (default) for real training.
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        teacher_logprobs_fn: TeacherLogprobsFn,
        prm_m: int = 3,
        eval_mode: bool = False,
        force_hint: str = "",
    ):
        self._generate = generate_fn
        self._teacher_logprobs = teacher_logprobs_fn
        self.m = prm_m
        self.eval_mode = eval_mode
        self.force_hint = (force_hint or "").strip()

    async def evaluate(
        self,
        response_text: str,
        next_state_text: str,
        next_state_role: str,
        turn: TurnRecord,
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict:
        if self.force_hint:
            teacher_lps = await self._teacher_logprobs(self.force_hint, turn)
            logger.info("[OPD] session=%s turn=%d FORCED hint (debug)", session_id, turn_num)
            return {
                "accepted": True,
                "teacher_log_probs": teacher_lps,
                "hint": self.force_hint,
                "eval_score": None,
                "hint_raw": "",
                "eval_raw": "",
            }

        msgs = build_hint_judge_messages(response_text, next_state_text, next_state_role)
        votes = await asyncio.gather(*[self._query_judge_once(msgs, i) for i in range(self.m)])

        eval_score = None
        eval_raw = ""
        if self.eval_mode:
            eval_msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)
            eval_results = await asyncio.gather(*[self._query_eval_once(eval_msgs, i) for i in range(self.m)])
            eval_scores = [r[0] for r in eval_results]
            eval_score = majority_vote(eval_scores)
            for s, raw in eval_results:
                if s is not None and s == int(eval_score):
                    eval_raw = raw
                    break

        selected = select_best_hint(votes)
        if selected is None:
            return {
                "accepted": False,
                "teacher_log_probs": None,
                "hint": "",
                "eval_score": eval_score,
                "hint_raw": "",
                "eval_raw": eval_raw,
            }

        hint = selected["hint"].strip()
        teacher_lps = await self._teacher_logprobs(hint, turn)
        logger.info("[OPD] session=%s turn=%d accepted hint_len=%d", session_id, turn_num, len(hint))
        return {
            "accepted": True,
            "teacher_log_probs": teacher_lps,
            "hint": hint,
            "eval_score": eval_score,
            "hint_raw": selected.get("raw", ""),
            "eval_raw": eval_raw,
        }

    async def _query_judge_once(self, messages: list[dict], vote_id: int) -> dict:
        try:
            content = await self._generate(messages)
            score, hint = parse_judge_result(content)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": content}
        except Exception as e:
            logger.exception("[OPD] judge query failed (vote %d): %s", vote_id, e)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _query_eval_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await self._generate(messages)
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.exception("[OPD] eval query failed (vote %d): %s", vote_id, e)
            return None, ""


class GenericCombinedScorer:
    """Hint judge + PRM eval + teacher log-probs for the Combine method.

    Always runs both judges so the dispatcher can choose OPD+RL, OPD-only,
    RL-only or nothing.
    """

    def __init__(
        self, generate_fn: GenerateFn, teacher_logprobs_fn: TeacherLogprobsFn, prm_m: int = 3, force_hint: str = ""
    ):
        self._generate = generate_fn
        self._teacher_logprobs = teacher_logprobs_fn
        self.m = prm_m
        self.force_hint = (force_hint or "").strip()

    async def evaluate(
        self,
        response_text: str,
        next_state_text: str,
        next_state_role: str,
        turn: TurnRecord,
        session_id: str = "",
        turn_num: int = 0,
    ) -> dict:
        if self.force_hint:
            eval_msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)
            eval_results = await asyncio.gather(*[self._query_eval_once(eval_msgs, i) for i in range(self.m)])
            eval_scores = [r[0] for r in eval_results]
            eval_score = majority_vote(eval_scores)
            eval_raw = ""
            for s, raw in eval_results:
                if s is not None and s == int(eval_score):
                    eval_raw = raw
                    break

            teacher_lps = await self._teacher_logprobs(self.force_hint, turn)
            logger.info(
                "[Combine] session=%s turn=%d FORCED hint (debug) eval_score=%s", session_id, turn_num, str(eval_score)
            )
            return {
                "accepted": True,
                "teacher_log_probs": teacher_lps,
                "hint": self.force_hint,
                "eval_score": eval_score,
                "hint_raw": "",
                "eval_raw": eval_raw,
            }

        hint_msgs = build_hint_judge_messages(response_text, next_state_text, next_state_role)
        eval_msgs = build_prm_eval_prompt(response_text, next_state_text, next_state_role)

        hint_coros = [self._query_judge_once(hint_msgs, i) for i in range(self.m)]
        eval_coros = [self._query_eval_once(eval_msgs, i) for i in range(self.m)]
        all_results = await asyncio.gather(*hint_coros, *eval_coros)
        votes = list(all_results[: self.m])
        eval_results = list(all_results[self.m :])

        eval_scores = [r[0] for r in eval_results]
        eval_score = majority_vote(eval_scores)
        eval_raw = ""
        for s, raw in eval_results:
            if s is not None and s == int(eval_score):
                eval_raw = raw
                break

        selected = select_best_hint(votes)
        if selected is None:
            return {
                "accepted": False,
                "teacher_log_probs": None,
                "hint": "",
                "eval_score": eval_score,
                "hint_raw": "",
                "eval_raw": eval_raw,
            }

        hint = selected["hint"].strip()
        teacher_lps = await self._teacher_logprobs(hint, turn)
        logger.info(
            "[Combine] session=%s turn=%d accepted hint_len=%d eval_score=%.1f",
            session_id,
            turn_num,
            len(hint),
            eval_score,
        )
        return {
            "accepted": True,
            "teacher_log_probs": teacher_lps,
            "hint": hint,
            "eval_score": eval_score,
            "hint_raw": selected.get("raw", ""),
            "eval_raw": eval_raw,
        }

    async def _query_judge_once(self, messages: list[dict], vote_id: int) -> dict:
        try:
            content = await self._generate(messages)
            score, hint = parse_judge_result(content)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": content}
        except Exception as e:
            logger.exception("[Combine] judge query failed (vote %d): %s", vote_id, e)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _query_eval_once(self, messages: list[dict], vote_id: int) -> tuple[Optional[int], str]:
        try:
            content = await self._generate(messages)
            return parse_prm_eval_score(content), content
        except Exception as e:
            logger.exception("[Combine] eval query failed (vote %d): %s", vote_id, e)
            return None, ""
