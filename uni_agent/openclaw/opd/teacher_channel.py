"""Teacher / judge channel for OpenClaw OPD on verl ``fully_async_policy``."""

from __future__ import annotations

import logging
from typing import Any, Optional

from uni_agent.openclaw import protocol, scorers
from uni_agent.openclaw.common.http import post_json as _post_json

logger = logging.getLogger(__name__)


def _resolve_url(address: str, path: str) -> str:
    base = address if address.startswith("http") else f"http://{address}"
    return f"{base.rstrip('/')}{path}"


def make_judge_generate_fn(address: str, model_name: str, *, temperature: float, max_tokens: int):
    """Build the judge ``generate_fn(messages) -> str`` against the GenRM chat API."""

    async def _generate(messages: list[dict]) -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        out = await _post_json(_resolve_url(address, "/v1/chat/completions"), payload)
        return (out.get("choices", [{}])[0].get("message", {}) or {}).get("content") or ""

    return _generate


def make_teacher_logprobs_fn(
    address: str,
    model_name: str,
    tokenizer,
    *,
    distill_topk: int = 0,
):
    """Build the teacher ``teacher_fn(hint, turn) -> TeacherSignal`` callable.

    ``TeacherSignal`` is a dict with at least ``teacher_log_probs`` (a per
    response-token list, length ``len(turn.response_ids)``). When
    ``distill_topk > 0`` it also carries ``teacher_topk_log_probs`` and
    ``teacher_topk_indices`` (each ``[T, K]``).
    """
    topk = int(distill_topk or 0)
    use_topk = topk > 0

    async def _teacher_fn(hint: str, turn: protocol.TurnRecord) -> dict[str, Any]:
        response_len = len(turn.response_ids)
        if response_len == 0:
            return {"teacher_log_probs": []}

        # Hint-augmented prompt: append the hint to the original messages and
        # re-apply the chat template, then concatenate the *original* response text
        enhanced_messages = scorers.append_hint_to_messages(turn.messages, hint)
        norm = protocol.normalize_messages(enhanced_messages, normalize_tool_calls=True)
        enhanced_prompt_text = tokenizer.apply_chat_template(
            norm, tools=turn.tools, tokenize=False, add_generation_prompt=True
        )
        enhanced_full_text = enhanced_prompt_text + turn.response_text
        enhanced_ids = tokenizer(enhanced_full_text, add_special_tokens=False)["input_ids"]
        if len(enhanced_ids) <= response_len:
            # Degenerate tokenization; fall back to neutral teacher signal.
            return {"teacher_log_probs": [0.0] * response_len}

        # vLLM accepts token ids directly for /v1/completions; prompt_logprobs
        # returns log p(token_i | token_<i) for every prompt position.
        n_logprobs = topk if use_topk else 0
        payload = {
            "model": model_name,
            "prompt": enhanced_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": n_logprobs,
        }
        out = await _post_json(_resolve_url(address, "/v1/completions"), payload)
        prompt_logprobs = (out.get("choices", [{}])[0] or {}).get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list):
            logger.warning("[OPD-teacher] missing prompt_logprobs; neutral teacher signal")
            return {"teacher_log_probs": [0.0] * response_len}

        # The last ``response_len`` positions correspond to the response tokens.
        resp_entries = prompt_logprobs[-response_len:]
        resp_ids = list(turn.response_ids)

        if not use_topk:
            tlp = [_actual_token_logprob(entry, tid) for entry, tid in zip(resp_entries, resp_ids, strict=False)]
            tlp = protocol.fit_length(tlp, response_len, pad=0.0)
            return {"teacher_log_probs": tlp}

        topk_lp: list[list[float]] = []
        topk_idx: list[list[int]] = []
        single_lp: list[float] = []
        for entry, tid in zip(resp_entries, resp_ids, strict=False):
            lp_row, idx_row = _topk_from_entry(entry, topk)
            topk_lp.append(lp_row)
            topk_idx.append(idx_row)
            single_lp.append(_actual_token_logprob(entry, tid))
        # Pad to response_len rows (defensive; resp_entries should already match).
        while len(topk_lp) < response_len:
            topk_lp.append([0.0] * topk)
            topk_idx.append(list(range(topk)))
            single_lp.append(0.0)
        return {
            "teacher_log_probs": single_lp[:response_len],
            "teacher_topk_log_probs": topk_lp[:response_len],
            "teacher_topk_indices": topk_idx[:response_len],
        }

    return _teacher_fn


def _actual_token_logprob(entry: Optional[dict], token_id: int) -> float:
    """Read the log-prob of ``token_id`` from a vLLM ``prompt_logprobs`` entry.

    Each entry maps ``str(token_id) -> {"logprob": float, ...}``. vLLM always
    includes the actual token, even when it is outside the requested top-K.
    """
    if not isinstance(entry, dict):
        return 0.0
    info = entry.get(str(token_id))
    if isinstance(info, dict) and info.get("logprob") is not None:
        return float(info["logprob"])
    # Fall back to the most probable entry if the actual id is absent.
    best = None
    for v in entry.values():
        lp = v.get("logprob") if isinstance(v, dict) else None
        if lp is not None and (best is None or lp > best):
            best = lp
    return float(best) if best is not None else 0.0


def _topk_from_entry(entry: Optional[dict], k: int) -> tuple[list[float], list[int]]:
    """Extract the top-K ``(logprob, token_id)`` rows from a prompt_logprobs entry."""
    pairs: list[tuple[float, int]] = []
    if isinstance(entry, dict):
        for tid_str, info in entry.items():
            if isinstance(info, dict) and info.get("logprob") is not None:
                try:
                    pairs.append((float(info["logprob"]), int(tid_str)))
                except (TypeError, ValueError):
                    continue
    pairs.sort(key=lambda p: p[0], reverse=True)
    pairs = pairs[:k]
    while len(pairs) < k:
        pairs.append((0.0, 0))
    lp_row = [p[0] for p in pairs]
    idx_row = [p[1] for p in pairs]
    return lp_row, idx_row
