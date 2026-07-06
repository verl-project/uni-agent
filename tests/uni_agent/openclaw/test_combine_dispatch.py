"""Unit tests for the OpenClaw Combine proxy's four-way sample dispatch.

`OpenClawCombineServer._score_and_submit` must turn one judged main turn into
exactly one of: OPD+RL, OPD-only, RL-only, or a drop. This exercises that
dispatch with a stub scorer + stub submit_fn (no rollout/teacher network).
"""

import asyncio

import pytest

from uni_agent.openclaw.combine.server import OpenClawCombineServer
from uni_agent.openclaw.opd.server import _OPDPendingTurn


class _StubScorer:
    def __init__(self, result):
        self.result = result

    async def evaluate(self, response_text, ns_text, ns_role, *, turn, session_id, turn_num):
        return self.result


def _make_server(scorer, captured):
    async def _submit(**kwargs):
        captured.append(kwargs)

    return OpenClawCombineServer(
        combine_scorer=scorer,
        submit_fn=_submit,
        is_paused=lambda: False,
        model_name="stub",
        gateway_manager=object(),
    )


def _pending():
    return _OPDPendingTurn(
        turn=0,
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_logprobs=[-0.11, -0.22],
        response_text="ans",
        messages=[{"role": "user", "content": "q"}],
        tools=None,
    )


def _run(scorer, next_state):
    captured = []
    server = _make_server(scorer, captured)
    asyncio.run(server._score_and_submit("sess", _pending(), next_state))
    return captured


NEXT_STATE = {"role": "user", "content": "corrective feedback"}


def test_dispatch_opd_plus_rl():
    scorer = _StubScorer(
        {"accepted": True, "eval_score": 1, "teacher_log_probs": {"teacher_log_probs": [-0.5, -0.6]}, "hint": "h"}
    )
    captured = _run(scorer, NEXT_STATE)
    assert len(captured) == 1
    kw = captured[0]
    assert kw["sample_kind"] == "opd+rl"
    assert kw["score"] == 1.0
    assert kw["teacher_log_probs"] == [-0.5, -0.6]


def test_dispatch_opd_only():
    scorer = _StubScorer({"accepted": True, "eval_score": None, "teacher_log_probs": [-0.5, -0.6], "hint": "h"})
    captured = _run(scorer, NEXT_STATE)
    assert len(captured) == 1
    kw = captured[0]
    assert kw["sample_kind"] == "opd_only"
    assert kw["score"] == 0.0
    assert kw["teacher_log_probs"] == [-0.5, -0.6]


def test_dispatch_rl_only():
    scorer = _StubScorer({"accepted": False, "eval_score": -1})
    captured = _run(scorer, NEXT_STATE)
    assert len(captured) == 1
    kw = captured[0]
    assert kw["sample_kind"] == "rl_only"
    assert kw["score"] == -1.0
    # RL-only carries rollout log-probs as teacher signal (teacher_adv ~= 0)
    assert kw["teacher_log_probs"] == pytest.approx([-0.11, -0.22], abs=1e-6)


def test_dispatch_drop_no_hint_invalid_eval():
    scorer = _StubScorer({"accepted": False, "eval_score": None})
    captured = _run(scorer, NEXT_STATE)
    assert captured == []


def test_dispatch_drop_no_next_state():
    scorer = _StubScorer({"accepted": True, "eval_score": 1, "teacher_log_probs": [-0.5, -0.6]})
    captured = _run(scorer, None)
    assert captured == []
