"""Unit tests for the backend-agnostic OpenClaw scorer/protocol logic."""

from __future__ import annotations

import asyncio

from uni_agent.openclaw import protocol, scorers

# --------------------------------------------------------------------------- parsing


def test_parse_prm_eval_score():
    assert scorers.parse_prm_eval_score("reasoning ... \\boxed{1}") == 1
    assert scorers.parse_prm_eval_score("\\boxed{-1}") == -1
    assert scorers.parse_prm_eval_score("\\boxed{0}") == 0
    # last boxed wins
    assert scorers.parse_prm_eval_score("\\boxed{1} then \\boxed{-1}") == -1
    # out-of-range / missing -> None
    assert scorers.parse_prm_eval_score("\\boxed{5}") is None
    assert scorers.parse_prm_eval_score("no box here") is None


def test_parse_judge_result():
    score, hint = scorers.parse_judge_result("\\boxed{1}\n[HINT_START]do X then Y[HINT_END]")
    assert score == 1
    assert hint == "do X then Y"
    # negative -> no hint required, score kept
    score, hint = scorers.parse_judge_result("\\boxed{-1} nothing useful")
    assert score == -1
    assert hint == ""
    # 0 is not a valid hint-judge score -> None
    score, _ = scorers.parse_judge_result("\\boxed{0}")
    assert score is None


def test_majority_vote_boundaries():
    assert scorers.majority_vote([1, 1, -1]) == 1.0
    assert scorers.majority_vote([-1, -1, 1]) == -1.0
    # tie -> neutral 0.0
    assert scorers.majority_vote([1, -1]) == 0.0
    # all None -> 0.0
    assert scorers.majority_vote([None, None]) == 0.0
    # None ignored
    assert scorers.majority_vote([1, 1, None]) == 1.0


def test_select_best_hint_longest():
    votes = [
        {"score": 1, "hint": "short hint!!"},
        {"score": 1, "hint": "a substantially longer and more detailed hint"},
        {"score": -1, "hint": ""},
        {"score": 1, "hint": "tiny"},  # too short (<= 10) -> ignored
    ]
    best = scorers.select_best_hint(votes)
    assert best is not None
    assert best["hint"] == "a substantially longer and more detailed hint"
    assert scorers.select_best_hint([{"score": -1, "hint": ""}]) is None


def test_select_candidate_hints_dedupe_and_cap():
    votes = [
        {"score": 1, "hint": "candidate alpha hint"},
        {"score": 1, "hint": "candidate alpha hint"},  # dup
        {"score": 1, "hint": "candidate beta longer hint"},
        {"score": 1, "hint": "candidate gamma even longer hint here"},
        {"score": -1, "hint": "ignored"},
    ]
    hints = scorers.select_candidate_hints(votes, max_candidates=2)
    assert len(hints) == 2
    # shortest first
    assert hints[0] == "candidate alpha hint"


def test_append_hint_to_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "please do the thing"},
    ]
    out = scorers.append_hint_to_messages(messages, "remember to use tool X")
    assert "remember to use tool X" in out[-1]["content"]
    # original is untouched (deep copy)
    assert messages[-1]["content"] == "please do the thing"


# --------------------------------------------------------------------------- protocol


def test_parse_control_fields():
    body = {"turn_type": "main", "session_id": "s1", "session_done": "true"}
    assert protocol.parse_turn_type(None, body) == "main"
    assert protocol.is_main_turn(protocol.parse_turn_type(None, body))
    assert protocol.parse_session_id(None, body) == "s1"
    assert protocol.parse_session_done(None, body) is True
    # header overrides body
    assert protocol.parse_turn_type("SIDE", body) == "side"
    assert protocol.parse_session_done("0", {}) is False


def test_strip_non_standard_keys():
    body = {"messages": [], "turn_type": "main", "session_id": "x", "temperature": 0.6}
    forward = protocol.strip_non_standard_keys(body)
    assert "turn_type" not in forward and "session_id" not in forward
    assert forward["temperature"] == 0.6


def test_flatten_content():
    assert protocol.flatten_content("hello") == "hello"
    assert protocol.flatten_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert protocol.flatten_content(None) == ""


def test_extract_logprobs_and_fit():
    choice = {"logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}}
    lps = protocol.extract_logprobs_from_choice(choice)
    assert lps == [-0.1, -0.2]
    assert protocol.fit_length(lps, 4) == [-0.1, -0.2, 0.0, 0.0]
    assert protocol.fit_length(lps, 1) == [-0.1]


def test_extract_tool_calls_qwen():
    text = '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>'
    clean, calls = protocol.extract_tool_calls(text)
    assert clean == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


# --------------------------------------------------------------------------- generic scorers


def test_generic_prm_scorer_majority():
    async def fake_generate(messages):
        return "\\boxed{1}"

    scorer = scorers.GenericPRMScorer(fake_generate, prm_m=3)
    result = asyncio.run(scorer.evaluate("resp", "thanks!", "user"))
    assert result["score"] == 1.0


def test_generic_opd_scorer_accepts_hint():
    async def fake_generate(messages):
        return "\\boxed{1}\n[HINT_START]use the calculator tool[HINT_END]"

    async def fake_teacher(hint, turn):
        return [0.0] * len(turn.response_ids)

    turn = protocol.TurnRecord(
        session_id="s",
        turn_num=1,
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_logprobs=[-0.1, -0.2, -0.3],
    )
    scorer = scorers.GenericOPDScorer(fake_generate, fake_teacher, prm_m=3)
    result = asyncio.run(scorer.evaluate("resp", "redo it", "user", turn))
    assert result["accepted"] is True
    assert result["hint"] == "use the calculator tool"
    assert len(result["teacher_log_probs"]) == 3
