"""Reward evaluator for the multi-agent blackbox example."""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

_EXPECTED_KEYS = (
    "target",
    "expected",
    "expected_text",
    "expected_answer",
    "expected_answers",
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "label",
    "solution",
    "ground_truth",
    "ground_truth_raw",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _keep_numeric_punctuation(text: str, index: int) -> bool:
    """Keep punctuation that carries numeric meaning: 3.5, -6, 1/2, 2016-12-21."""
    ch = text[index]
    previous = text[index - 1] if index > 0 else ""
    following = text[index + 1] if index + 1 < len(text) else ""

    if ch == ".":
        # Preserve decimals such as 3.5 and .5, but not a sentence-final dot.
        return following.isdigit() and (previous.isdigit() or not previous or not previous.isalnum())
    if ch == "-":
        # Preserve a leading minus sign or a hyphen between numeric fields.
        return following.isdigit() and (previous.isdigit() or not previous or not previous.isalnum())
    if ch == "/":
        # Preserve fractions and date/slash forms such as 1/2 and 2017/18.
        return previous.isdigit() and following.isdigit()
    return False


def normalize_answer(value: Any) -> str:
    """Normalize an answer before exact matching.

    Mirrors the Search-R1 / reference Search MAS normalization (case,
    articles, punctuation, whitespace) and additionally folds Unicode accents
    so ``café`` matches
    ``cafe``. Punctuation inside numeric tokens is preserved so ``3.5`` does
    not collapse to ``35`` and ``-6`` does not collapse to ``6``.
    """
    text = unicodedata.normalize("NFKD", _text(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    # Preserve single-token answers such as multiple-choice "A"; only strip
    # articles when they are part of a longer answer.
    if len(text.split()) > 1:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(
        ch
        for index, ch in enumerate(text)
        if not unicodedata.category(ch).startswith("P") or _keep_numeric_punctuation(text, index)
    )
    return " ".join(text.split())


def extract_answer(value: Any) -> str | None:
    """Return the last ``<answer>...</answer>`` payload, if present."""
    text = _text(value)
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _answer_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        candidates: list[str] = []
        for key in _EXPECTED_KEYS:
            if key in value:
                candidates.extend(_answer_candidates(value[key]))
        return candidates
    if isinstance(value, list | tuple | set):
        candidates = []
        for item in value:
            candidates.extend(_answer_candidates(item))
        return candidates
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(parsed, list | tuple | set):
                    return _answer_candidates(parsed)
        return [stripped] if stripped else []
    return [str(value).strip()]


def _dedupe(candidates: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        # Keep punctuation-only answers ("—", "(") that normalize to an empty
        # string; they are valid answers for a small number of questions.
        key = normalize_answer(text) or text
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def extract_expected_answers(metadata: Any) -> list[str]:
    """Extract ground-truth answers from task metadata."""
    if metadata is None:
        return []
    if not isinstance(metadata, Mapping):
        return _dedupe(_answer_candidates(metadata))

    # Prefer the clean target list when available. Some datasets store a
    # malformed stringified list in reward_model.ground_truth; Python
    # concatenates adjacent string literals, merging multiple answers into one.
    ground_truth_raw = metadata.get("ground_truth_raw")
    if isinstance(ground_truth_raw, Mapping):
        target = ground_truth_raw.get("target")
        if isinstance(target, list | tuple | set):
            raw_candidates = [str(item).strip() for item in target if str(item).strip()]
            if raw_candidates:
                return _dedupe(raw_candidates)
        elif target is not None:
            text = str(target).strip()
            if text:
                return [text]

    candidates: list[str] = []
    reward_model = metadata.get("reward_model")
    if reward_model is not None:
        candidates.extend(_answer_candidates(reward_model))

    prompt_row = metadata.get("prompt_row")
    if isinstance(prompt_row, Mapping):
        for key in _EXPECTED_KEYS:
            if key in prompt_row:
                candidates.extend(_answer_candidates(prompt_row[key]))

    for key in _EXPECTED_KEYS:
        if key in metadata:
            candidates.extend(_answer_candidates(metadata[key]))

    return _dedupe(candidates)


def exact_match_score(final_answer: Any, expected_answers: Any) -> float:
    """Return 1.0 for a normalized exact match against any expected answer."""
    prediction = normalize_answer(final_answer)
    raw_prediction = _text(final_answer).strip()
    if not prediction and not raw_prediction:
        return 0.0

    if isinstance(expected_answers, Mapping):
        candidates = _answer_candidates(expected_answers)
    elif isinstance(expected_answers, list | tuple | set):
        candidates = [str(answer) for answer in expected_answers]
    else:
        candidates = [expected_answers]

    for candidate in candidates:
        normalized = normalize_answer(candidate)
        if normalized and prediction == normalized:
            return 1.0
        # Punctuation-only answers normalize to ""; compare them verbatim.
        if not normalized and raw_prediction and raw_prediction == _text(candidate).strip():
            return 1.0
    return 0.0


def compute_score(final_answer: str | None, ground_truths: list[str]) -> float:
    """Return 1.0 for a normalized exact match against any expected answer.

    ``ExternalMASTask`` imports this function directly. A new external MAS app
    with a different correctness rule should provide its own Task class and
    evaluator module.
    """
    return exact_match_score(final_answer, ground_truths)
