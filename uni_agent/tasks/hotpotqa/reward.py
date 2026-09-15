"""Reward used by the HotpotQA task."""

from __future__ import annotations


def compute_score(solution: str, ground_truths: list[str]) -> float:
    """Score the final boxed answer with MemAgent's normalized exact match."""

    solution = solution[-300:].lower()
    return max((_compute_single(solution, answer) for answer in ground_truths), default=0.0)


def _compute_single(solution: str, ground_truth: str) -> float:
    ground_truth = ground_truth.lower()
    try:
        boxed = last_boxed_only_string(solution)
        if boxed is None:
            return 0.0
        answer = remove_boxed(boxed)
        return 1.0 if is_equiv(answer, ground_truth) else 0.0
    except Exception:  # noqa: BLE001 - match the official MemAgent verifier
        return 0.0


def is_equiv(left: str | None, right: str | None) -> bool:
    """Compare two answers using the official MemAgent normalization."""

    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    try:
        return strip_string(left) == strip_string(right)
    except Exception:  # noqa: BLE001 - match the official MemAgent verifier
        return left == right


def remove_boxed(value: str) -> str:
    if "\\boxed " in value:
        prefix = "\\boxed "
        assert value[: len(prefix)] == prefix
        return value[len(prefix) :]

    prefix = "\\boxed{"
    assert value[: len(prefix)] == prefix
    assert value[-1] == "}"
    return value[len(prefix) : -1]


def last_boxed_only_string(value: str) -> str | None:
    index = value.rfind("\\boxed")
    if "\\boxed " in value:
        return "\\boxed " + value.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = value.rfind("\\fbox")
        if index < 0:
            return None

    open_braces = 0
    for position in range(index, len(value)):
        if value[position] == "{":
            open_braces += 1
        elif value[position] == "}":
            open_braces -= 1
            if open_braces == 0:
                return value[index : position + 1]
    return None


def strip_string(value: str) -> str:
    """Normalize strings exactly as the official MemAgent HotpotQA verifier."""

    value = value.replace("\n", "")
    value = value.replace("\\!", "")
    value = value.replace("\\\\", "\\")
    value = value.replace("tfrac", "frac")
    value = value.replace("dfrac", "frac")
    value = value.replace("\\left", "")
    value = value.replace("\\right", "")
    value = value.replace("^{\\circ}", "")
    value = value.replace("^\\circ", "")
    value = value.replace("\\$", "")
    value = value.replace("\\%", "")
    value = value.replace(" .", " 0.")
    value = value.replace("{.", "{0.")
    if not value:
        return value
    return value.replace(" ", "")
