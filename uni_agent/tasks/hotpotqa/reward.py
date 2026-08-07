"""Reward used by the HotpotQA task."""

from __future__ import annotations


def compute_score(solution: str, ground_truths: list[str]) -> float:
    """Score the final boxed answer with the original token-level LCS metric."""

    solution = solution[-300:].lower()
    return max((_compute_single(solution, answer) for answer in ground_truths), default=0.0)


def _compute_single(solution: str, ground_truth: str) -> float:
    try:
        boxed = last_boxed_only_string(solution)
        if boxed is None:
            return 0.0
        return _lcs_ratio(remove_boxed(boxed), ground_truth.lower())
    except (AssertionError, ValueError):
        return 0.0


def _lcs_ratio(value: str, ground_truth: str) -> float:
    left = value.lower().split()
    right = ground_truth.lower().split()
    if not left or not right:
        return 0.0

    dp = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            current = dp[index]
            if left_token == right_token:
                dp[index] = previous + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous = current
    return dp[-1] / max(len(left), len(right))


def remove_boxed(value: str) -> str:
    if value.startswith("\\boxed "):
        return value[len("\\boxed ") :]

    prefix = "\\boxed{"
    if not value.startswith(prefix) or not value.endswith("}"):
        raise ValueError(f"Invalid boxed answer: {value!r}")
    answer = value[len(prefix) : -1]
    if "\\text{" in answer and "}" in answer:
        answer = answer.split("\\text")[-1].strip(" {}")
    return answer


def last_boxed_only_string(value: str) -> str | None:
    if "\\boxed " in value:
        return "\\boxed " + value.split("\\boxed ")[-1].split("$")[0]

    index = value.rfind("\\boxed")
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
