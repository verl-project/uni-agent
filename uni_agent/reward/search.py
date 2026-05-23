"""Search task reward spec: exact-match scoring against ground truth answers."""

import json
import re
import string

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


def _normalize_answer(s: str) -> str:
    """Normalize answer for exact-match comparison."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _em_check(prediction: str, golden_answers: list[str]) -> bool:
    normalized = _normalize_answer(prediction)
    return any(_normalize_answer(g) == normalized for g in golden_answers)


def _extract_answer_from_trajectory(trajectory: list, messages: list[dict]) -> str | None:
    """Extract the answer submitted via the finish tool.

    Looks for the last trajectory step with exit_reason == "finished" and tries,
    in order:
    1. JSON-parse the <tool_call> block from the model response (most reliable).
    2. Fallback: use the finish/submit tool observation
       (the finish script echoes the answer to stdout).
    """
    for step in reversed(trajectory):
        if getattr(step, "exit_reason", None) != "finished":
            continue

        response = getattr(step, "response", "")

        # Primary: parse <tool_call> JSON
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if tc_match:
            try:
                tc_data = json.loads(tc_match.group(1).strip())
                args = tc_data.get("arguments") or tc_data.get("parameters") or {}
                # Some models emit `arguments` as a JSON-encoded string instead
                # of a nested object; tolerate both shapes.
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict) and "answer" in args:
                    return str(args["answer"])
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: the finish script prints the answer to stdout
        tool_results = getattr(step, "tool_results", []) or []
        for tr in reversed(tool_results):
            if getattr(tr, "name", "") in ("finish", "submit") and getattr(tr, "status", "") == "ok":
                obs = (getattr(tr, "observation", "") or "").strip()
                if obs:
                    return obs
                break

    return None


@register_reward_spec("search")
class SearchRewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, ground_truth: dict | None = None, env=None, **kwargs):
        self.run_id = run_id
        self.ground_truth = ground_truth or {}
        self.logger = get_logger("search-reward", run_id=run_id)

    @auto_await
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple[float, dict]:
        trajectory = interaction_result.get("trajectory", [])
        messages = interaction_result.get("messages", [])
        answer = _extract_answer_from_trajectory(trajectory, messages)

        result = {
            "extracted_answer": answer,
            "ground_truth": self.ground_truth,
            "score": 0.0,
        }

        if answer is None:
            self.logger.warning("No answer extracted from trajectory")
            return 0.0, result

        targets = self.ground_truth.get("target", [])
        if not targets:
            self.logger.warning("No ground truth targets provided")
            return 0.0, result

        score = 1.0 if _em_check(answer, targets) else 0.0
        result["score"] = score
        self.logger.info(f"Answer: {answer!r}, Targets: {targets}, Score: {score}")
        return score, result
