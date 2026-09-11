"""Small text task that exercises rollout, reward reporting, and policy update."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_reward, compute_score

logger = logging.getLogger(__name__)


class PipelineSmokeTaskConfig(TaskConfig):
    """One deterministic prompt and its expected short answer."""

    name: str = "pipeline_smoke"
    expected_answer: str = Field(description="Exact answer after whitespace/case normalization.")


_FINISH_TAG_RE = re.compile(r"<finish>\s*(.*?)\s*</finish>", re.IGNORECASE | re.DOTALL)
_FINAL_LABEL_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*:?\s*([^\n]+)", re.IGNORECASE)
_ANSWER_JSON_RE = re.compile(r"['\"]answer['\"]\s*:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_EQUATION_SUFFIX_RE = re.compile(r"(?<![!<>=])=(?!=)\s*['\"]?([^'\"\n,;!?=]+)['\"]?[.,;!?]*\s*$")
_IS_SUFFIX_RE = re.compile(r"\bis\s+['\"]?([^'\"\n.,:;!?]+)['\"]?[.,;!?]*\s*$", re.IGNORECASE)
_OBSERVATION_PREFIX_RE = re.compile(r"^Observation:[ \t]*(?:\r?\n)?", re.IGNORECASE)


def _clean_candidate(value: str) -> str:
    return value.strip().strip("`*_ ").strip(".,:;!? ").strip("'\"").strip()


def _extract_answer_from_text(content: str) -> str:
    """Extract common concise-answer forms emitted by small chat models."""
    tagged = _FINISH_TAG_RE.search(content)
    if tagged:
        return _clean_candidate(tagged.group(1))

    json_answer = _ANSWER_JSON_RE.search(content)
    if json_answer:
        return _clean_candidate(json_answer.group(1))

    labelled = _FINAL_LABEL_RE.search(content)
    if labelled:
        return _clean_candidate(labelled.group(1))

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in reversed(lines):
        equation_suffix = _EQUATION_SUFFIX_RE.search(line)
        if equation_suffix:
            return _clean_candidate(equation_suffix.group(1))
        is_suffix = _IS_SUFFIX_RE.search(line)
        if is_suffix:
            return _clean_candidate(is_suffix.group(1))
    return _clean_candidate(lines[-1]) if lines else ""


def _extract_final_answer_and_source(transcript: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    """Return the latest answer and whether it came from a finish observation."""
    for message in reversed(transcript):
        if message.get("role") == "tool" and message.get("name") == "finish":
            observation = _OBSERVATION_PREFIX_RE.sub("", str(message.get("content", "")), count=1)
            return _clean_candidate(observation), True
        if message.get("role") == "assistant" and message.get("content"):
            return _extract_answer_from_text(str(message["content"])), False
    return "", False


def extract_final_answer(transcript: Sequence[Mapping[str, Any]]) -> str:
    """Read the latest finish observation or conservative plain-text answer."""
    return _extract_final_answer_and_source(transcript)[0]


@register_task("pipeline_smoke")
class PipelineSmokeTask(Task):
    """Run a tool-safe ReAct episode and score its final text answer."""

    config_model = PipelineSmokeTaskConfig

    async def run(self) -> TaskResult:
        cfg: PipelineSmokeTaskConfig = self.config  # type: ignore[assignment]
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir=None,
            )

        answer, used_finish_tool = _extract_final_answer_and_source(agent_result.transcript)
        accuracy = compute_score(answer, cfg.expected_answer)
        reward = compute_reward(answer, cfg.expected_answer, used_finish_tool=used_finish_tool)
        if used_finish_tool and accuracy == 1.0 and agent_result.finished is True:
            logger.info("pipeline_smoke: correct_finish_observed")
        return TaskResult(
            reward=reward,
            accuracy=accuracy,
            finished=agent_result.finished,
            extra_info={
                "answer": answer,
                "expected_answer": cfg.expected_answer,
                "exact_match": bool(accuracy),
                "used_finish_tool": used_finish_tool,
                "steps": agent_result.info.get("steps", 0),
                "tool_calls": agent_result.info.get("num_tool_calls", 0),
            },
        )
