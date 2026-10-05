"""LLM-as-a-Judge reward for the DeepEyes task."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class DeepEyesJudgeConfig(BaseModel):
    """Connection and generation settings for the semantic Judge."""

    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    strict: bool = True
    temperature: float = Field(default=0.1, ge=0.0)
    preflight_max_tokens: int = Field(default=8, gt=0)
    enable_thinking: bool = False

    model_config = ConfigDict(extra="forbid")


class DeepEyesRewardConfig(BaseModel):
    """DeepEyes reward limits and Judge configuration."""

    max_answer_chars: int = Field(default=1000, gt=0)
    judge: DeepEyesJudgeConfig = Field(default_factory=DeepEyesJudgeConfig)

    model_config = ConfigDict(extra="forbid")


def _judge_settings(config: DeepEyesJudgeConfig) -> tuple[str, str, str, float, int]:
    base_url = (config.base_url or os.environ.get("LLM_AS_A_JUDGE_BASE", "")).rstrip("/")
    if not base_url:
        raise ValueError("DeepEyes Judge requires reward.judge.base_url or LLM_AS_A_JUDGE_BASE")
    api_key = config.api_key or os.environ.get("LLM_AS_A_JUDGE_API_KEY", "EMPTY")
    model_name = config.model_name or os.environ.get("LLM_AS_A_JUDGE_MODEL", "")
    return base_url, api_key, model_name, config.timeout_seconds, config.max_retries


@lru_cache(maxsize=8)
def _build_judge_client(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout: float,
    max_retries: int,
) -> tuple[Any, str]:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            # Reward workers inherit the host's proxy environment.  The Judge
            # is a local service and must never be routed through that proxy.
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        if not model_name:
            models = client.models.list()
            if models.data:
                model_name = models.data[0].id
        if not model_name:
            raise RuntimeError(f"judge service at {base_url} returned no models")
        return client, model_name
    except Exception as error:  # noqa: BLE001 - annotated with the service endpoint
        raise RuntimeError(f"DeepEyes judge initialization failed for {base_url}: {error}") from error


def _get_judge_client(config: DeepEyesJudgeConfig) -> tuple[Any | None, str]:
    try:
        return _build_judge_client(*_judge_settings(config))
    except Exception as error:  # noqa: BLE001 - handled by the configured strict policy
        if config.strict:
            raise
        logger.warning("DeepEyes judge unavailable; returning zero rewards: %s", error)
        return None, ""


def check_judge(config: DeepEyesRewardConfig | None = None) -> str:
    """Run a real semantic judgement and return the selected model name."""
    reward_config = config or DeepEyesRewardConfig()
    judge_config = reward_config.judge
    client, model_name = _get_judge_client(judge_config)
    if client is None or not model_name:
        raise RuntimeError("DeepEyes judge is unavailable")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _judge_system_prompt()},
                {
                    "role": "user",
                    "content": _judge_user_prompt(
                        question="What is two plus two?",
                        ground_truth="4",
                        answer="4",
                    ),
                },
            ],
            temperature=judge_config.temperature,
            max_tokens=judge_config.preflight_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": judge_config.enable_thinking}},
        )
    except Exception as error:  # noqa: BLE001 - preflight must fail closed
        raise RuntimeError(f"DeepEyes judge completion preflight failed: {error}") from error
    judgement = (response.choices[0].message.content or "").strip()
    if not re.search(r"\bCORRECT\b", judgement, re.IGNORECASE):
        raise RuntimeError(f"DeepEyes judge completion preflight returned {judgement!r}, expected CORRECT")
    return model_name


def compute_score(
    final_answer: str,
    ground_truth: str,
    *,
    question: str,
    finished: bool,
    tool_successes: int,
    reward_config: DeepEyesRewardConfig | None = None,
) -> dict[str, float]:
    """Score one structured final assistant answer.

    Episode telemetry is passed explicitly; no serialized transcript parsing is
    performed here.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("DeepEyes reward requires a question")
    if not isinstance(final_answer, str):
        raise TypeError("DeepEyes final_answer must be a string")
    config = reward_config or DeepEyesRewardConfig()
    answer_text, format_error, answer_tags = _parse_final_answer(final_answer)
    if not finished:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=True,
            tool_successes=tool_successes,
            answer_tags=answer_tags,
        )

    # Keep this guard before the Judge request: empty/very long output must
    # not push the Judge prompt beyond its serving context window.
    if not answer_text or len(answer_text) >= config.max_answer_chars:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=True,
            tool_successes=tool_successes,
            answer_tags=answer_tags,
        )

    client, model_name = _get_judge_client(config.judge)
    if client is None or not model_name:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=format_error,
            tool_successes=tool_successes,
            answer_tags=answer_tags,
            score_override=0.0,
        )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _judge_system_prompt()},
                {
                    "role": "user",
                    "content": _judge_user_prompt(
                        question=question,
                        ground_truth=str(ground_truth),
                        answer=answer_text,
                    ),
                },
            ],
            temperature=config.judge.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": config.judge.enable_thinking}},
        )
    except Exception as error:  # noqa: BLE001 - policy is controlled by the task config
        if config.judge.strict:
            raise RuntimeError(f"DeepEyes judge request failed: {error}") from error
        logger.warning("DeepEyes judge request failed; returning zero rewards: %s", error)
        return _reward_result(
            accuracy_reward=0.0,
            format_error=format_error,
            tool_successes=tool_successes,
            answer_tags=answer_tags,
            score_override=0.0,
        )
    judgement = (response.choices[0].message.content or "").strip()
    if re.search(r"\bINCORRECT\b", judgement, re.IGNORECASE):
        accuracy_reward = 0.0
    elif re.search(r"\bCORRECT\b", judgement, re.IGNORECASE):
        accuracy_reward = 1.0
    else:
        raise ValueError(f"Judge returned neither CORRECT nor INCORRECT: {judgement!r}")

    return _reward_result(
        accuracy_reward=accuracy_reward,
        format_error=format_error,
        tool_successes=tool_successes,
        answer_tags=answer_tags,
    )


def score_from_runner_result(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any],
    **_reward_manager_kwargs: Any,
) -> dict[str, int | float | bool]:
    """Adapt a DeepEyes TaskResult for a streaming RewardLoopWorker.

    The Task has already run the Judge. This callback forwards its scalar result
    and promotes the stable DeepEyes reward/tool components from reward_context
    into validation metrics without running the Judge a second time.
    """
    del data_source, solution_str, ground_truth
    runner_reward_info = extra_info.get("runner_reward_info")
    if not isinstance(runner_reward_info, dict):
        raise ValueError("DeepEyes scorer requires extra_info.runner_reward_info")
    reward = runner_reward_info.get("reward")
    if reward is None:
        raise ValueError("DeepEyes runner result must contain a scalar reward")
    metrics = runner_reward_info.get("metrics") or {}
    reward_context = runner_reward_info.get("reward_context") or {}
    if not isinstance(metrics, dict) or not isinstance(reward_context, dict):
        raise ValueError("DeepEyes runner metrics and reward_context must be mappings")

    result: dict[str, int | float | bool] = dict(metrics)
    for key in ("format", "tool", "answer_tags", "tool_calls", "tool_successes", "tool_errors"):
        value = reward_context.get(key)
        if not isinstance(value, int | float | bool):
            raise ValueError(f"DeepEyes runner reward_context.{key} must be numeric")
        result[key] = value
    result["score"] = float(reward)
    return result


def _reward_result(
    *,
    accuracy_reward: float,
    format_error: bool,
    tool_successes: int,
    answer_tags: bool,
    score_override: float | None = None,
) -> dict[str, float]:
    tool_reward = 1.0 if tool_successes > 0 and accuracy_reward > 0.5 else 0.0
    format_reward = -1.0 if format_error else 0.0
    final_score = 0.8 * accuracy_reward + 0.2 * format_reward + 1.2 * tool_reward
    return {
        "score": final_score if score_override is None else score_override,
        "acc": accuracy_reward,
        "format": format_reward,
        "tool": tool_reward,
        "answer_tags": float(answer_tags),
    }


def _parse_final_answer(final_content: str) -> tuple[str, bool, bool]:
    """Extract answer text from one final assistant content value."""
    think_open_count = final_content.count("<think>")
    think_close_count = final_content.count("</think>")
    # Qwen3.5's native chat template pre-fills ``<think>`` in the generation
    # prompt. It is therefore part of prompt_ids, while the structured final
    # assistant content starts after that token and contains only ``</think>``.
    # Treat exactly one such missing opener as a valid native turn boundary.
    prefilled_think = think_open_count == 0 and think_close_count == 1
    format_error = think_open_count != think_close_count and not prefilled_think
    answer_region = (
        final_content.split("</think>")[-1].strip() if "</think>" in final_content else final_content.strip()
    )
    answer_open_count = answer_region.count("<answer>")
    answer_close_count = answer_region.count("</answer>")
    match = re.search(r"<answer>(.*?)</answer>", answer_region, re.DOTALL)
    answer_tags = answer_open_count == 1 and answer_close_count == 1 and match is not None
    if not answer_tags:
        format_error = True

    if answer_tags and match:
        answer = match.group(1).strip()
    else:
        # Plain text remains eligible for semantic judging, with a format
        # penalty. Malformed answer blocks are likewise passed as-is.
        answer = answer_region
    return answer, format_error, answer_tags


def _judge_system_prompt() -> str:
    return (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )


def _judge_user_prompt(*, question: str, ground_truth: str, answer: str) -> str:
    return (
        "I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        "answer is correct.\n\n"
        "---\n"
        "**Example 1:**\n"
        "[Question]: Is the countertop tan or blue?\n"
        "[Standard Answer]: The countertop is tan.\n"
        "[Model's Answer]: tan\n"
        "[Your Judgement]: CORRECT\n"
        "---\n"
        "**Example 2:**\n"
        "[Question]: Is the man phone both blue and closed?\n"
        "[Standard Answer]: Yes, the man phone is both blue and closed.\n"
        "[Model's Answer]: No.\n"
        "[Your Judgement]: INCORRECT\n"
        "---\n"
        "**Task:**\n"
        f"[Question]: {question}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer}\n"
        "[Your Judgement]:"
    )


if __name__ == "__main__":
    selected_model = check_judge()
    print(f"DeepEyes judge ready: {selected_model}")
