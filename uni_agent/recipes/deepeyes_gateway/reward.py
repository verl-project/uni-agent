"""Reward function for the DeepEyes gateway recipe."""

from __future__ import annotations

import logging
import os
import random
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_BASE = "http://127.0.0.1:18901/v1"


@lru_cache(maxsize=1)
def _get_judge_client() -> tuple[Any | None, str]:
    api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", DEFAULT_JUDGE_BASE)
    model_name = os.environ.get("LLM_AS_A_JUDGE_MODEL", "")

    try:
        from openai import OpenAI
        import requests
    except ImportError as error:
        logger.warning("Reward scoring dependencies are unavailable: %s", error)
        return None, ""

    client = OpenAI(api_key=os.environ.get("LLM_AS_A_JUDGE_API_KEY", "EMPTY"), base_url=api_base)
    if model_name:
        return client, model_name

    try:
        timeout = float(os.environ.get("LLM_AS_A_JUDGE_DISCOVERY_TIMEOUT_SECONDS", "10"))
        response = requests.get(f"{api_base.rstrip('/')}/models", timeout=timeout)
        response.raise_for_status()
        models = response.json()
        if models.get("data"):
            model_name = models["data"][0]["id"]
        else:
            logger.warning("No models found at the specified API base for reward scoring.")
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as error:
        logger.warning("Failed to get model from %s: %s. Reward scoring will be disabled.", api_base, error)

    return client, model_name


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    is_format_error = False
    if solution_str.count("<think>") != solution_str.count("</think>"):
        is_format_error = True

    predict_no_think = solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()

    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        is_format_error = True

    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        is_format_error = True
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        elif "</think>" in solution_str:
            remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", predict_no_think, flags=re.DOTALL)
            remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
            answer_text = remaining_content.strip()
        else:
            answer_text = solution_str.strip()

    answer_text = answer_text.strip()
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    return answer_text, is_format_error


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Compute the DeepEyes answer reward with format and tool-use shaping."""
    del data_source
    answer_text, is_format_error = _extract_answer(solution_str)
    question_text = extra_info.get("question", "") if extra_info else ""

    client, model_name = _get_judge_client()
    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    system_prompt = (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )

    user_prompt = (
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
        f"[Question]: {question_text}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer_text}\n"
        "[Your Judgement]:"
    )

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as error:
        logger.warning(" [WARNING] Chat completion request failed: %s", error)
        return 0.0

    if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
        acc_reward = 1.0
    elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
        acc_reward = 0.0
    else:
        logger.warning(
            " [WARNING] Judgement format error. Expected 'CORRECT' or 'INCORRECT'.\n"
            "Response: '%s'\n"
            "Model Answer: '%s'\n"
            "Ground Truth: '%s'",
            response,
            answer_text,
            ground_truth,
        )
        acc_reward = 0.0

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    has_tool_usage = bool(
        re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
        or re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
    )
    tool_reward = 1.0 if has_tool_usage and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0

    if is_format_error or not answer_text:
        logger.debug(
            "Format issue detected:\nSolution: %s...\nExtracted answer: '%s'\nFormat error: %s\nTool usage: %s",
            solution_str[:200],
            answer_text,
            is_format_error,
            has_tool_usage,
        )

    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
