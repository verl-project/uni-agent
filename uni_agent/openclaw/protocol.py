"""Backend-agnostic OpenClaw session protocol.

OpenClaw drives a self-hosted model through an OpenAI-compatible proxy. Each
request carries control fields that decide whether the turn is trainable:

- ``turn_type``    ``"main"`` (trainable) vs anything else (``"side"``, skipped).
- ``session_id``   groups multi-turn conversations.
- ``session_done`` marks the end of a session.

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Body keys that are OpenClaw-specific and must be stripped before forwarding a
# request to the underlying inference engine.
NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}

_TRUE_STRINGS = {"1", "true", "yes", "on"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_KIMI_TC_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------
# Control-field parsing
# ---------------------------------------------------------------------------


def parse_turn_type(header_value: str | None, body: dict[str, Any]) -> str:
    """Resolve the turn type from header or body, defaulting to ``"side"``."""
    raw = header_value or body.get("turn_type") or "side"
    return str(raw).strip().lower()


def parse_session_id(header_value: str | None, body: dict[str, Any]) -> str:
    return header_value or body.get("session_id") or "unknown"


def parse_session_done(header_value: str | None, body: dict[str, Any]) -> bool:
    if header_value and header_value.strip().lower() in _TRUE_STRINGS:
        return True
    return str(body.get("session_done", "")).strip().lower() in _TRUE_STRINGS


def is_main_turn(turn_type: str) -> bool:
    return turn_type == "main"


def strip_non_standard_keys(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``body`` without OpenClaw-specific control keys."""
    return {k: v for k, v in body.items() if k not in NON_STANDARD_BODY_KEYS}


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def flatten_content(content: Any) -> str:
    """Flatten an OpenAI message ``content`` (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def normalize_tool_call(tc: dict) -> dict:
    """Normalize a tool call so ``function.arguments`` is a dict (for templating)."""
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        tc["function"] = fn
    return tc


def normalize_messages(messages: list[dict], *, normalize_tool_calls: bool = False) -> list[dict]:
    """Normalize messages for a chat template.

    - ``developer`` role -> ``system``.
    - non-string content is flattened to text.
    - optionally, ``tool_calls[*].function.arguments`` parsed from JSON strings.
    """
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = flatten_content(raw)
        if normalize_tool_calls and m.get("tool_calls"):
            m["tool_calls"] = [normalize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def next_state_text_role(next_state: dict | None) -> tuple[str, str]:
    """Return ``(text, role)`` for a next-state message, with safe defaults."""
    if not next_state:
        return "", "user"
    return flatten_content(next_state.get("content")), next_state.get("role", "user")


# ---------------------------------------------------------------------------
# Logprob / tool-call extraction from chat responses
# ---------------------------------------------------------------------------


def extract_logprobs_from_choice(choice: dict[str, Any]) -> list[float]:
    """Extract per-token logprobs from an OpenAI ``choices[i].logprobs`` block."""
    lp_obj = choice.get("logprobs")
    if not isinstance(lp_obj, dict):
        return []
    content = lp_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def fit_length(values: list[float], length: int, pad: float = 0.0) -> list[float]:
    """Truncate or right-pad ``values`` to exactly ``length`` entries."""
    if len(values) > length:
        return values[:length]
    if len(values) < length:
        return values + [pad] * (length - len(values))
    return values


def extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse Kimi/Qwen tool-call tags from assistant text into OpenAI tool_calls.

    Returns ``(clean_text, tool_calls)``.
    """
    if not text:
        return "", []
    tool_calls: list[dict] = []
    for i, m in enumerate(_KIMI_TC_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        try:
            args_str = json.dumps(json.loads(args_raw), ensure_ascii=False)
        except Exception:
            args_str = args_raw
        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": raw_name or "unknown_tool", "arguments": args_str},
            }
        )
    for i, m in enumerate(_QWEN_TC_RE.finditer(text), start=len(tool_calls)):
        try:
            payload = json.loads(m.group(1).strip())
        except Exception:
            continue
        name = payload.get("name") or payload.get("function", {}).get("name") or "unknown_tool"
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": str(name), "arguments": args},
            }
        )
    clean = _THINK_RE.sub("", text)
    clean = clean.replace("</think>", "")
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", "", clean, flags=re.DOTALL)
    clean = _QWEN_TC_RE.sub("", clean)
    return clean.strip(), tool_calls


# ---------------------------------------------------------------------------
# Turn buffering
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One buffered main-line turn awaiting a next-state PRM/judge signal.

    Backends store their tokenized turn here; the ``messages``/``tools`` fields
    are needed to build the hint-augmented teacher prompt for OPD.
    """

    session_id: str
    turn_num: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]
    prompt_text: str = ""
    response_text: str = ""
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    has_next_state: bool = False
