"""Request-level trajectory annotation policies for gateway requests.

The default policy only emits conservative, non-sensitive evidence.  It does
not decide which chain a request belongs to and it does not claim that a
compaction request completed successfully.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable, TypeAlias

TrajectoryAnnotationPolicy: TypeAlias = Callable[[Mapping[str, str], Mapping[str, Any], str], Mapping[str, Any]]

_CLAUDE_AGENT_HEADER = "x-claude-code-agent-id"
_CODEX_SUBAGENT_HEADER = "x-openai-subagent"
_CODEX_TURN_METADATA_HEADER = "x-codex-turn-metadata"
_DEEPSEEK_COMPACT_HEADER = "x-deepseek-harness-compact"
_OPENCLAW_SUBAGENT_SYSTEM_MARKERS = (
    "Subagent spawned by main agent; one specific task.",
    "Subagent spawned by parent orchestrator; one specific task.",
)
_OPENCLAW_SUBAGENT_USER_MARKER = "[Subagent Context] You are running as a subagent"
_OPENCLAW_SUMMARIZER_MARKER = "You are a context summarization assistant."
_OPENCLAW_COMPACTED_HISTORY_MARKER = "The conversation history before this point was compacted into"
_OPENCLAW_BRANCH_SUMMARY_MARKER = "The following is a summary of a branch that this conversation came back from:"


def _header_values(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _prompt_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _prompt_text(item)]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _prompt_text(item)]
    return []


def _prompt_channels(body: Mapping[str, Any]) -> tuple[str, str]:
    system_parts = _prompt_text(body.get("system"))
    first_user_parts: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "system":
                system_parts.extend(_prompt_text(message.get("content")))
            elif message.get("role") == "user" and not first_user_parts:
                first_user_parts.extend(_prompt_text(message.get("content")))
    return "\n".join(system_parts), "\n".join(first_user_parts)


def _nested_value(value: Any, path: tuple[str, ...]) -> Any:
    if not path or not isinstance(value, Mapping):
        return value if not path else None
    return _nested_value(value[path[0]], path[1:]) if path[0] in value else None


def _compaction_request_kind(value: str) -> str | None:
    try:
        metadata = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(metadata, Mapping) and metadata.get("request_kind") == "compaction":
        return "compaction"
    return None


def infer_trajectory_annotations(
    headers: Mapping[str, str], body: Mapping[str, Any], protocol: str
) -> dict[str, list[str]]:
    """Infer conservative request tags from provider-visible fields.

    ``protocol`` is part of the policy contract so custom policies can use the
    wire protocol without inspecting gateway internals.  The built-in policy
    intentionally applies the same checks to both supported protocols because
    harnesses commonly send compatible extensions through either route.
    """
    del protocol
    normalized_headers = _header_values(headers)
    system_prompt, first_user_prompt = _prompt_channels(body)
    tags: set[str] = set()
    evidence: set[str] = set()

    if normalized_headers.get(_CLAUDE_AGENT_HEADER, "").strip():
        tags.add("role:subagent")
        evidence.add("header:x-claude-code-agent-id:present")
    if normalized_headers.get(_CODEX_SUBAGENT_HEADER) == "collab_spawn":
        tags.add("role:subagent")
        evidence.add("header:x-openai-subagent=collab_spawn")

    session_origin = _nested_value(body, ("dsh_session_log", "session", "origin"))
    if session_origin == "subagent":
        tags.add("role:subagent")
        evidence.add("body:dsh_session_log.session.origin=subagent")

    openclaw_subagent_system = any(marker in system_prompt for marker in _OPENCLAW_SUBAGENT_SYSTEM_MARKERS)
    openclaw_subagent_user = first_user_prompt.lstrip().startswith(_OPENCLAW_SUBAGENT_USER_MARKER)
    if openclaw_subagent_system or openclaw_subagent_user:
        tags.add("role:subagent")
        evidence.add("prompt:openclaw-subagent-marker")

    codex_kind = _compaction_request_kind(normalized_headers.get(_CODEX_TURN_METADATA_HEADER, ""))
    if codex_kind == "compaction":
        tags.add("purpose:compaction_request")
        evidence.add("header:x-codex-turn-metadata.request_kind=compaction")
    if normalized_headers.get(_DEEPSEEK_COMPACT_HEADER) == "1":
        tags.add("purpose:compaction_request")
        evidence.add("header:x-deepseek-harness-compact=1")
    if _OPENCLAW_SUMMARIZER_MARKER in system_prompt:
        tags.add("purpose:summary_generation")
        evidence.add("prompt:openclaw-summarizer-system")

    if first_user_prompt.lstrip().startswith(_OPENCLAW_COMPACTED_HISTORY_MARKER):
        tags.add("context:compacted_history")
        evidence.add("prompt:openclaw-compacted-history-marker")
    if first_user_prompt.lstrip().startswith(_OPENCLAW_BRANCH_SUMMARY_MARKER):
        tags.add("context:branch_return_summary")
        evidence.add("prompt:openclaw-branch-summary-marker")

    if not any(tag.startswith("role:") for tag in tags):
        tags.add("role:unknown")
    return {"tags": sorted(tags), "evidence": sorted(evidence)}


def normalize_trajectory_annotations(value: Mapping[str, Any]) -> dict[str, list[str]]:
    """Validate the small serializable result accepted from custom policies."""
    if not isinstance(value, Mapping):
        raise TypeError(f"trajectory annotation policy must return a mapping, got {type(value).__name__}")
    tags = value.get("tags", [])
    evidence = value.get("evidence", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise TypeError("trajectory annotation policy 'tags' must be a list of strings")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise TypeError("trajectory annotation policy 'evidence' must be a list of strings")
    return {"tags": sorted(set(tags)), "evidence": sorted(set(evidence))}
