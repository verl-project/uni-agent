"""Sample select functions for ``agent_runners.<name>.trajectory_selection``.

Each function receives the session's trajectories and returns the ones to keep.
Config and the full contract are in README.md; the markers below are illustrative
and agent-specific.
"""

from __future__ import annotations

# A caller asking the model to name the session before/after the real work.
TITLE_MARKERS = (
    "create a concise title",
    "generate the session title",
    "return only the title",
)

# A summarisation request replaying compacted history. Match the *request*
# prompt only: the summarised result is later injected into the working chain,
# so matching the result would drop the task rollout.
COMPACTION_MARKERS = (
    "<state_snapshot>",
    "<coverage_check>",
    "summarize the conversation so far",
)

# A shell segment that carries someone else's work rather than the task's.
SUBAGENT_MARKERS = (
    "you are a subagent",
    "spawned by the main agent",
)


def _message_text(message: dict) -> str:
    """Flatten one message's text, tool names and tool arguments into a string."""
    content = message.get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        # Multimodal content arrives as a list of parts; only text is searchable.
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
    else:
        parts = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            parts.append(str(function.get("name") or ""))
            parts.append(str(function.get("arguments") or ""))
    return "\n".join(part for part in parts if part)


def _history(trajectory) -> list[dict]:
    return trajectory.extra_fields.get("message_history") or []


def trajectory_text(trajectory) -> str:
    """All text of a chain, in message order."""
    return "\n".join(_message_text(message) for message in _history(trajectory))


def system_text(trajectory) -> str:
    """Only the system messages of a chain."""
    return "\n".join(_message_text(message) for message in _history(trajectory) if message.get("role") == "system")


def _matches(trajectory, markers: tuple[str, ...]) -> bool:
    text = trajectory_text(trajectory).lower()
    return any(marker in text for marker in markers)


def is_title_segment(trajectory) -> bool:
    """True for a chain that only asked the model to name the session."""
    return _matches(trajectory, TITLE_MARKERS)


def is_compaction_segment(trajectory) -> bool:
    """True for a chain that only asked the model to summarise history."""
    return _matches(trajectory, COMPACTION_MARKERS)


def is_subagent_segment(trajectory) -> bool:
    """True for a chain opened on another agent's behalf."""
    return _matches(trajectory, SUBAGENT_MARKERS)


def drop_title_segments(session_id: str, trajectories: list) -> list:
    """Keep everything except title generation."""
    return [trajectory for trajectory in trajectories if not is_title_segment(trajectory)]


def drop_compaction_segments(session_id: str, trajectories: list) -> list:
    """Keep everything except compaction replays."""
    return [trajectory for trajectory in trajectories if not is_compaction_segment(trajectory)]


def drop_subagent_segments(session_id: str, trajectories: list) -> list:
    """Keep everything except subagent shells."""
    return [trajectory for trajectory in trajectories if not is_subagent_segment(trajectory)]


def drop_non_work_segments(session_id: str, trajectories: list) -> list:
    """Drop title, compaction and subagent segments."""
    return [
        trajectory
        for trajectory in trajectories
        if not (is_title_segment(trajectory) or is_compaction_segment(trajectory) or is_subagent_segment(trajectory))
    ]


def keep_first_work_segment(session_id: str, trajectories: list) -> list:
    """Keep only the earliest chain that is not a title request.

    ``chain_id`` is the creation order, so ``min`` finds the chain that was
    started first even though the list arrives in close order. Returning exactly
    one entry also makes it the entry reward scoring sees.
    """
    work = [trajectory for trajectory in trajectories if not is_title_segment(trajectory)]
    if not work:
        return []
    return [min(work, key=lambda trajectory: trajectory.chain_id or 0)]
