"""Select KernelBench assistant prefixes without changing their reward."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_agent.gateway.session import Trajectory
    from uni_agent.tasks import TaskResult

logger = logging.getLogger(__name__)

_STALE_AFTER_CROP = {
    "materialization_reason",
    "min_global_steps",
    "max_global_steps",
}


def process_trajectories(
    trajectories: tuple[Trajectory, ...],
    *,
    task_result: TaskResult,
    selection: str = "best",
) -> list[Trajectory]:
    """Select best, all finalized prefixes, or the last finalized prefix.

    A scalar best hint is usable only for one trajectory. Missing or ambiguous
    hints fall back to all finalized prefixes. No-implementation outcomes are
    retained with their existing reward; token-array misalignment raises.
    """
    if selection not in {"best", "all_final", "final"}:
        raise ValueError(f"invalid selection={selection!r}; expected 'best', 'all_final' or 'final'")
    if not trajectories:
        return []
    for trajectory in trajectories:
        _validate_alignment(trajectory)

    if selection == "best":
        if len(trajectories) == 1:
            selected = _select_best(trajectories[0], task_result.extra_info)
            if selected is not None:
                return [selected]
        elif isinstance(task_result.extra_info.get("train_best"), Mapping):
            logger.warning(
                "ignoring scalar best-assistant hint for %s finalized trajectories; falling back to all_final",
                len(trajectories),
            )

    candidates = trajectories[-1:] if selection == "final" else trajectories
    reason = "final" if selection == "final" else "all_final"
    result = []
    for trajectory in candidates:
        spans = assistant_spans(trajectory)
        if spans:
            result.append(crop_to_assistant_prefix(trajectory, spans[-1][1], reason=reason))
    return result


def assistant_spans(trajectory: Trajectory) -> list[tuple[int, int]]:
    """Return half-open model-token runs in response token coordinates."""
    _validate_alignment(trajectory)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(trajectory.response_mask):
        active = bool(int(value))
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(trajectory.response_ids)))
    return spans


def crop_to_assistant_prefix(
    trajectory: Trajectory,
    response_end: int,
    *,
    reason: str,
) -> Trajectory:
    """Copy tokens and routing through an assistant boundary, preserving reward."""
    spans = assistant_spans(trajectory)
    if response_end not in {end for _start, end in spans}:
        raise ValueError(f"response_end={response_end} is not an assistant boundary")

    extra_fields = dict(trajectory.extra_fields)
    if response_end < len(trajectory.response_ids):
        for key in _STALE_AFTER_CROP:
            extra_fields.pop(key, None)
    kept_spans = sum(end <= response_end for _start, end in spans)
    extra_fields.update(
        {
            "trajectory_postprocessed": True,
            "trajectory_postprocess_reason": reason,
            "original_response_length": len(trajectory.response_ids),
            "assistant_spans_kept": kept_spans,
            # Claude prefixes use the Gateway's initial/terminal turn offsets.
            "postprocess_num_turns_recomputed": True,
        }
    )
    logprobs = trajectory.response_logprobs
    return replace(
        trajectory,
        response_ids=list(trajectory.response_ids[:response_end]),
        response_mask=list(trajectory.response_mask[:response_end]),
        response_logprobs=None if logprobs is None else list(logprobs[:response_end]),
        num_turns=2 * kept_spans + 1,
        routed_experts=_crop_routed_experts(trajectory, response_end),
        extra_fields=extra_fields,
    )


def _crop_routed_experts(trajectory: Trajectory, response_end: int):
    """Keep prompt and response-prefix routes; leave terminal alignment to Framework."""
    routed_experts = trajectory.routed_experts
    if routed_experts is None:
        return None
    prefix = routed_experts[: len(trajectory.prompt_ids) + response_end]
    clone = getattr(prefix, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(prefix, "copy", None)
    return copy() if callable(copy) else prefix


def _select_best(trajectory: Trajectory, task_info: Mapping[str, Any]) -> Trajectory | None:
    train_best = task_info.get("train_best")
    if not isinstance(train_best, Mapping):
        return None
    best_index = _as_int_or_none(train_best.get("assistant_index"))
    if best_index is None:
        messages_seen = _as_int_or_none(train_best.get("assistant_messages_seen"))
        best_index = None if messages_seen is None else messages_seen - 1
    spans = assistant_spans(trajectory)
    if best_index is None or not 0 <= best_index < len(spans):
        return None
    return crop_to_assistant_prefix(trajectory, spans[best_index][1], reason="best_assistant")


def _validate_alignment(trajectory: Trajectory) -> None:
    response_len = len(trajectory.response_ids)
    if len(trajectory.response_mask) != response_len:
        raise ValueError(f"response_ids/response_mask misaligned: {response_len} != {len(trajectory.response_mask)}")
    if trajectory.response_logprobs is not None and len(trajectory.response_logprobs) != response_len:
        raise ValueError(
            f"response_ids/response_logprobs misaligned: {response_len} != {len(trajectory.response_logprobs)}"
        )


def _as_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
