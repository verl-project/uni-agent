"""Trajectory selection: config parsing, function loading, execution.

Split out of ``framework.py`` so the policy can be tested without standing up a
Gateway session. The Framework owns the session lifecycle; this module owns what
happens to the trajectories once finalization has produced them.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from omegaconf import OmegaConf

from uni_agent.gateway.session import Trajectory
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


TrajectorySelectFunction = Callable[[str, list[Trajectory]], list[Trajectory] | Awaitable[list[Trajectory]]]


@dataclass(frozen=True)
class _CustomSelectFunction:
    """``{path, name}`` reference to a user-owned select function."""

    path: str
    name: str


def _parse_trajectory_selection(
    runner_name: object,
    runner_cfg,
) -> tuple[str, _CustomSelectFunction | None]:
    """Decode ``trajectory_selection`` into (selection, custom_select_function)."""
    # A top-level key would be silently ignored by the all/longest path.
    if "custom_select_function" in runner_cfg:
        raise ValueError(
            f"agent_runners.{runner_name}: 'custom_select_function' must be nested under "
            "'trajectory_selection' (mapping form), not a top-level runner key"
        )

    raw = runner_cfg.get("trajectory_selection", "all")
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, Mapping):
        csf = OmegaConf.to_container(OmegaConf.create(raw.get("custom_select_function", {})), resolve=True) or {}
        if not isinstance(csf, Mapping):
            raise ValueError(
                f"agent_runners.{runner_name}: trajectory_selection.custom_select_function must be a mapping, "
                f"got {type(csf).__name__}"
            )
        # The mapping form keeps the neutral value: select_fn decides the mode.
        return "all", _CustomSelectFunction(
            path=str(csf.get("path") or ""),
            name=str(csf.get("name") or ""),
        )
    raise ValueError(
        f"agent_runners.{runner_name}: trajectory_selection must be 'all'/'longest' or a mapping with "
        f"custom_select_function.{{path, name}}, got {type(raw).__name__}"
    )


def _load_select_function(custom: _CustomSelectFunction) -> TrajectorySelectFunction:
    """Resolve a ``{path, name}`` reference to a callable, failing fast at start-up."""
    try:
        fn = load_extern_object(module_path=custom.path, object_name=custom.name)
    except Exception as exc:
        raise ValueError(f"cannot load select function {custom.path}.{custom.name}: {exc}") from exc
    if not callable(fn):
        raise TypeError(f"select function {custom.path}.{custom.name} is not callable")
    return fn


async def _select_session_trajectories(
    session_id: str,
    trajectories: list[Trajectory],
    selection: str,
    *,
    select_fn: TrajectorySelectFunction | None = None,
) -> list[Trajectory]:
    """Apply the runner's trajectory-retention policy before scoring and TQ writes."""
    if select_fn is None:
        if selection == "all" or len(trajectories) <= 1:
            return trajectories

        index, trajectory = max(
            enumerate(trajectories),
            key=lambda item: (
                sum(item[1].response_mask),
                len(item[1].response_ids),
                item[1].num_turns,
                item[0],
            ),
        )
        logger.info(
            "session %s: selected longest trajectory index=%s model_tokens=%s "
            "response_tokens=%s turns=%s candidates=%s",
            session_id,
            index,
            sum(trajectory.response_mask),
            len(trajectory.response_ids),
            trajectory.num_turns,
            len(trajectories),
        )
        return [trajectory]

    # Custom mode skips the len<=1 short-circuit: a single trajectory may be the one to drop.
    # This is what the session keeps if the function mutates the list and then raises.
    fallback_trajectories = list(trajectories)
    try:
        kept: list[Trajectory] | Awaitable[list[Trajectory]] = select_fn(session_id, trajectories)
        if inspect.isawaitable(kept):
            kept = await kept
    except Exception:
        logger.exception("session %s: custom select raised; falling back to 'all'", session_id)
        kept = fallback_trajectories
    return _finalize_custom_select(session_id, fallback_trajectories, kept)


def _finalize_custom_select(
    session_id: str,
    fallback_trajectories: list[Trajectory],
    kept: list[Trajectory],
) -> list[Trajectory]:
    """Normalise the user's return value and release message text."""
    if not isinstance(kept, list):
        kept = list(kept)

    # Before reward scoring, artifact logging and the TQ write, so the text reaches
    # none of them. ``kept`` too: it may hold copies the input list no longer reaches.
    for trajectory in [*fallback_trajectories, *kept]:
        trajectory.extra_fields.pop("message_history", None)

    if not kept:
        logger.warning("session %s: custom select returned no trajectories -> status=empty", session_id)
    else:
        logger.info("session %s: custom select kept=%s/%s", session_id, len(kept), len(fallback_trajectories))
    return kept
