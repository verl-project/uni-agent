"""Plain trajectory selector for the multi_policy recipe.

The multi_policy framework wraps custom select functions so each role's
trajectories are passed separately. A selector therefore receives one role's
list and can stay role-agnostic, exactly like a single-agent selector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uni_agent.gateway.session import Trajectory


def longest(session_id: str, trajectories: list[Trajectory]) -> list[Trajectory]:
    """Keep the longest trajectory in the list.

    Uses the same tie-breakers as the framework's built-in ``longest``: model
    tokens, response length, turns, then list position.
    """
    if len(trajectories) <= 1:
        return list(trajectories)
    _, trajectory = max(
        enumerate(trajectories),
        key=lambda item: (
            sum(item[1].response_mask),
            len(item[1].response_ids),
            item[1].num_turns,
            item[0],
        ),
    )
    return [trajectory]
