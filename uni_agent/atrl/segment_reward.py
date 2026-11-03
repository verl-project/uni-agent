"""Mixed segment reward R̃_i = α R + (1-α) (reward_end - reward_start)."""

from __future__ import annotations


def combined_segment_rewards(
    boundary_rewards: list[float],
    final_reward: float,
    *,
    alpha: float = 0.5,
) -> list[float]:
    """Return one combined reward per segment.

    ``boundary_rewards`` are scores at the end of each compacted segment, in
    order. The final segment ends at ``final_reward``. The trajectory starts
    at reward 0, so with no compaction the only segment scores ``final_reward``.
    """
    ends = [float(reward) for reward in boundary_rewards]
    boundaries = [0.0, *ends, float(final_reward)]
    combined: list[float] = []
    for index in range(len(boundaries) - 1):
        delta = boundaries[index + 1] - boundaries[index]
        combined.append(round(alpha * float(final_reward) + (1.0 - alpha) * delta, 6))
    return combined
