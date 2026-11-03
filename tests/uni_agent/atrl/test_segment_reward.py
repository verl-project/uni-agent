from uni_agent.atrl.segment_reward import combined_segment_rewards


def test_no_compaction_keeps_the_final_reward():
    assert combined_segment_rewards([], 1.3, alpha=0.5) == [1.3]


def test_mixed_reward_uses_boundary_delta():
    # boundaries: 0 -> 0.2 -> 1.0, alpha 0.5
    assert combined_segment_rewards([0.2], 1.0, alpha=0.5) == [0.6, 0.9]
