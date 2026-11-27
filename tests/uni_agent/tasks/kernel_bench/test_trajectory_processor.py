import pytest

from examples.claude_code_kernel_task.trajectory_processor import crop_to_assistant_prefix
from uni_agent.gateway.session.types import Trajectory


@pytest.mark.cpu
@pytest.mark.level0
def test_crop_to_assistant_prefix_retains_only_generations_within_the_prefix():
    retained = {"generation_index": 0, "response_start": 0, "response_end": 2, "tool_call_ids": ["call_a"]}
    removed = {"generation_index": 1, "response_start": 3, "response_end": 4, "tool_call_ids": ["call_b"]}
    removed_empty = {"generation_index": 2, "response_start": 4, "response_end": 4, "tool_call_ids": ["call_c"]}
    original_spans = [retained, removed, removed_empty]
    trajectory = Trajectory(
        prompt_ids=[1],
        response_ids=[2, 3, 4, 5],
        response_mask=[1, 1, 0, 1],
        response_logprobs=[-0.1, -0.2, 0.0, -0.3],
        reward_score=1.0,
        extra_fields={"generation_spans": original_spans, "min_global_steps": 2, "metadata": "retained"},
    )

    cropped = crop_to_assistant_prefix(trajectory, 2, reason="best_assistant")

    assert cropped.response_ids == [2, 3]
    assert cropped.response_mask == [1, 1]
    assert cropped.response_logprobs == [-0.1, -0.2]
    assert cropped.reward_score == trajectory.reward_score
    assert cropped.extra_fields["generation_spans"] == [retained]
    assert cropped.extra_fields["metadata"] == "retained"
    assert "min_global_steps" not in cropped.extra_fields
    assert trajectory.extra_fields["generation_spans"] == [retained, removed, removed_empty]
