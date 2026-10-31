from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf

from examples.multi_policy.framework import MultiAgentGatewayFramework
from uni_agent.framework import framework as generic_module
from uni_agent.framework.framework import _list_of_tq_fields_to_tensordict
from uni_agent.gateway.session import Trajectory
from verl.utils import tensordict_utils as tu


def _framework(policy_processors=None, *, mask_unfinished=False):
    framework = MultiAgentGatewayFramework.__new__(MultiAgentGatewayFramework)
    framework._policy_processors = policy_processors or {"policy-1": None}
    framework._mask_unfinished_episode = mask_unfinished
    return framework


def convert_trajectory(trajectory, *, mask_unfinished=False):
    sample = {"uid": "prompt-1", "raw_prompt": "question"}
    framework = _framework(mask_unfinished=mask_unfinished)
    return framework._trajectory_to_tq_field_and_tag(
        trajectory=trajectory,
        sample_fields=sample,
        session_index=0,
        global_steps=1,
        uid=sample["uid"],
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_extra_fields_preserve_policy_routing_and_reward_metrics():
    trajectory = Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        reward_metrics={"acc": 1.0},
        extra_fields={
            "role": "answerer",
            "policy_name": "policy-1",
            "session_group_id": "rollout-1",
            "reward_extra_info": {"stale": True},
        },
    )
    fields, tags = convert_trajectory(trajectory)

    assert fields["extra_fields"]["reward_extra_info"] == {"acc": 1.0}
    assert fields["extra_fields"]["role"] == "answerer"
    assert not (trajectory.extra_fields.keys() & fields.keys())
    assert tags["policy_name"] == "policy-1"
    assert tags["role"] == "answerer"
    assert tags["session_group_id"] == "rollout-1"
    assert fields["session_id"] == 0

    # Metadata must survive the same TensorDict packing used before TQ writes.
    packed = _list_of_tq_fields_to_tensordict([fields])
    metadata = tu.get(packed[0], "extra_fields")
    assert metadata["reward_extra_info"]["acc"] == 1.0
    fields["extra_fields"]["reward_extra_info"]["acc"] = 0.0
    assert trajectory.reward_metrics == {"acc": 1.0}
    assert trajectory.extra_fields["reward_extra_info"] == {"stale": True}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("logprobs", [None, [-0.25], [0.0], []])
def test_optional_logprobs(logprobs):
    responses = [] if logprobs == [] else [2]
    trajectory = Trajectory(
        prompt_ids=[1],
        response_ids=responses,
        response_mask=[1] * len(responses),
        response_logprobs=logprobs,
        extra_fields={"policy_name": "policy-1"},
    )
    fields, _ = convert_trajectory(trajectory)
    assert ("rollout_log_probs" in fields) == (logprobs is not None)
    if logprobs is not None:
        assert fields["rollout_log_probs"].tolist() == logprobs


@pytest.mark.cpu
@pytest.mark.level0
def test_extra_fields_cannot_override_training_fields():
    trajectory = Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        extra_fields={
            "policy_name": "policy-1",
            "uid": "metadata-uid",
            "response_mask": [0],
            "rollout_log_probs": [0.0],
        },
    )
    fields, _ = convert_trajectory(trajectory)

    assert fields["uid"] == "prompt-1"
    assert fields["response_mask"].tolist() == [1]
    assert "rollout_log_probs" not in fields


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("finished", [None, False, True])
def test_unfinished_masks(enabled, finished):
    trajectory = Trajectory(
        [1],
        [2, 3],
        [1, 0],
        finished=finished,
        extra_fields={"policy_name": "policy-1"},
    )
    fields, _ = convert_trajectory(trajectory, mask_unfinished=enabled)
    expected = [0, 0] if enabled and finished is False else [1, 0]
    assert fields["response_mask"].tolist() == expected
    assert fields["loss_mask"].tolist() == expected
    assert trajectory.response_mask == [1, 0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("finished", [0, "false"])
def test_invalid_finished_is_rejected(finished):
    trajectory = Trajectory([1], [2], [1], extra_fields={"policy_name": "policy-1"})
    trajectory.finished = finished
    with pytest.raises(ValueError, match="finished"):
        convert_trajectory(trajectory)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("version, expected", [(None, 1), (0, 0), (5, 5)])
def test_weight_version_fallback_and_materialization_tag(version, expected):
    trajectory = Trajectory(
        [1],
        [2],
        [1],
        extra_fields={
            "policy_name": "policy-1",
            "min_global_steps": version,
            "max_global_steps": version,
            "materialization_reason": "length",
        },
    )
    _, tags = convert_trajectory(trajectory)
    assert tags["min_global_steps"] == tags["max_global_steps"] == expected
    assert tags["materialization_reason"] == "length"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("shape", [(1, 2, 1), (5, 2, 1), (0, 2, 1), (2, 2)])
def test_routed_experts_alignment(shape):
    source = torch.ones(shape, dtype=torch.int32)
    trajectory = Trajectory(
        [1],
        [2, 3],
        [1, 1],
        routed_experts=source,
        extra_fields={"policy_name": "policy-1"},
    )
    fields, _ = convert_trajectory(trajectory)
    assert ("routed_experts" in fields) == (len(shape) == 3)
    if len(shape) == 3:
        assert fields["routed_experts"].shape == (3, 2, 1)
        assert fields["routed_experts"].device.type == "cpu"
    assert source.shape == shape


@pytest.mark.cpu
@pytest.mark.level0
def test_multimodal_kwargs_and_processor_follow_trajectory_policy(monkeypatch):
    class Processor:
        def __bool__(self):
            return False  # Presence, not truthiness, selects the multimodal branch.

    processors = {"p1": Processor(), "p2": Processor()}
    seen = []

    def multimodal(processor, input_ids, data, kwargs=None):
        seen.append((processor, data, kwargs))
        return {"marker": kwargs}

    def positions(processor, input_ids, attention, multi_modal_inputs):
        assert multi_modal_inputs["marker"] == seen[-1][2]
        return torch.full_like(input_ids, 7)

    monkeypatch.setattr(generic_module, "compute_multi_modal_inputs", multimodal)
    monkeypatch.setattr(generic_module, "compute_position_ids", positions)
    framework = _framework(processors)
    for index, (policy, processor) in enumerate(processors.items()):
        kwargs = {"min_pixels": 16 * (index + 1)}
        data = {"images": [f"image-{index}"]}
        trajectory = Trajectory(
            [1],
            [2],
            [1],
            multi_modal_data=data,
            extra_fields={"policy_name": policy, "mm_processor_kwargs": kwargs},
        )
        fields, tags = framework._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields={"uid": "u", "raw_prompt": "q"},
            session_index=index,
            global_steps=1,
            uid="u",
        )
        assert seen[-1] == (processor, data, kwargs)
        assert fields["position_ids"].tolist() == [7, 7]
        assert tags["policy_name"] == policy


@pytest.mark.cpu
@pytest.mark.level0
def test_trajectory_without_policy_name_is_rejected():
    trajectory = Trajectory([1], [2], [1], extra_fields={})
    with pytest.raises(ValueError, match="policy_name"):
        convert_trajectory(trajectory)


@pytest.mark.cpu
@pytest.mark.level0
def test_unknown_policy_processor_is_rejected():
    trajectory = Trajectory([1], [2], [1], extra_fields={"policy_name": "missing"})
    with pytest.raises(ValueError, match="No processor configured for policy 'missing'"):
        convert_trajectory(trajectory)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("kwargs", [[], "invalid"])
def test_invalid_multimodal_kwargs_are_rejected(kwargs):
    trajectory = Trajectory(
        [1],
        [2],
        [1],
        extra_fields={"policy_name": "policy-1", "mm_processor_kwargs": kwargs},
    )
    with pytest.raises(ValueError, match="mm_processor_kwargs"):
        convert_trajectory(trajectory)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("setting", [None, False, True, "false", 1])
def test_mask_config_is_shared_and_requires_boolean(setting):
    af_config = {
        "agent_runners": {"multi_agent": {"runner_fqn": "unused", "dispatch_mode": "ray_task"}},
        "role_policy_mapping": {"answer": "policy-1"},
    }
    if setting is not None:
        af_config["mask_unfinished_episode"] = setting
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"custom": {"agent_framework": af_config}}},
        }
    )
    kwargs = dict(
        config=config,
        gateway_manager=SimpleNamespace(),
        policy_rollout_configs={},
        policy_processors={"policy-1": None},
    )
    if setting is not None and type(setting) is not bool:
        with pytest.raises(ValueError, match="mask_unfinished_episode"):
            MultiAgentGatewayFramework.from_config(**kwargs)
    else:
        framework = MultiAgentGatewayFramework.from_config(**kwargs)
        assert framework._mask_unfinished_episode is (setting is True)


@pytest.mark.cpu
@pytest.mark.level0
def test_num_turns_is_explicitly_long():
    trajectory = Trajectory([1], [2], [1], extra_fields={"policy_name": "policy-1"})
    trajectory.num_turns = 2.0
    fields, _ = convert_trajectory(trajectory)
    assert fields["num_turns"].dtype == torch.long


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_writer_uses_explicit_uid_for_record_key_and_tag(monkeypatch):
    framework = _framework()
    write = AsyncMock()
    monkeypatch.setattr(generic_module, "tq", SimpleNamespace(async_kv_batch_put=write))
    trajectory = Trajectory([1], [2], [1], extra_fields={"policy_name": "policy-1"})
    await framework._write_session_trajectories_to_tq(
        uid="record-owner",
        session_index=2,
        trajectories=[trajectory],
        sample_fields={"uid": "sample-metadata", "raw_prompt": "question"},
        global_steps=1,
        partition_id="train",
    )
    written = write.call_args.kwargs
    assert written["keys"] == ["record-owner_2_0"]
    assert written["tags"][0]["uid"] == "record-owner"
    # Like the native framework, sample fields remain the supplied metadata.
    assert tu.get(written["fields"][0], "uid") == "sample-metadata"
