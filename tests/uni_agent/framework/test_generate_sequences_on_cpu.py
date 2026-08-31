from __future__ import annotations

import asyncio
import logging
import types
from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.uni_agent.support import logging_runner
from uni_agent.framework.framework import GatewayAgentFramework, _align_routed_experts
from uni_agent.gateway.session import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu

_RUNNER_CALLS = []
_TEST_INLINE_RUNNERS = {}
_POSTPROCESSOR_CALLS = []
_NOT_CALLABLE_POSTPROCESSOR = 42


def _recording_trajectory_postprocessor(trajectories, *, policy=None):
    _POSTPROCESSOR_CALLS.append((trajectories, policy))
    return list(reversed(trajectories))


async def _async_trajectory_postprocessor(trajectories):
    await asyncio.sleep(0)
    return list(trajectories[-1:])


def _empty_trajectory_postprocessor(_trajectories):
    return []


def _tuple_trajectory_postprocessor(trajectories):
    return tuple(trajectories)


def _invalid_item_trajectory_postprocessor(trajectories):
    return ["not-a-trajectory"]


def _dropping_reward_info_postprocessor(trajectories):
    return [replace(trajectories[-1], reward_info={})]


async def _config_recording_runner(*, raw_prompt, session, sample_index, marker=None, **kwargs):
    _RUNNER_CALLS.append(
        {
            "runner": marker,
            "raw_prompt": raw_prompt,
            "session_id": session.session_id,
            "base_url": session.base_url,
            "reward_info_url": session.reward_info_url,
            "sample_index": sample_index,
            "kwargs": dict(kwargs),
        }
    )


class _ConfigRecordingClassRunner:
    def __init__(self, marker=None):
        self.marker = marker

    async def __call__(self, *, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        _RUNNER_CALLS.append(
            {
                "runner": self.marker,
                "raw_prompt": raw_prompt,
                "session_id": session.session_id,
                "base_url": session.base_url,
                "reward_info_url": session.reward_info_url,
                "sample_index": sample_index,
                "kwargs": {**dict(kwargs), "tools_kwargs": tools_kwargs},
            }
        )


async def _async_noop_runner(**kwargs):
    return None


async def _inline_runner_proxy(*, runner_key, **kwargs):
    runner = _TEST_INLINE_RUNNERS[runner_key]
    await runner(**kwargs)


def _inline_runner_config(
    runner,
    *,
    dispatch_mode: str = "inline_async",
    trajectory_selection: str | None = None,
) -> dict[str, object]:
    runner_key = f"runner-{len(_TEST_INLINE_RUNNERS)}"
    _TEST_INLINE_RUNNERS[runner_key] = runner
    config = {
        "runner_fqn": f"{__name__}._inline_runner_proxy",
        "runner_kwargs": {"runner_key": runner_key},
        "dispatch_mode": dispatch_mode,
    }
    if trajectory_selection is not None:
        config["trajectory_selection"] = trajectory_selection
    return config


async def _build_framework_with_agent_runners(
    *,
    agent_runners: dict[str, dict[str, object]],
    gateway_manager,
    reward_loop_worker_handles=None,
    n: int = 1,
    val_n: int = 1,
    log_dir: str | None = None,
    mask_unfinished_episode: bool = False,
    trajectory_postprocessor_fqn: str | None = None,
    trajectory_postprocessor_kwargs: object | None = None,
):
    from omegaconf import OmegaConf

    agent_framework_cfg: dict[str, object] = {
        "agent_runners": agent_runners,
        "mask_unfinished_episode": mask_unfinished_episode,
    }
    if log_dir is not None:
        agent_framework_cfg["log_dir"] = log_dir
    if trajectory_postprocessor_fqn is not None:
        agent_framework_cfg["trajectory_postprocessor_fqn"] = trajectory_postprocessor_fqn
    if trajectory_postprocessor_kwargs is not None:
        agent_framework_cfg["trajectory_postprocessor_kwargs"] = trajectory_postprocessor_kwargs

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": n,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "calculate_log_probs": True,
                    "val_kwargs": {
                        "n": val_n,
                        "temperature": 0,
                        "top_p": 0.95,
                        "top_k": -1,
                    },
                    "custom": {"agent_framework": agent_framework_cfg},
                }
            }
        }
    )
    return GatewayAgentFramework.from_config(
        config=config,
        gateway_manager=gateway_manager,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    (
        "data_config",
        "agent_framework_config",
        "expected_rollback",
        "expected_cache",
        "expected_chat_template_kwargs",
    ),
    [
        ({}, {}, True, True, {}),
        (
            {"apply_chat_template_kwargs": {"thinking": True}},
            {"enable_last_assistant_rollback": False},
            False,
            True,
            {"thinking": True},
        ),
        ({}, {"enable_tool_parser_cache": False}, True, False, {}),
    ],
)
def test_build_gateway_manager_wires_gateway_config_defaults(
    monkeypatch,
    data_config,
    agent_framework_config,
    expected_rollback,
    expected_cache,
    expected_chat_template_kwargs,
):
    from omegaconf import OmegaConf

    from uni_agent.framework import entry as entry_module

    class _ModelConfig:
        tokenizer = object()
        processor = None

    captured = {}

    class _FakeGatewayManager:
        def __init__(self, *, llm_client, gateway_count, gateway_actor_config):
            captured["llm_client"] = llm_client
            captured["gateway_count"] = gateway_count
            captured["gateway_actor_config"] = gateway_actor_config

    monkeypatch.setattr(entry_module, "omega_conf_to_dataclass", lambda _config: _ModelConfig())
    monkeypatch.setattr(entry_module, "GatewayManager", _FakeGatewayManager)

    llm_client = object()
    config = OmegaConf.create(
        {
            "data": data_config,
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "name": "vllm",
                    "prompt_length": 128,
                    "response_length": 64,
                    "multi_turn": {"format": "hermes"},
                    "custom": {
                        "agent_framework": {
                            "gateway_count": 2,
                            **agent_framework_config,
                        }
                    },
                },
            },
        }
    )

    manager = entry_module.build_gateway_manager(config=config, llm_client=llm_client)

    assert isinstance(manager, _FakeGatewayManager)
    assert captured["llm_client"] is llm_client
    assert captured["gateway_count"] == 2
    assert captured["gateway_actor_config"].prompt_length == 128
    assert captured["gateway_actor_config"].response_length == 64
    assert captured["gateway_actor_config"].tool_parser_name == "hermes"
    assert captured["gateway_actor_config"].rollout_backend == "vllm"
    assert captured["gateway_actor_config"].enable_last_assistant_rollback is expected_rollback
    assert captured["gateway_actor_config"].enable_tool_parser_cache is expected_cache
    assert isinstance(captured["gateway_actor_config"].apply_chat_template_kwargs, dict)
    assert captured["gateway_actor_config"].apply_chat_template_kwargs == expected_chat_template_kwargs


class _FakeTransferQueue:
    def __init__(self):
        self.puts = []
        self.batch_puts = []

    async def async_kv_put(self, *, key, partition_id, tag):
        self.puts.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.batch_puts.append(
            {
                "keys": list(keys),
                "fields": fields,
                "tags": [dict(tag) for tag in tags],
                "partition_id": partition_id,
            }
        )


@pytest.fixture
def fake_tq(monkeypatch):
    from uni_agent.framework import framework as framework_module

    fake = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake)
    return fake


class _FakeGatewayManager:
    """Fake runtime that matches session IDs by prefix (``session-sample-{sample}-rollout-{rollout}``)
    to support the real uuid-suffixed IDs produced by the framework."""

    def __init__(self, finalized_by_session_prefix: dict[str, list[Trajectory]]):
        self._finalized_by_prefix = finalized_by_session_prefix
        self.created_sessions = []
        self.created_session_kwargs = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    def _lookup(self, session_id: str) -> list[Trajectory]:
        for prefix, trajectories in self._finalized_by_prefix.items():
            if session_id.startswith(prefix):
                return trajectories
        raise KeyError(f"No prefix match for session_id={session_id}")

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        self.created_session_kwargs.append(dict(kwargs))
        return SessionHandle(
            session_id=session_id,
            base_url=f"http://fake/{session_id}/v1",
            reward_info_url=f"http://fake/{session_id}/reward_info",
        )

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self._lookup(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)


def _build_prompts(
    count: int = 2,
    *,
    global_steps: int | None = 7,
    validate: bool | None = None,
    do_sample: bool | None = None,
):
    non_tensor_dict = {"global_steps": global_steps}
    if validate is not None:
        non_tensor_dict["validate"] = validate
    tensor_dict = {
        "raw_prompt": [[{"role": "user", "content": f"sample {i}"}] for i in range(count)],
        "uid": [f"uid-{i}" for i in range(count)],
        "data_source": ["deepeyes"] * count,
        "reward_model": [{"ground_truth": f"answer-{i}"} for i in range(count)],
        "extra_info": [{"index": i} for i in range(count)],
        "tools_kwargs": [{"tool": i} for i in range(count)],
        "agent_name": ["deepeyes"] * count,
    }
    if do_sample is not None:
        tensor_dict["__do_sample__"] = [do_sample] * count
    return tu.get_tensordict(
        tensor_dict=tensor_dict,
        non_tensor_dict=non_tensor_dict,
    )


def _trajectory(
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    response_mask: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    reward_info: dict[str, object] | None = None,
    num_turns: int = 2,
    routed_experts: object | None = None,
    extra_fields: dict[str, object] | None = None,
):
    prompt_ids = prompt_ids or [10, 11]
    response_ids = response_ids or [20, 21]
    response_mask = response_mask if response_mask is not None else [1] * len(response_ids)
    return Trajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        reward_info=dict(reward_info or {}),
        reward_score=None,
        num_turns=num_turns,
        routed_experts=routed_experts,
        multi_modal_data={"images": ["raw-image-should-not-be-written"]},
        extra_fields=dict(extra_fields or {}),
    )


def _install_fake_score(monkeypatch, *, score_from_sample_fields=None, default_score=1.0):
    """Replace GatewayAgentFramework._score_trajectories with a fake.

    Keeps ``generate_sequences`` tests focused on TQ output by returning the
    same deterministic score for every trajectory in the session.
    """
    from uni_agent.framework.framework import GatewayAgentFramework

    async def fake_score(self, trajectories, sample_fields):
        if score_from_sample_fields is not None:
            score = float(score_from_sample_fields(sample_fields))
        else:
            score = float(default_score)
        return [(score, {})] * len(trajectories)

    monkeypatch.setattr(GatewayAgentFramework, "_score_trajectories", fake_score)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_agent_runners_registry_materializes_runners_and_selects_by_agent_name(fake_tq):
    """Function and class runners keep per-runner kwargs, and each prompt's
    ``agent_name`` selects the matching runner without leaking internals."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [_trajectory()],
            "session-sample-1-rollout-0": [_trajectory()],
        }
    )
    _RUNNER_CALLS.clear()
    runner_fqn = f"{__name__}._config_recording_runner"
    prompts = _build_prompts(count=2, global_steps=6)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["deepeyes", "swe"]},
        non_tensor_dict={},
    )["agent_name"]

    framework = await _build_framework_with_agent_runners(
        agent_runners={
            "deepeyes": {
                "runner_fqn": runner_fqn,
                "runner_kwargs": {"marker": "deepeyes"},
                "dispatch_mode": "inline_async",
            },
            "swe": {
                "runner_fqn": f"{__name__}._ConfigRecordingClassRunner",
                "runner_kwargs": {"marker": "swe"},
                "dispatch_mode": "inline_async",
            },
        },
        gateway_manager=runtime,
    )

    await framework.generate_sequences(prompts)

    calls = sorted(_RUNNER_CALLS, key=lambda call: call["sample_index"])
    assert [call["runner"] for call in calls] == ["deepeyes", "swe"]
    assert [call["raw_prompt"] for call in calls] == [
        [{"role": "user", "content": "sample 0"}],
        [{"role": "user", "content": "sample 1"}],
    ]
    assert all(call["base_url"].endswith("/v1") for call in calls)
    assert all(call["reward_info_url"].endswith("/reward_info") for call in calls)
    assert [call["sample_index"] for call in calls] == [0, 1]
    runner_tools_kwargs = [
        {key: value for key, value in call["kwargs"]["tools_kwargs"].items() if key != "_trace_identity"}
        for call in calls
    ]
    assert runner_tools_kwargs == [{"tool": 0}, {"tool": 1}]
    assert all("gateway_manager" not in call["kwargs"] for call in calls)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("dispatch_mode", ["inline_async", "ray_task"])
async def test_framework_and_runner_logs_share_one_session_directory(tmp_path, fake_tq, dispatch_mode):
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    runner_config = (
        _inline_runner_config(logging_runner)
        if dispatch_mode == "inline_async"
        else {
            "runner_fqn": "tests.uni_agent.support.logging_runner",
            "dispatch_mode": "ray_task",
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": runner_config},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=12))

    step_dir = tmp_path / "step_12"
    session_dirs = list(step_dir.iterdir())
    assert len(session_dirs) == 1
    assert session_dirs[0].name.startswith("session-sample-0-rollout-0-")

    framework_log = session_dirs[0] / "framework.log"
    task_log = session_dirs[0] / "task.log"
    for _ in range(50):
        if task_log.exists() and "runner task log" in task_log.read_text():
            parent_log = framework_log if dispatch_mode == "ray_task" else task_log
            if parent_log.exists() and "session session-sample-0-rollout-0-" in parent_log.read_text():
                break
        await asyncio.sleep(0.02)

    assert "runner task log" in task_log.read_text()
    if dispatch_mode == "ray_task":
        assert "session session-sample-0-rollout-0-" in framework_log.read_text()
    else:
        assert not framework_log.exists()
        assert "session session-sample-0-rollout-0-" in task_log.read_text()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_validation_logs_omit_global_step_directory(tmp_path, fake_tq):
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(logging_runner)},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=None, validate=True))

    session_dirs = list(tmp_path.iterdir())
    assert len(session_dirs) == 1
    assert session_dirs[0].name.startswith("session-sample-0-rollout-0-")
    assert (session_dirs[0] / "task.log").exists()

    batch = fake_tq.batch_puts[0]
    assert batch["partition_id"] == "val"
    assert batch["tags"][0]["global_steps"] is None
    assert tu.get(batch["fields"], "global_steps") == [None]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_validation_logs_keep_provided_global_step(tmp_path, fake_tq):
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=12, validate=True))

    session_dirs = list((tmp_path / "step_12").iterdir())
    assert len(session_dirs) == 1
    assert session_dirs[0].name.startswith("session-sample-0-rollout-0-")

    batch = fake_tq.batch_puts[0]
    assert batch["partition_id"] == "val"
    assert batch["tags"][0]["global_steps"] == 12
    assert tu.get(batch["fields"], "global_steps") == [12]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_training_requires_global_steps(fake_tq):
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=_FakeGatewayManager({}),
    )

    with pytest.raises(ValueError, match=r"prompts\['global_steps'\] for training"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=None))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("validate", "do_sample", "expected_sampling_params"),
    [
        (
            False,
            None,
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "logprobs": True,
            },
        ),
        (
            True,
            None,
            {
                "temperature": 0,
                "top_p": 0.95,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "logprobs": True,
            },
        ),
        (
            False,
            False,
            {
                "temperature": 0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "logprobs": True,
            },
        ),
    ],
)
async def test_framework_binds_sampling_defaults_to_gateway_sessions(
    fake_tq,
    validate,
    do_sample,
    expected_sampling_params,
):
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(
        _build_prompts(
            count=1,
            validate=validate,
            do_sample=do_sample,
        )
    )

    assert [kwargs["sampling_params"] for kwargs in runtime.created_session_kwargs] == [expected_sampling_params]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_writes_tq_schema_for_each_session(monkeypatch, fake_tq):
    """Full ``generate_sequences`` path writes one TQ batch per successful
    session and trainer-compatible trajectory fields."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_logprobs=[-0.1, -0.2],
                    routed_experts=np.array(
                        [
                            [[0, 1], [2, 3]],
                            [[4, 5], [6, 7]],
                            [[8, 9], [10, 11]],
                        ],
                        dtype=np.uint8,
                    ),
                    extra_fields={"materialization_reason": "max_trajectory_length"},
                )
            ],
            "session-sample-0-rollout-1": [_trajectory(response_logprobs=[-0.3, -0.4])],
        }
    )

    # Nonzero score proves reward_score lands on the final response token.
    _install_fake_score(
        monkeypatch,
        score_from_sample_fields=lambda sf: sf["extra_info"]["index"] + 0.25,
    )

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        reward_loop_worker_handles=["sentinel"],
        n=2,
        val_n=2,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.batch_puts[1]["keys"] == ["uid-0_1_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]

    first = fake_tq.batch_puts[0]
    fields = first["fields"]
    assert first["partition_id"] == "train"
    tag = first["tags"][0]
    assert {
        key: tag[key]
        for key in (
            "global_steps",
            "status",
            "prompt_len",
            "response_len",
            "seq_len",
            "uid",
            "materialization_reason",
        )
    } == {
        "global_steps": 7,
        "status": "success",
        "prompt_len": 2,
        "response_len": 2,
        "seq_len": 4,
        "uid": "uid-0",
        "materialization_reason": "max_trajectory_length",
    }
    assert "finished" not in tag
    assert "length_truncated" not in tag
    assert "traj_exit_reason" not in tag
    assert "materialization_reason" not in fields
    # No gateway-reported weight version, so both fall back to the dataloader step.
    assert (tag["min_global_steps"], tag["max_global_steps"]) == (7, 7)
    assert fields["input_ids"].is_nested
    assert fields["response_mask"].is_nested
    assert fields["position_ids"].is_nested
    assert fields["routed_experts"].is_nested
    assert fields["routed_experts"].dtype == torch.uint8
    assert fields["prompts"][0].tolist() == [10, 11]
    assert fields["responses"][0].tolist() == [20, 21]
    assert fields["response_mask"][0].tolist() == [1, 1]
    assert fields["loss_mask"][0].tolist() == [1, 1]
    assert fields["input_ids"][0].tolist() == [10, 11, 20, 21]
    assert fields["attention_mask"][0].tolist() == [1, 1, 1, 1]
    assert fields["position_ids"][0].tolist() == [0, 1, 2, 3]
    assert fields["routed_experts"][0].tolist() == [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9], [10, 11]],
        [[0, 0], [0, 0]],
    ]
    assert fields["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2])
    assert fields["rm_scores"][0].tolist() == [0.0, 0.25]
    assert tu.get(fields, "multi_modal_inputs") == [{}]
    assert tu.get(fields, "uid") == ["uid-0"]
    assert tu.get(fields, "raw_prompt") == [[{"role": "user", "content": "sample 0"}]]
    assert tu.get(fields, "data_source") == ["deepeyes"]
    assert tu.get(fields, "reward_model") == [{"ground_truth": "answer-0"}]
    assert tu.get(fields, "extra_info") == [{"index": 0}]
    assert tu.get(fields, "tools_kwargs") == [{"tool": 0}]
    assert tu.get(fields, "agent_name") == ["deepeyes"]
    assert tu.get(fields, "session_id") == [0]
    assert tu.get(fields, "global_steps") == [7]
    assert fields["num_turns"].tolist() == [2]
    assert "multi_modal_data" not in fields.keys()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_masks_unfinished_trajectory_without_dropping_it(fake_tq):
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_ids=[20, 21, 22],
                    response_mask=[1, 0, 1],
                    reward_info={"reward": 0.5, "finished": False},
                    extra_fields={
                        "response_mask": torch.ones(3, dtype=torch.long),
                        "loss_mask": torch.ones(3, dtype=torch.long),
                    },
                )
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        mask_unfinished_episode=True,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    batch = fake_tq.batch_puts[0]
    assert batch["keys"] == ["uid-0_0_0"]
    assert batch["fields"]["responses"][0].tolist() == [20, 21, 22]
    assert batch["fields"]["response_mask"][0].tolist() == [0, 0, 0]
    assert batch["fields"]["loss_mask"][0].tolist() == [0, 0, 0]
    assert batch["fields"]["rm_scores"][0].tolist() == [0.0, 0.0, 0.5]
    assert batch["tags"][0]["status"] == "success"
    assert "finished" not in batch["tags"][0]
    assert "finished" not in batch["fields"].keys()
    assert tu.get(batch["fields"], "reward_extra_info") == [{}]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_unfinished_trajectory_remains_trainable_when_masking_is_disabled(fake_tq):
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_ids=[20, 21],
                    response_mask=[1, 1],
                    reward_info={"reward": 0.5, "finished": False},
                )
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    batch = fake_tq.batch_puts[0]
    assert batch["fields"]["response_mask"][0].tolist() == [1, 1]
    assert batch["fields"]["loss_mask"][0].tolist() == [1, 1]
    assert "finished" not in batch["tags"][0]
    assert "finished" not in batch["fields"].keys()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_masking_keeps_trajectory_trainable_when_completion_metadata_is_missing(fake_tq):
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_ids=[20, 21],
                    response_mask=[1, 0],
                    reward_info={"reward": 0.5},
                )
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        mask_unfinished_episode=True,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    batch = fake_tq.batch_puts[0]
    assert batch["fields"]["response_mask"][0].tolist() == [1, 0]
    assert batch["fields"]["loss_mask"][0].tolist() == [1, 0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_reports_unfinished_episode_count(fake_tq, caplog):
    # A session materializing two trajectories is still one episode: completion is
    # session-level metadata copied onto every trajectory it produced.
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(reward_info={"reward": 0.5, "finished": False}),
                _trajectory(reward_info={"reward": 0.5, "finished": False}),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        mask_unfinished_episode=True,
    )

    with caplog.at_level(logging.INFO, logger="uni_agent.framework.framework"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    assert "num_success_outputs=2" in caplog.text
    assert "num_unfinished_episodes=1" in caplog.text


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_framework_rejects_non_boolean_masking_config():
    with pytest.raises(ValueError, match="mask_unfinished_episode must be a bool"):
        await _build_framework_with_agent_runners(
            agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
            gateway_manager=_FakeGatewayManager({}),
            mask_unfinished_episode="true",  # type: ignore[arg-type]
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_align_routed_experts_preserves_backend_dtype():
    aligned = _align_routed_experts(np.array([[[256, 511]]], dtype=np.uint16), seq_len=2)

    assert aligned is not None
    assert aligned.dtype == torch.uint16
    assert aligned.tolist() == [[[256, 511]], [[0, 0]]]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_batches_length_trajectory_before_normal_trajectory(fake_tq):
    """Keep length metadata in tags when mixed trajectories share one TQ batch."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_ids=[20],
                    extra_fields={"materialization_reason": "max_trajectory_length"},
                ),
                _trajectory(response_ids=[21]),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert len(fake_tq.batch_puts) == 1
    batch = fake_tq.batch_puts[0]
    assert batch["keys"] == ["uid-0_0_0", "uid-0_0_1"]
    assert batch["tags"][0]["materialization_reason"] == "max_trajectory_length"
    assert "materialization_reason" not in batch["tags"][1]
    assert "materialization_reason" not in batch["fields"].keys()
    assert batch["fields"]["responses"][0].tolist() == [20]
    assert batch["fields"]["responses"][1].tolist() == [21]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_selects_longest_model_token_trajectory(fake_tq):
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(
                    response_ids=[20, 21, 22, 23, 24, 25],
                    response_mask=[1, 0, 0, 0, 0, 0],
                    num_turns=10,
                ),
                _trajectory(
                    response_ids=[30, 31, 32],
                    response_mask=[1, 1, 1],
                    num_turns=2,
                ),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={
            "runner": _inline_runner_config(
                _async_noop_runner,
                trajectory_selection="longest",
            )
        },
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert len(fake_tq.batch_puts) == 1
    batch = fake_tq.batch_puts[0]
    assert batch["keys"] == ["uid-0_0_0"]
    assert batch["fields"]["responses"][0].tolist() == [30, 31, 32]
    assert batch["fields"]["response_mask"][0].tolist() == [1, 1, 1]
    assert batch["fields"]["num_turns"].tolist() == [2]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_framework_rejects_unknown_trajectory_selection(fake_tq):
    with pytest.raises(ValueError, match="Unknown trajectory selection"):
        await _build_framework_with_agent_runners(
            agent_runners={
                "runner": _inline_runner_config(
                    _async_noop_runner,
                    trajectory_selection="shortest",
                )
            },
            gateway_manager=_FakeGatewayManager({}),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_trajectory_postprocessor_applies_kwargs_before_scoring_and_tq(monkeypatch, fake_tq):
    _POSTPROCESSOR_CALLS.clear()
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[20]),
                _trajectory(response_ids=[30]),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        trajectory_postprocessor_fqn=f"{__name__}._recording_trajectory_postprocessor",
        trajectory_postprocessor_kwargs={"policy": {"thresholds": [1, 2]}},
    )

    scored_responses = []

    def score_processed_trajectories(trajectories):
        scored_responses.extend(trajectory.response_ids for trajectory in trajectories)
        return [(0.5, {}) for _ in trajectories]

    monkeypatch.setattr(framework, "_score_from_reward_info", score_processed_trajectories)

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert len(_POSTPROCESSOR_CALLS) == 1
    processed_input, policy = _POSTPROCESSOR_CALLS[0]
    assert isinstance(processed_input, tuple)
    assert policy == {"thresholds": [1, 2]}
    assert scored_responses == [[30], [20]]

    fields = fake_tq.batch_puts[0]["fields"]
    assert [response.tolist() for response in fields["responses"]] == [[30], [20]]
    assert [score.tolist() for score in fields["rm_scores"]] == [[0.5], [0.5]]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_async_trajectory_postprocessor_is_awaited(fake_tq):
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[20]),
                _trajectory(response_ids=[30]),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        trajectory_postprocessor_fqn=f"{__name__}._async_trajectory_postprocessor",
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.batch_puts[0]["fields"]["responses"][0].tolist() == [30]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("postprocessor_name", "error_type", "error_message"),
    [
        ("_tuple_trajectory_postprocessor", TypeError, r"must return list\[Trajectory\], got tuple"),
        ("_invalid_item_trajectory_postprocessor", TypeError, "returned a non-Trajectory item"),
    ],
)
async def test_trajectory_postprocessor_reports_invalid_extensions(
    postprocessor_name,
    error_type,
    error_message,
):
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=_FakeGatewayManager({}),
        trajectory_postprocessor_fqn=f"{__name__}.{postprocessor_name}",
    )

    with pytest.raises(error_type, match=error_message):
        await framework._apply_trajectory_postprocessor([_trajectory()])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_trajectory_postprocessor_rejects_dropped_reward_info():
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=_FakeGatewayManager({}),
        trajectory_postprocessor_fqn=f"{__name__}._dropping_reward_info_postprocessor",
    )

    with pytest.raises(ValueError, match="must preserve finalized reward_info"):
        await framework._apply_trajectory_postprocessor([_trajectory(reward_info={"reward": 0.5, "finished": False})])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_framework_rejects_non_callable_trajectory_postprocessor():
    with pytest.raises(TypeError, match="must resolve to a callable"):
        await _build_framework_with_agent_runners(
            agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
            gateway_manager=_FakeGatewayManager({}),
            trajectory_postprocessor_fqn=f"{__name__}._NOT_CALLABLE_POSTPROCESSOR",
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("postprocessor_fqn", "postprocessor_kwargs", "error_type", "error_message"),
    [
        ("   ", None, ValueError, "trajectory_postprocessor_fqn must be a non-empty string"),
        (123, None, ValueError, "trajectory_postprocessor_fqn must be a non-empty string"),
        (None, {"enabled": True}, ValueError, "requires trajectory_postprocessor_fqn"),
        (f"{__name__}._recording_trajectory_postprocessor", [1], TypeError, "must be a mapping"),
    ],
)
async def test_framework_rejects_invalid_trajectory_postprocessor_config(
    postprocessor_fqn,
    postprocessor_kwargs,
    error_type,
    error_message,
):
    with pytest.raises(error_type, match=error_message):
        await _build_framework_with_agent_runners(
            agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
            gateway_manager=_FakeGatewayManager({}),
            trajectory_postprocessor_fqn=postprocessor_fqn,
            trajectory_postprocessor_kwargs=postprocessor_kwargs,
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_keeps_successful_sessions_when_one_session_fails(fake_tq):
    """A failed rollout session aborts only that session; other successful
    sessions for the same prompt are still finalized and written to TQ."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [_trajectory()],
            "session-sample-0-rollout-1": [_trajectory()],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        if session.session_id.startswith("session-sample-0-rollout-1-"):
            raise RuntimeError("gateway failed once")

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        gateway_manager=runtime,
        n=2,
        val_n=2,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-sample-0-rollout-1-")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_marks_prompt_failure_when_all_sessions_fail(fake_tq):
    """Filtering every session to empty marks the uid failed and raises."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [_trajectory()],
            "session-sample-0-rollout-1": [_trajectory()],
        }
    )

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        trajectory_postprocessor_fqn=f"{__name__}._empty_trajectory_postprocessor",
        n=1,
        val_n=2,
    )

    with pytest.raises(RuntimeError, match="All rollouts failed at global_steps=9"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=9, validate=True))

    assert fake_tq.batch_puts == []
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "val", "tag": {"status": "failure"}}]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_omits_missing_rollout_log_probs(fake_tq):
    """Missing backend logprobs are omitted while reward scores remain zero-filled."""
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory(response_logprobs=None)]})

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        n=1,
        val_n=1,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    fields = fake_tq.batch_puts[0]["fields"]
    assert fields["rm_scores"][0].tolist() == [0.0, 0.0]
    assert "rollout_log_probs" not in fields


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generate_sequences_keeps_other_prompts_when_one_prompt_fails(fake_tq):
    """Prompt-level failures are isolated: one uid can fail while another uid
    in the same batch still writes successful output."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-1-rollout-0": [_trajectory()],
        }
    )

    async def agent_runner(*, sample_index, **kwargs):
        if sample_index == 0:
            raise RuntimeError("prompt 0 exploded")

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        gateway_manager=runtime,
        n=1,
        val_n=1,
    )

    await framework.generate_sequences(_build_prompts(count=2, global_steps=11))

    assert [put["keys"] for put in fake_tq.batch_puts] == [["uid-1_0_0"]]
    assert sorted(fake_tq.puts, key=lambda put: put["key"]) == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "failure"}},
        {"key": "uid-1", "partition_id": "train", "tag": {"status": "finished"}},
    ]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-sample-0-rollout-0-")
    assert all(session_id.startswith("session-sample-1-rollout-0-") for session_id in runtime.finalized_sessions)


# ---------------------------------------------------------------------------
# _score_trajectories method-level tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_score_trajectories_merges_final_reward_info_into_reward_extra_info():
    """Reward scoring dispatches only the final trajectory to the worker and
    broadcasts that score and extra info to every trajectory in the session.

    Session-level reward_info submitted by the runner is merged into reward
    extra_info for scoring, with reward_info taking precedence on key
    collisions.
    """

    class _ComputeScoreRemote:
        def __init__(self):
            self.calls = []

        async def remote(self, data):
            self.calls.append(data)
            return {"reward_score": 0.42, "reward_extra_info": {"acc": 1.0, "format": 0.8}}

    class _StubWorker:
        def __init__(self):
            self.compute_score = _ComputeScoreRemote()

    worker = _StubWorker()

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=_FakeGatewayManager({}),
        reward_loop_worker_handles=[worker],
        n=1,
        val_n=1,
    )

    trajectories = [
        Trajectory(prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 1], num_turns=1),
        Trajectory(prompt_ids=[5, 6], response_ids=[7, 8], response_mask=[1, 1], num_turns=2),
        Trajectory(
            prompt_ids=[9, 10],
            response_ids=[11, 12],
            response_mask=[1, 1],
            reward_info={"reward_score": 0.9, "index": "from-reward-info"},
            num_turns=3,
        ),
    ]
    sample_fields = {
        "data_source": "test",
        "raw_prompt": [{"role": "user", "content": "hi"}],
        "reward_model": {"ground_truth": "answer"},
        "extra_info": {"index": "from-sample", "case_id": "case-1"},
        "tools_kwargs": {"tool": "search"},
        "agent_name": "deepeyes",
    }
    annotations = await framework._score_trajectories(trajectories, sample_fields)

    assert len(worker.compute_score.calls) == 1
    data = worker.compute_score.calls[0]
    assert data.batch["prompts"].tolist() == [[9, 10]]
    assert data.batch["responses"].tolist() == [[11, 12]]
    assert data.batch["input_ids"].tolist() == [[9, 10, 11, 12]]
    assert data.batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert data.non_tensor_batch["data_source"].tolist() == ["test"]
    assert data.non_tensor_batch["raw_prompt"].tolist() == [[{"role": "user", "content": "hi"}]]
    assert data.non_tensor_batch["reward_model"].tolist() == [{"ground_truth": "answer"}]
    assert data.non_tensor_batch["extra_info"].tolist() == [
        {"index": "from-reward-info", "case_id": "case-1", "reward_score": 0.9}
    ]
    assert data.non_tensor_batch["tools_kwargs"].tolist() == [{"tool": "search"}]
    assert data.non_tensor_batch["agent_name"].tolist() == ["deepeyes"]
    assert data.non_tensor_batch["__num_turns__"].tolist() == [3]
    assert annotations == [
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
    ]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["timeout", "parent_cancel"])
async def test_ray_task_termination_cancels_runner_and_aborts_session(monkeypatch, fake_tq, termination):
    """Timeout and parent cancellation clean up both sides of a ray_task session.

    ``asyncio.wait_for`` only bounds the parent's await; without an explicit
    ``ray.cancel`` the remote runner (and its sandbox) would keep running and
    consuming a worker slot. Mirrors real Ray semantics: after a graceful
    ``ray.cancel``, awaiting the ObjectRef raises (TaskCancelledError), so the
    framework must not escalate to a force-kill.
    """
    from uni_agent.framework import framework as framework_module

    remote_started = asyncio.Event()
    cancel_calls: list[dict] = []

    class _TaskCancelledError(Exception):
        pass

    class _PendingRef:
        def __init__(self):
            self.cancelled = False

        def __await__(self):
            remote_started.set()
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            if self.cancelled:
                fut.set_exception(_TaskCancelledError("runner task cancelled"))
            return fut.__await__()

    def fake_remote(*args, **kwargs):
        return _PendingRef()

    def fake_cancel(ref, *, force=False):
        cancel_calls.append({"ref": ref, "force": force})
        ref.cancelled = True

    monkeypatch.setattr(framework_module._run_agent_runner_ray_task, "remote", fake_remote)
    monkeypatch.setattr(framework_module, "ray", types.SimpleNamespace(cancel=fake_cancel))

    class _GatewayManager(_FakeGatewayManager):
        async def abort_session(self, session_id: str) -> None:
            await super().abort_session(session_id)
            if termination == "parent_cancel":
                raise RuntimeError("gateway unavailable")

    runtime = _GatewayManager({})
    framework = await _build_framework_with_agent_runners(
        agent_runners={
            "runner": {
                "runner_fqn": "tests.uni_agent.support.logging_runner",
                "dispatch_mode": "ray_task",
                "session_timeout_seconds": 0.01 if termination == "timeout" else 10.0,
            }
        },
        gateway_manager=runtime,
    )
    framework._RUNNER_CANCEL_GRACE_SECONDS = 1.0

    task = asyncio.create_task(framework.generate_sequences(_build_prompts(count=1, global_steps=3)))
    await remote_started.wait()
    if termination == "parent_cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="All rollouts failed"):
            await task

    # One graceful cancel only: the post-cancel await raised TaskCancelledError,
    # so no force-kill escalation happened.
    assert [call["force"] for call in cancel_calls] == [False]
    assert runtime.aborted_sessions, "terminated session must be aborted"
