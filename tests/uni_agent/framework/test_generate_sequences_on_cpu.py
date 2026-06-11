from __future__ import annotations

import pytest

from uni_agent.framework.framework import OpenAICompatibleAgentFramework
from uni_agent.gateway.types import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu


_CONFIG_RUNNER_CALLS = []
_CLASS_RUNNER_CALLS = []
_TEST_INLINE_RUNNERS = {}


async def _config_recording_runner(*, raw_prompt, session, sample_index, marker=None, **kwargs):
    _CONFIG_RUNNER_CALLS.append(
        {
            "runner": marker,
            "raw_prompt": raw_prompt,
            "session_id": session.session_id,
            "base_url": session.base_url,
            "complete_url": session.complete_url,
            "sample_index": sample_index,
            "kwargs": dict(kwargs),
        }
    )


class _ConfigRecordingClassRunner:
    def __init__(self, marker=None):
        self.marker = marker

    async def __call__(self, *, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        _CLASS_RUNNER_CALLS.append(
            {
                "runner": self.marker,
                "raw_prompt": raw_prompt,
                "complete_url": session.complete_url,
                "sample_index": sample_index,
                "tools_kwargs": tools_kwargs,
                "kwargs": dict(kwargs),
            }
        )


async def _ray_task_recording_runner(*, raw_prompt, session, sample_index, marker=None, **kwargs):
    assert marker == "ray"
    assert session.complete_url.endswith("/complete")
    assert raw_prompt == [{"role": "user", "content": f"sample {sample_index}"}]


async def _ray_task_failing_runner(**kwargs):
    raise RuntimeError("ray runner failed")


async def _async_noop_runner(**kwargs):
    return None


async def _inline_runner_proxy(*, runner_key, **kwargs):
    runner = _TEST_INLINE_RUNNERS[runner_key]
    await runner(**kwargs)


def _inline_runner_config(
    runner,
    *,
    dispatch_mode: str = "inline_async",
    max_concurrent_sessions: int = 0,
) -> dict[str, object]:
    runner_key = f"runner-{len(_TEST_INLINE_RUNNERS)}"
    _TEST_INLINE_RUNNERS[runner_key] = runner
    config = {
        "runner_fqn": f"{__name__}._inline_runner_proxy",
        "runner_kwargs": {"runner_key": runner_key},
        "dispatch_mode": dispatch_mode,
    }
    if max_concurrent_sessions:
        config["max_concurrent_sessions"] = max_concurrent_sessions
    return config


async def _framework_from_agent_runners(
    *,
    agent_runners: dict[str, dict[str, object]],
    session_runtime,
    replay_buffer=None,
    reward_loop_worker_handles=None,
    n: int = 1,
    val_n: int = 1,
):
    from omegaconf import OmegaConf

    agent_framework_cfg: dict[str, object] = {"agent_runners": agent_runners}

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": n,
                    "val_kwargs": {"n": val_n},
                    "custom": {"agent_framework": agent_framework_cfg},
                }
            }
        }
    )
    return await OpenAICompatibleAgentFramework.from_config(
        config=config,
        session_runtime=session_runtime,
        replay_buffer=replay_buffer,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


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


class _FakeReplayBuffer:
    def __init__(self):
        self.adds = []

    def add(self, partition_id, items):
        self.adds.append({"partition_id": partition_id, "items": dict(items)})


class _FakeSessionRuntime:
    """Fake runtime that matches session IDs by prefix (``session-{sample}-{session}``)
    to support the real uuid-suffixed IDs produced by the framework."""

    def __init__(self, finalized_by_session_prefix: dict[str, list[Trajectory]]):
        self._finalized_by_prefix = finalized_by_session_prefix
        self.created_sessions = []
        self.finalized_sessions = []
        self.aborted_sessions = []

    def _lookup(self, session_id: str) -> list[Trajectory]:
        for prefix, trajectories in self._finalized_by_prefix.items():
            if session_id.startswith(prefix):
                return trajectories
        raise KeyError(f"No prefix match for session_id={session_id}")

    async def create_session(self, session_id: str, **kwargs):
        self.created_sessions.append(session_id)
        return SessionHandle(
            session_id=session_id,
            base_url=f"http://fake/{session_id}/v1",
            complete_url=f"http://fake/{session_id}/complete",
        )

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self._lookup(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        return None


def _build_prompts(count: int = 2, *, global_steps: int = 7, validate: bool = False):
    non_tensor_dict = {"global_steps": global_steps}
    if validate:
        non_tensor_dict["validate"] = True
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": f"sample {i}"}] for i in range(count)],
            "uid": [f"uid-{i}" for i in range(count)],
            "data_source": ["deepeyes"] * count,
            "reward_model": [{"ground_truth": f"answer-{i}"} for i in range(count)],
            "extra_info": [{"index": i} for i in range(count)],
            "tools_kwargs": [{"tool": i} for i in range(count)],
            "agent_name": ["deepeyes"] * count,
        },
        non_tensor_dict=non_tensor_dict,
    )


def _trajectory(
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    response_logprobs: list[float] | None = None,
    num_turns: int = 2,
    extra_fields: dict[str, object] | None = None,
):
    prompt_ids = prompt_ids or [10, 11]
    response_ids = response_ids or [20, 21]
    return Trajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        response_logprobs=response_logprobs,
        reward_score=None,
        num_turns=num_turns,
        multi_modal_data={"images": ["raw-image-should-not-be-written"]},
        extra_fields=dict(extra_fields or {}),
    )


def _install_fake_score(monkeypatch, *, score_from_sample_fields=None, default_score=1.0):
    """Replace OpenAICompatibleAgentFramework._score_trajectories with a fake.

    Mirrors the production "score-last + broadcast" behavior: returns the same
    (score, extra_info) for every trajectory in the session. The score is
    derived from sample_fields if a callable is provided; otherwise default_score.
    """
    from uni_agent.framework.framework import OpenAICompatibleAgentFramework

    async def fake_score(self, trajectories, sample_fields):
        if score_from_sample_fields is not None:
            score = float(score_from_sample_fields(sample_fields))
        else:
            score = float(default_score)
        return [(score, {})] * len(trajectories)

    monkeypatch.setattr(OpenAICompatibleAgentFramework, "_score_trajectories", fake_score)


@pytest.mark.asyncio
async def test_class_runner_with_async_call_works_like_function_runner(monkeypatch):
    from omegaconf import OmegaConf

    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()]})
    _CLASS_RUNNER_CALLS.clear()
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "deepeyes": {
                                    "runner_fqn": f"{__name__}._ConfigRecordingClassRunner",
                                    "runner_kwargs": {"marker": "class-runner"},
                                    "dispatch_mode": "inline_async",
                                }
                            }
                        }
                    },
                }
            }
        }
    )

    framework = await OpenAICompatibleAgentFramework.from_config(
        config=config,
        session_runtime=runtime,
        replay_buffer=replay_buffer,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=6))

    assert _CLASS_RUNNER_CALLS[0]["runner"] == "class-runner"
    assert _CLASS_RUNNER_CALLS[0]["raw_prompt"] == [{"role": "user", "content": "sample 0"}]
    assert _CLASS_RUNNER_CALLS[0]["complete_url"] == f"http://fake/{runtime.created_sessions[0]}/complete"
    assert _CLASS_RUNNER_CALLS[0]["sample_index"] == 0
    assert _CLASS_RUNNER_CALLS[0]["tools_kwargs"] == {"tool": 0}
    assert "session_runtime" not in _CLASS_RUNNER_CALLS[0]["kwargs"]


@pytest.mark.asyncio
async def test_from_config_preserves_runner_fqn_and_kwargs_for_single_named_runner(monkeypatch):
    from omegaconf import OmegaConf

    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()]})
    _CONFIG_RUNNER_CALLS.clear()
    runner_fqn = f"{__name__}._config_recording_runner"
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "deepeyes": {
                                    "runner_fqn": runner_fqn,
                                    "runner_kwargs": {"marker": "single-runner"},
                                    "dispatch_mode": "inline_async",
                                }
                            }
                        }
                    },
                }
            }
        }
    )

    framework = await OpenAICompatibleAgentFramework.from_config(
        config=config,
        session_runtime=runtime,
        replay_buffer=replay_buffer,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=6))

    assert [call["runner"] for call in _CONFIG_RUNNER_CALLS] == ["single-runner"]
    assert _CONFIG_RUNNER_CALLS[0]["raw_prompt"] == [{"role": "user", "content": "sample 0"}]
    assert _CONFIG_RUNNER_CALLS[0]["base_url"].endswith("/v1")
    assert _CONFIG_RUNNER_CALLS[0]["complete_url"].endswith("/complete")
    assert _CONFIG_RUNNER_CALLS[0]["kwargs"]["tools_kwargs"] == {"tool": 0}
    assert "session_runtime" not in _CONFIG_RUNNER_CALLS[0]["kwargs"]


@pytest.mark.asyncio
async def test_from_config_requires_agent_runners_map():
    from omegaconf import OmegaConf

    from uni_agent.framework.framework import OpenAICompatibleAgentFramework

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                    "custom": {
                        "agent_framework": {
                            "agent_runner_fqn": f"{__name__}._config_recording_runner",
                            "agent_runner_kwargs": {"marker": "legacy-default"},
                        }
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="agent_framework.agent_runners is required"):
        await OpenAICompatibleAgentFramework.from_config(
            config=config,
            session_runtime=_FakeSessionRuntime({}),
            replay_buffer=_FakeReplayBuffer(),
        )


@pytest.mark.asyncio
async def test_agent_runners_registry_selects_runner_by_agent_name(monkeypatch):
    from omegaconf import OmegaConf

    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()], "session-1-0": [_trajectory()]})
    _CONFIG_RUNNER_CALLS.clear()
    runner_fqn = f"{__name__}._config_recording_runner"
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "deepeyes": {
                                    "runner_fqn": runner_fqn,
                                    "runner_kwargs": {"marker": "deepeyes"},
                                    "dispatch_mode": "inline_async",
                                },
                                "swe": {
                                    "runner_fqn": runner_fqn,
                                    "runner_kwargs": {"marker": "swe"},
                                    "dispatch_mode": "inline_async",
                                },
                            }
                        }
                    },
                }
            }
        }
    )
    prompts = _build_prompts(count=2, global_steps=6)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["deepeyes", "swe"]},
        non_tensor_dict={},
    )["agent_name"]

    framework = await OpenAICompatibleAgentFramework.from_config(
        config=config,
        session_runtime=runtime,
        replay_buffer=replay_buffer,
    )

    await framework.generate_sequences(prompts)

    assert [call["runner"] for call in _CONFIG_RUNNER_CALLS] == ["deepeyes", "swe"]
    assert [call["sample_index"] for call in _CONFIG_RUNNER_CALLS] == [0, 1]

@pytest.mark.asyncio
async def test_generate_sequences_writes_tq_schema_for_each_session(monkeypatch):
    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)

    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory(response_logprobs=[-0.1, -0.2])],
            "session-0-1": [_trajectory(response_logprobs=[-0.3, -0.4])],
            "session-1-0": [_trajectory(response_logprobs=[-0.5, -0.6])],
            "session-1-1": [_trajectory(response_logprobs=[-0.7, -0.8])],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        assert raw_prompt == [{"role": "user", "content": f"sample {sample_index}"}]
        assert tools_kwargs == {"tool": sample_index}

    # Score derived from sample_fields["extra_info"]["index"] + 0.25 (same as legacy lambda)
    _install_fake_score(
        monkeypatch,
        score_from_sample_fields=lambda sf: sf["extra_info"]["index"] + 0.25,
    )

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        n=2,
        val_n=2,
    )

    await framework.generate_sequences(_build_prompts(global_steps=7))

    assert replay_buffer.adds == [
        {
            "partition_id": "train",
            "items": {
                "uid-0": {"global_steps": 7, "status": "running"},
                "uid-1": {"global_steps": 7, "status": "running"},
            },
        }
    ]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.batch_puts[1]["keys"] == ["uid-0_1_0"]
    assert fake_tq.batch_puts[2]["keys"] == ["uid-1_0_0"]
    assert fake_tq.batch_puts[3]["keys"] == ["uid-1_1_0"]
    assert fake_tq.puts == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}},
        {"key": "uid-1", "partition_id": "train", "tag": {"status": "finished"}},
    ]

    first = fake_tq.batch_puts[0]
    fields = first["fields"]
    assert first["partition_id"] == "train"
    assert first["tags"] == [{"global_steps": 7, "status": "success", "prompt_len": 2, "response_len": 2, "seq_len": 4}]
    assert fields["input_ids"].is_nested
    assert fields["response_mask"].is_nested
    assert fields["position_ids"].is_nested
    assert fields["prompts"][0].tolist() == [10, 11]
    assert fields["responses"][0].tolist() == [20, 21]
    assert fields["response_mask"][0].tolist() == [1, 1]
    assert fields["loss_mask"][0].tolist() == [1, 1]
    assert fields["input_ids"][0].tolist() == [10, 11, 20, 21]
    assert fields["attention_mask"][0].tolist() == [1, 1, 1, 1]
    assert fields["position_ids"][0].tolist() == [0, 1, 2, 3]
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


@pytest.mark.asyncio
async def test_generate_sequences_keeps_successful_sessions_when_one_session_fails(monkeypatch):
    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory()],
            "session-0-1": [_trajectory()],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        if session.session_id.startswith("session-0-1-"):
            raise RuntimeError("gateway failed once")

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        n=2,
        val_n=2,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=8))

    assert replay_buffer.adds == [
        {"partition_id": "train", "items": {"uid-0": {"global_steps": 8, "status": "running"}}}
    ]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-0-1-")


@pytest.mark.asyncio
async def test_generate_sequences_marks_prompt_failure_when_all_sessions_fail(monkeypatch):
    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [], "session-0-1": []})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        raise RuntimeError(f"failed {session.session_id}")

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        n=1,
        val_n=2,
    )

    with pytest.raises(RuntimeError, match="All rollouts failed at global_steps=9"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=9, validate=True))

    assert replay_buffer.adds == [{"partition_id": "val", "items": {"uid-0": {"global_steps": 9, "status": "running"}}}]
    assert fake_tq.batch_puts == []
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "val", "tag": {"status": "failure"}}]


@pytest.mark.asyncio
async def test_generate_sequences_zero_fills_rm_scores_when_no_reward_handles(monkeypatch):
    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()]})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        return None

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    # rm_scores is always written (zero-filled when no reward) so the trainer's
    # KVBatchMeta select_fields never hits a missing field across the batch.
    rm_scores = fake_tq.batch_puts[0]["fields"]["rm_scores"]
    assert rm_scores[0].tolist() == [0.0, 0.0]


@pytest.mark.asyncio
async def test_generate_sequences_keeps_other_prompts_when_prompt_task_raises(monkeypatch, caplog):
    replay_buffer = _FakeReplayBuffer()
    runtime = _FakeSessionRuntime(
        {
            "session-1-0": [_trajectory()],
        }
    )

    _install_fake_score(monkeypatch, default_score=1.0)

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

    async def fake_run_prompt_sessions_to_tq(*, sample_index, **kwargs):
        if sample_index == 0:
            raise RuntimeError("prompt 0 exploded")
        return {
            "num_success_sessions": 1,
            "num_failed_sessions": 0,
            "num_success_outputs": 1,
            "num_failed_uids": 0,
            "failure_reasons": [],
        }

    monkeypatch.setattr(framework, "_run_prompt_sessions_to_tq", fake_run_prompt_sessions_to_tq)

    caplog.set_level("INFO")
    await framework.generate_sequences(_build_prompts(count=2, global_steps=11))

    assert replay_buffer.adds == [
        {
            "partition_id": "train",
            "items": {
                "uid-0": {"global_steps": 11, "status": "running"},
                "uid-1": {"global_steps": 11, "status": "running"},
            },
        }
    ]
    assert "num_failed_uids=1" in caplog.text
    assert "prompt 0 exploded" in caplog.text


@pytest.mark.asyncio
async def test_generate_sequences_zero_fills_rollout_log_probs_when_missing(monkeypatch):
    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    # Trajectory without response_logprobs (e.g. backend returned no logprobs).
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory(response_logprobs=None)]})

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        return None

    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    # rollout_log_probs is zero-filled rather than omitted so the trainer's
    # bypass-mode select_fields(["rollout_log_probs"]) never KeyErrors.
    rollout_log_probs = fake_tq.batch_puts[0]["fields"]["rollout_log_probs"]
    assert rollout_log_probs[0].tolist() == [0.0, 0.0]


@pytest.mark.asyncio
async def test_per_runner_max_concurrent_sessions_caps_only_selected_runner(monkeypatch):
    import asyncio

    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({f"session-{i}-0": [_trajectory()] for i in range(4)})
    prompts = _build_prompts(count=4, global_steps=13)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["limited", "wide", "wide", "limited"]},
        non_tensor_dict={},
    )["agent_name"]

    release = asyncio.Event()
    in_flight = {"limited": 0, "wide": 0}
    max_observed = {"limited": 0, "wide": 0}
    started = {"limited": 0, "wide": 0}
    limited_first_started = asyncio.Event()
    wide_two_started = asyncio.Event()

    async def limited_runner(**kwargs):
        started["limited"] += 1
        limited_first_started.set()
        in_flight["limited"] += 1
        max_observed["limited"] = max(max_observed["limited"], in_flight["limited"])
        await release.wait()
        in_flight["limited"] -= 1

    async def wide_runner(**kwargs):
        started["wide"] += 1
        if started["wide"] == 2:
            wide_two_started.set()
        in_flight["wide"] += 1
        max_observed["wide"] = max(max_observed["wide"], in_flight["wide"])
        await release.wait()
        in_flight["wide"] -= 1

    framework = await _framework_from_agent_runners(
        agent_runners={
            "limited": _inline_runner_config(
                limited_runner,
                max_concurrent_sessions=1,
            ),
            "wide": _inline_runner_config(
                wide_runner,
                max_concurrent_sessions=2,
            ),
        },
        session_runtime=runtime,
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

    task = asyncio.create_task(framework.generate_sequences(prompts))
    try:
        await asyncio.wait_for(limited_first_started.wait(), timeout=5)
        await asyncio.wait_for(wide_two_started.wait(), timeout=5)
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert max_observed == {"limited": 1, "wide": 2}


@pytest.mark.asyncio
async def test_ray_task_exception_aborts_only_that_session(monkeypatch, ray_runtime):
    from omegaconf import OmegaConf

    from uni_agent.framework import framework as framework_module
    fake_tq = _FakeTransferQueue()
    replay_buffer = _FakeReplayBuffer()
    monkeypatch.setattr(framework_module, "tq", fake_tq)
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()], "session-1-0": [_trajectory()]})
    success_fqn = "tests.uni_agent.framework.test_generate_sequences_on_cpu._ray_task_recording_runner"
    failure_fqn = "tests.uni_agent.framework.test_generate_sequences_on_cpu._ray_task_failing_runner"
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "val_kwargs": {"n": 1},
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "swe-ok": {
                                    "runner_fqn": success_fqn,
                                    "runner_kwargs": {"marker": "ray"},
                                    "dispatch_mode": "ray_task",
                                    "max_concurrent_sessions": 1,
                                },
                                "swe-fail": {
                                    "runner_fqn": failure_fqn,
                                    "dispatch_mode": "ray_task",
                                    "max_concurrent_sessions": 1,
                                },
                            }
                        }
                    },
                }
            }
        }
    )
    prompts = _build_prompts(count=2, global_steps=16)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["swe-ok", "swe-fail"]},
        non_tensor_dict={},
    )["agent_name"]
    framework = await OpenAICompatibleAgentFramework.from_config(
        config=config,
        session_runtime=runtime,
        replay_buffer=replay_buffer,
    )

    await framework.generate_sequences(prompts)

    assert [put["keys"] for put in fake_tq.batch_puts] == [["uid-0_0_0"]]
    assert sorted(fake_tq.puts, key=lambda put: put["key"]) == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}},
        {"key": "uid-1", "partition_id": "train", "tag": {"status": "failure"}},
    ]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-1-0-")
    assert all(not session_id.startswith("session-1-0-") for session_id in runtime.finalized_sessions)


@pytest.mark.asyncio
async def test_trajectory_to_tq_field_and_tag_copies_finish_reason():
    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=_FakeSessionRuntime({}),
    )
    trajectory = _trajectory(extra_fields={"finish_reason": "length"})

    _, tag = framework._trajectory_to_tq_field_and_tag(
        trajectory=trajectory,
        sample_fields={},
        session_index=0,
        global_steps=12,
    )

    assert tag["finish_reason"] == "length"
    assert "length_truncated" not in tag
    assert "traj_exit_reason" not in tag


# ---------------------------------------------------------------------------
# _score_trajectories method-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ray_runtime():
    import ray

    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_score_trajectories_dispatches_only_final_trajectory_and_broadcasts(ray_runtime):
    """_score_trajectories scores trajectories[-1] only, broadcasts to all (matches AgentLoopWorkerTQ)."""
    import ray as ray_module

    from uni_agent.gateway.types import Trajectory

    @ray_module.remote
    class _StubWorker:
        def __init__(self):
            self.calls = []

        def compute_score(self, data):
            self.calls.append(data)
            return {"reward_score": 0.42, "reward_extra_info": {"acc": 1.0, "format": 0.8}}

        def get_call_count(self):
            return len(self.calls)

    worker = _StubWorker.remote()

    runtime = _FakeSessionRuntime({})  # not used in this test
    framework = await _framework_from_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=[worker],
        replay_buffer=_FakeReplayBuffer(),
        n=1,
        val_n=1,
    )

    trajectories = [
        Trajectory(prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 1], num_turns=1),
        Trajectory(prompt_ids=[5, 6], response_ids=[7, 8], response_mask=[1, 1], num_turns=2),
        Trajectory(prompt_ids=[9, 10], response_ids=[11, 12], response_mask=[1, 1], num_turns=3),
    ]
    sample_fields = {"data_source": "test", "raw_prompt": [{"role": "user", "content": "hi"}]}
    annotations = await framework._score_trajectories(trajectories, sample_fields)

    # Score-last + broadcast: 3 trajectories, but only 1 worker call
    assert ray_module.get(worker.get_call_count.remote()) == 1
    # All 3 trajectories get the same score and extra_info
    assert annotations == [
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
        (0.42, {"acc": 1.0, "format": 0.8}),
    ]
