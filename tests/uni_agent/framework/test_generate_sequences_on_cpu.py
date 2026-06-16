from __future__ import annotations

import pytest

from uni_agent.framework.framework import OpenAICompatibleAgentFramework
from uni_agent.gateway.types import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu

_RUNNER_CALLS = []
_TEST_INLINE_RUNNERS = {}


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
) -> dict[str, object]:
    runner_key = f"runner-{len(_TEST_INLINE_RUNNERS)}"
    _TEST_INLINE_RUNNERS[runner_key] = runner
    config = {
        "runner_fqn": f"{__name__}._inline_runner_proxy",
        "runner_kwargs": {"runner_key": runner_key},
        "dispatch_mode": dispatch_mode,
    }
    return config


async def _build_framework_with_agent_runners(
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


@pytest.fixture
def fake_tq(monkeypatch):
    from uni_agent.framework import framework as framework_module

    fake = _FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake)
    return fake


class _FakeReplayBuffer:
    def __init__(self):
        self.adds = []

    def add(self, partition_id, items):
        self.adds.append({"partition_id": partition_id, "items": dict(items)})


@pytest.fixture
def replay_buffer():
    return _FakeReplayBuffer()


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
            reward_info_url=f"http://fake/{session_id}/reward_info",
        )

    async def finalize_session(self, session_id: str):
        self.finalized_sessions.append(session_id)
        return self._lookup(session_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)


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

    Keeps ``generate_sequences`` tests focused on TQ output by returning the
    same deterministic score for every trajectory in the session.
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
async def test_agent_runners_registry_materializes_runners_and_selects_by_agent_name(fake_tq, replay_buffer):
    """Function and class runners keep per-runner kwargs, and each prompt's
    ``agent_name`` selects the matching runner without leaking internals."""
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory()], "session-1-0": [_trajectory()]})
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
        session_runtime=runtime,
        replay_buffer=replay_buffer,
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
    assert [call["kwargs"]["tools_kwargs"] for call in calls] == [{"tool": 0}, {"tool": 1}]
    assert all("session_runtime" not in call["kwargs"] for call in calls)


@pytest.mark.asyncio
async def test_generate_sequences_writes_tq_schema_for_each_session(monkeypatch, fake_tq, replay_buffer):
    """Full ``generate_sequences`` path writes replay-buffer status, one TQ
    batch per successful session, and trainer-compatible trajectory fields."""
    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory(response_logprobs=[-0.1, -0.2], extra_fields={"finish_reason": "length"})],
            "session-0-1": [_trajectory(response_logprobs=[-0.3, -0.4])],
        }
    )

    # Nonzero score proves reward_score lands on the final response token.
    _install_fake_score(
        monkeypatch,
        score_from_sample_fields=lambda sf: sf["extra_info"]["index"] + 0.25,
    )

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=runtime,
        reward_loop_worker_handles=["sentinel"],
        replay_buffer=replay_buffer,
        n=2,
        val_n=2,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    assert replay_buffer.adds == [
        {
            "partition_id": "train",
            "items": {
                "uid-0": {"global_steps": 7, "status": "running"},
            },
        }
    ]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]
    assert fake_tq.batch_puts[1]["keys"] == ["uid-0_1_0"]
    assert fake_tq.puts == [{"key": "uid-0", "partition_id": "train", "tag": {"status": "finished"}}]

    first = fake_tq.batch_puts[0]
    fields = first["fields"]
    assert first["partition_id"] == "train"
    tag = first["tags"][0]
    assert {
        key: tag[key]
        for key in ("global_steps", "status", "prompt_len", "response_len", "seq_len", "uid", "finish_reason")
    } == {
        "global_steps": 7,
        "status": "success",
        "prompt_len": 2,
        "response_len": 2,
        "seq_len": 4,
        "uid": "uid-0",
        "finish_reason": "length",
    }
    assert "length_truncated" not in tag
    assert "traj_exit_reason" not in tag
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
async def test_generate_sequences_keeps_successful_sessions_when_one_session_fails(fake_tq, replay_buffer):
    """A failed rollout session aborts only that session; other successful
    sessions for the same prompt are still finalized and written to TQ."""
    runtime = _FakeSessionRuntime(
        {
            "session-0-0": [_trajectory()],
            "session-0-1": [_trajectory()],
        }
    )

    async def agent_runner(*, raw_prompt, session, sample_index, tools_kwargs, **kwargs):
        if session.session_id.startswith("session-0-1-"):
            raise RuntimeError("gateway failed once")

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
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
async def test_generate_sequences_marks_prompt_failure_when_all_sessions_fail(fake_tq, replay_buffer):
    """If every session for a validation prompt fails, the uid is marked
    failed in TQ and ``generate_sequences`` raises the all-rollouts failure."""
    runtime = _FakeSessionRuntime({"session-0-0": [], "session-0-1": []})

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=runtime,
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
async def test_generate_sequences_zero_fills_missing_trainer_fields(fake_tq, replay_buffer):
    """Without reward workers or backend logprobs, trainer-selected optional
    fields are still emitted as zeros."""
    runtime = _FakeSessionRuntime({"session-0-0": [_trajectory(response_logprobs=None)]})

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        session_runtime=runtime,
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=10))

    fields = fake_tq.batch_puts[0]["fields"]
    assert fields["rm_scores"][0].tolist() == [0.0, 0.0]
    assert fields["rollout_log_probs"][0].tolist() == [0.0, 0.0]


@pytest.mark.asyncio
async def test_generate_sequences_keeps_other_prompts_when_one_prompt_fails(fake_tq, replay_buffer):
    """Prompt-level failures are isolated: one uid can fail while another uid
    in the same batch still writes successful output."""
    runtime = _FakeSessionRuntime(
        {
            "session-1-0": [_trajectory()],
        }
    )

    async def agent_runner(*, sample_index, **kwargs):
        if sample_index == 0:
            raise RuntimeError("prompt 0 exploded")

    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(agent_runner)},
        session_runtime=runtime,
        replay_buffer=replay_buffer,
        n=1,
        val_n=1,
    )

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
    assert [put["keys"] for put in fake_tq.batch_puts] == [["uid-1_0_0"]]
    assert sorted(fake_tq.puts, key=lambda put: put["key"]) == [
        {"key": "uid-0", "partition_id": "train", "tag": {"status": "failure"}},
        {"key": "uid-1", "partition_id": "train", "tag": {"status": "finished"}},
    ]
    assert len(runtime.aborted_sessions) == 1
    assert runtime.aborted_sessions[0].startswith("session-0-0-")
    assert all(session_id.startswith("session-1-0-") for session_id in runtime.finalized_sessions)


# ---------------------------------------------------------------------------
# _score_trajectories method-level tests
# ---------------------------------------------------------------------------


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
        session_runtime=_FakeSessionRuntime({}),
        reward_loop_worker_handles=[worker],
        replay_buffer=_FakeReplayBuffer(),
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
