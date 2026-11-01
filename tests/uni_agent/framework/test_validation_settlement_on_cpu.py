"""Validation termination must settle results without cancelling training or fabricating trajectories."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from uni_agent.framework import framework as module
from uni_agent.gateway.session import SessionHandle, Trajectory
from verl.utils import tensordict_utils as tu

RUNNERS = {}


async def run_case(*, runner_key, **kwargs):
    return await RUNNERS[runner_key](**kwargs)


class Queue:
    def __init__(self):
        self.statuses = {}
        self.outputs = []

    async def async_kv_put(self, *, key, partition_id, tag):
        self.statuses[(partition_id, key)] = tag["status"]

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.outputs.extend(keys)


class Gateway:
    def __init__(self):
        self.created = []
        self.aborted = []

    async def create_session(self, session_id, **kwargs):
        self.created.append(session_id)
        return SessionHandle(session_id=session_id)

    async def finalize_session(self, session_id):
        fields = dict(prompt_ids=[1], response_ids=[2], response_mask=[1], response_logprobs=[-0.2], num_turns=1)
        if "reward_info" in Trajectory.__dataclass_fields__:
            fields["reward_info"] = {"reward": 1.0, "acc": 1.0, "finished": True}
        return [Trajectory(**fields)]

    async def abort_session(self, session_id):
        self.aborted.append(session_id)


def build(monkeypatch, runner, *, log_dir=None, n=1, cap=0):
    queue = Queue()
    gateway = Gateway()
    monkeypatch.setattr(module, "tq", queue)
    key = str(len(RUNNERS))
    RUNNERS[key] = runner
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": n,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": -1,
                    "calculate_log_probs": True,
                    "val_kwargs": {"n": n, "temperature": 0.2, "top_p": 0.95, "top_k": -1},
                    "custom": {
                        "agent_framework": {
                            "log_dir": str(log_dir) if log_dir else None,
                            "agent_runners": {
                                "case": {
                                    "runner_fqn": __name__ + ".run_case",
                                    "runner_kwargs": {"runner_key": key},
                                    "max_concurrent_sessions": cap,
                                    "dispatch_mode": "inline_async",
                                }
                            },
                        }
                    },
                }
            }
        }
    )
    return module.GatewayAgentFramework.from_config(config=config, gateway_manager=gateway), queue, gateway


def prompts(count=5, *, validation_id="run-a", validate=True, offset=0):
    metadata = {"validate": validate, "global_steps": 7}
    if validation_id is not None:
        metadata["validation_id"] = validation_id
    return tu.get_tensordict(
        tensor_dict={
            "uid": ["uid-" + str(i + offset) for i in range(count)],
            "raw_prompt": [[{"role": "user", "content": str(i)}] for i in range(count)],
            "data_source": ["swe_bench"] * count,
        },
        non_tensor_dict=metadata,
    )


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_stop_preserves_completed_results_and_commits_before_resume(monkeypatch, tmp_path):
    tails = []

    async def runner(*, sample_index, **kwargs):
        if sample_index >= 3:
            tails.append(asyncio.current_task())
            await asyncio.Event().wait()

    framework, queue, gateway = build(monkeypatch, runner, log_dir=tmp_path)
    await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts()))
    await until(lambda: len(tails) == 2 and sum(v in {"finished", "failure"} for v in queue.statuses.values()) == 3)
    await framework.stop_validation("run-a")
    await asyncio.wait_for(batch, 2)
    assert len(queue.outputs) == 3
    assert [queue.statuses[("val", "uid-" + str(i))] for i in range(5)] == ["finished"] * 3 + ["failure"] * 2
    assert len(gateway.aborted) == 2
    state = await framework.publish_validation_metrics("run-a", {"validation/score": 0.4})
    saved = json.loads((tmp_path / "validation/run-a.json").read_text())
    assert saved["metrics"] == state["metrics"] == {"validation/score": 0.4}
    assert state["pending_sessions"] == 0
    assert await framework.stop_validation("run-a") == state
    with pytest.raises(ValueError, match="already committed"):
        await framework.publish_validation_metrics("run-a", {"validation/score": 1.0})


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_stop_covers_queued_episodes_and_future_batches(monkeypatch):
    entered = []

    async def runner(**kwargs):
        entered.append(1)
        await asyncio.Event().wait()

    framework, queue, gateway = build(monkeypatch, runner, cap=1)
    await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts(3)))
    await until(lambda: bool(entered))
    await framework.stop_validation("run-a")
    await asyncio.wait_for(batch, 2)
    await framework.generate_sequences(prompts(2, offset=3))
    assert len(entered) == len(gateway.created) == 1
    assert len(queue.statuses) == 5 and set(queue.statuses.values()) == {"failure"}
    assert not queue.outputs


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_next_validation_is_not_closed_by_old_stop(monkeypatch):
    async def runner(**kwargs):
        return None

    framework, queue, _ = build(monkeypatch, runner)
    await framework.begin_validation("run-a", global_steps=7)
    await framework.stop_validation("run-a")
    await framework.begin_validation("run-b", global_steps=7)
    await framework.generate_sequences(prompts(1, validation_id="run-b"))
    assert queue.statuses[("val", "uid-0")] == "finished"
    assert not framework.validation_status("run-b")["stop_requested"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_parent_cancellation_is_never_reclassified_as_tail_failure(monkeypatch, validate):
    entered = asyncio.Event()

    async def runner(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    framework, queue, _ = build(monkeypatch, runner)
    if validate:
        await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts(1, validate=validate)))
    await entered.wait()
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch
    assert "failure" not in queue.statuses.values()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_completed_n_branch_survives_cancellation_of_its_sibling(monkeypatch):
    entered = asyncio.Event()

    async def runner(*, session, **kwargs):
        if "-rollout-1-" in session.session_id:
            entered.set()
            await asyncio.Event().wait()

    framework, queue, _ = build(monkeypatch, runner, n=2)
    await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts(1)))
    await entered.wait()
    await asyncio.sleep(0.01)
    await framework.stop_validation("run-a")
    await asyncio.wait_for(batch, 2)
    assert len(queue.outputs) == 1 and queue.statuses[("val", "uid-0")] == "finished"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_scores_cannot_be_published_while_an_episode_is_active(monkeypatch):
    async def runner(**kwargs):
        await asyncio.Event().wait()

    framework, _, _ = build(monkeypatch, runner)
    await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts(1)))
    await until(lambda: framework.validation_status("run-a")["pending_sessions"] == 1)
    with pytest.raises(RuntimeError, match="active episodes"):
        await framework.publish_validation_metrics("run-a", {"validation/score": 0.0})
    await framework.stop_validation("run-a")
    await batch


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_metric_commit_accepts_numpy_scalars_and_rejects_nonfinite(monkeypatch, tmp_path):
    async def runner(**kwargs):
        return None

    framework, _, _ = build(monkeypatch, runner, log_dir=tmp_path)
    await framework.begin_validation("run-a", global_steps=7)
    with pytest.raises(ValueError):
        await framework.publish_validation_metrics("run-a", {"score": float("nan")})
    assert framework.validation_status("run-a")["metrics"] is None
    result = await framework.publish_validation_metrics("run-a", {"turns": np.int64(3), "score": np.float64(0.4)})
    assert result["metrics"] == {"turns": 3.0, "score": 0.4}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_generation_drain_saves_score_but_does_not_release_training(monkeypatch, tmp_path):
    entered = asyncio.Event()

    async def runner(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    framework, queue, gateway = build(monkeypatch, runner, log_dir=tmp_path)

    async def cannot_drain(session_id):
        raise RuntimeError("controlled backend cancellation failure")

    gateway.abort_session = cannot_drain
    await framework.begin_validation("run-a", global_steps=7)
    batch = asyncio.create_task(framework.generate_sequences(prompts(1)))
    await entered.wait()
    await framework.stop_validation("run-a")
    await batch
    assert queue.statuses[("val", "uid-0")] == "failure"
    with pytest.raises(RuntimeError, match="training must not resume"):
        await framework.publish_validation_metrics("run-a", {"validation/score": 0.0})
    record = json.loads((tmp_path / "validation/run-a.json").read_text())
    assert record["metrics"] == {"validation/score": 0.0} and record["phase"] == "unsafe"
    assert framework.validation_status("run-a")["phase"] == "unsafe"
