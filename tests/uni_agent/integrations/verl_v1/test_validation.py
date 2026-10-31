"""Validation collection remains valid for empty and partially failed batches."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from uni_agent.integrations.verl_v1 import validation as module
from uni_agent.integrations.verl_v1.trainer import UniAgentSeparateAsyncTrainer
from uni_agent.integrations.verl_v1.validation import ValidationMixin
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerClient


class TokenRows:
    def __init__(self, size):
        self.size = size

    def to_padded_tensor(self, **kwargs):
        return torch.ones((self.size, 1), dtype=torch.long)


class Queue:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.results = {}
        self.current = []
        self.reads = []

    def kv_batch_put(self, *, keys, **kwargs):
        scores = next(self.scores)
        self.current = []
        if scores is not None:
            assert len(scores) == len(keys)
            self.current = [uid + "_0_0" for uid in keys]
            self.results.update({key: (uid, score) for key, uid, score in zip(self.current, keys, scores, strict=True)})

    def kv_batch_get(self, *, keys, select_fields, **kwargs):
        assert keys, "Failed observations must never request empty trajectory tensors"
        self.reads.append(list(keys))
        if select_fields == ["prompts", "responses"]:
            return {name: TokenRows(len(keys)) for name in select_fields}
        values = [self.results[key] for key in keys]
        return {
            "uid": np.array([uid for uid, _ in values]),
            "rm_scores": torch.tensor([[score] for _, score in values]),
            "num_turns": np.array([1] * len(keys)),
            "reward_model": None,
            "data_source": np.array(["swe"] * len(keys)),
            "extra_fields": np.array([{"reward_extra_info": {"acc": score, "score": score}} for _, score in values]),
        }

    def kv_clear(self, *, keys, **kwargs):
        for key in keys:
            del self.results[key]


class Manager:
    def __init__(self):
        self.events = []

    def begin_validation(self, validation_id, **kwargs):
        self.events.append(("begin", validation_id))

    def generate_sequences(self, batch):
        self.events.append(("generate", tu.get(batch, "validation_id")))

    def publish_validation_metrics(self, validation_id, metrics):
        self.events.append(("publish", validation_id, dict(metrics)))


class Harness(ValidationMixin):
    def __init__(self, queue, sizes):
        self.global_steps = 7
        self.config = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
                "trainer": {"validation_data_dir": None},
            }
        )
        self.val_dataloader = [
            {"raw_prompt": [[{"role": "user", "content": "task"}]] * size, "data_source": ["swe"] * size}
            for size in sizes
        ]
        self.agent_loop_manager = Manager()
        self.replay_buffer = SimpleNamespace(
            sample=lambda **kwargs: (SimpleNamespace(keys=list(queue.current), partition_id="val"), None)
        )
        self.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
        self.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *args, **kwargs: "text")

    def _maybe_log_val_generations(self, **kwargs):
        pass


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("scores", [[None, None], [None, [1.0, 0.0]], [[1.0, 1.0, 0.0], None]])
def test_multiple_batches_keep_expected_counts_and_publish_once(monkeypatch, scores):
    queue = Queue(scores)
    monkeypatch.setattr(module, "tq", queue)
    trainer = Harness(queue, [3, 2])
    result = trainer._validate()
    scored = [score for batch in scores if batch is not None for score in batch]
    core = "acc" if scored else "reward"
    assert result[f"val-core/swe/{core}/mean@1"] == pytest.approx(sum(scored) / 5)
    assert result["val-control/expected_sessions"] == 5
    assert result["val-control/scored_sessions"] == len(scored)
    assert result["val-control/failed_sessions"] == 5 - len(scored)
    events = trainer.agent_loop_manager.events
    assert [event[0] for event in events] == ["begin", "generate", "generate", "publish"]
    assert len({event[1] for event in events}) == 1
    assert events[-1][2] == result
    assert not queue.results
    if not scored:
        assert not queue.reads
        assert not any("/acc/" in name for name in result)


@pytest.mark.cpu
@pytest.mark.level0
def test_repeated_validation_at_one_step_has_distinct_round_ids(monkeypatch):
    queue = Queue([None, None])
    monkeypatch.setattr(module, "tq", queue)
    trainer = Harness(queue, [1])
    trainer._validate()
    trainer._validate()
    begun = [event[1] for event in trainer.agent_loop_manager.events if event[0] == "begin"]
    assert len(set(begun)) == 2


@pytest.mark.cpu
@pytest.mark.level0
def test_native_training_methods_and_dependency_are_not_replaced():
    assert UniAgentSeparateAsyncTrainer.fit is PPOTrainerSeparateAsync.fit
    assert UniAgentSeparateAsyncTrainer.step is PPOTrainerSeparateAsync.step
    assert UniAgentSeparateAsyncTrainer.on_step_end is PPOTrainerSeparateAsync.on_step_end
    assert UniAgentSeparateAsyncTrainer.on_validate_begin is PPOTrainerSeparateAsync.on_validate_begin
    assert "validation_id" not in inspect.getsource(PPOTrainer._validate)
    assert "ray.cancel(" not in inspect.getsource(LLMServerClient.generate)
    assert PPOTrainer._validate.__module__.startswith("verl.")
    assert UniAgentSeparateAsyncTrainer._validate.__module__.startswith("uni_agent.")
