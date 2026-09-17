import json
from pathlib import Path

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.pipeline_smoke.preprocess import build_rows
from uni_agent.tasks.pipeline_smoke.reward import compute_reward, compute_score, normalize_answer
from uni_agent.tasks.pipeline_smoke.task import PipelineSmokeTask, PipelineSmokeTaskConfig, extract_final_answer
from uni_agent.tasks.pipeline_smoke.verify_training import verify_finish_evidence, verify_training_log

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_single_sample_rloo_keeps_equal_positive_rewards():
    import numpy as np
    import torch

    from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage

    rewards = torch.tensor([[0.0, 0.5], [0.0, 0.5]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_rloo_outcome_advantage(rewards, mask, np.array(["a", "b"]))
    assert torch.equal(advantages, torch.full_like(rewards, 0.5))


def test_finish_evidence_rejects_plain_text_and_requires_saved_reward(tmp_path):
    session = tmp_path / "step_1" / "session-example"
    session.mkdir(parents=True)
    task_log = session / "task.log"
    task_log.write_text('finish {"answer": "42"}', encoding="utf-8")
    with pytest.raises(ValueError, match="no correct real finish"):
        verify_finish_evidence(tmp_path)
    task_log.write_text("pipeline_smoke: correct_finish_observed", encoding="utf-8")
    with pytest.raises(ValueError, match="missing trajectory"):
        verify_finish_evidence(tmp_path)
    trajectory = session / "trajectory.json"
    trajectory.write_text(json.dumps({"trajectories": [{"finished": True, "reward_score": 0.5}]}))
    with pytest.raises(ValueError, match="no correct real finish"):
        verify_finish_evidence(tmp_path)
    trajectory.write_text(json.dumps({"trajectories": [{"finished": True, "reward_score": 1.0}]}))
    assert verify_finish_evidence(tmp_path) == 1


@pytest.mark.asyncio
async def test_real_finish_tool_observation_is_scored():
    from uni_agent.tools import Toolbox

    toolbox = Toolbox.from_specs([{"name": "finish"}], sandbox=None)
    result = await toolbox.call("finish", {"answer": "42"})
    assert result.status == "ok"
    transcript = [{"role": "tool", "name": "finish", "content": result.to_observation()}]
    assert compute_reward(extract_final_answer(transcript), "42", used_finish_tool=True) == 1.0


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        ("  Quartz  ", "quartz"),
        ("Forty   Two", "forty two"),
        (42, "42"),
    ],
)
def test_normalize_answer(value, normalized):
    assert normalize_answer(value) == normalized


def test_compute_score_is_deterministic_exact_match():
    assert compute_score(" QUARTZ ", "quartz") == 1.0
    assert compute_score("quartz stone", "quartz") == 0.0
    assert compute_score("", "quartz") == 0.0


def test_compute_reward_adds_finish_tool_compliance_signal():
    assert compute_reward("quartz", "quartz", used_finish_tool=True) == 1.0
    assert compute_reward("quartz", "quartz", used_finish_tool=False) == 0.5
    assert compute_reward("wrong", "quartz", used_finish_tool=True) == 0.0


def test_extract_final_answer_prefers_finish_tool():
    transcript = [
        {"role": "assistant", "content": "I think the answer is quartz."},
        {"role": "tool", "name": "finish", "content": "Observation:\nquartz"},
    ]
    assert extract_final_answer(transcript) == "quartz"


def test_extract_final_answer_accepts_plain_text_fallback():
    assert extract_final_answer([{"role": "assistant", "content": "42"}]) == "42"
    assert extract_final_answer([{"role": "assistant", "content": "17 minus 9 is 8."}]) == "8"
    assert extract_final_answer([]) == ""


def test_extract_final_answer_does_not_reward_a_quoted_non_answer():
    transcript = [{"role": "assistant", "content": 'I considered "nova", but I am not giving a final answer.'}]
    assert extract_final_answer(transcript) != "nova"


def test_extract_final_answer_does_not_treat_comparison_as_equation_result():
    assert extract_final_answer([{"role": "assistant", "content": "9 != 8"}]) != "8"
    assert extract_final_answer([{"role": "assistant", "content": "x == 8"}]) != "8"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("17 - 9 = 8", "8"),
        ("6 x 7 = 42\n\n<finish>\n42\n</finish>", "42"),
        ('The word NOVA in lowercase is "nova".', "nova"),
        ("The final answer is quartz.", "quartz"),
        ('finish\n{"answer": "quartz"}', "quartz"),
    ],
)
def test_extract_final_answer_handles_small_model_output(content, expected):
    assert extract_final_answer([{"role": "assistant", "content": content}]) == expected


def test_dataset_rows_are_deterministic_and_self_contained():
    rows = build_rows(3)
    assert rows == build_rows(3)
    assert [row["data_source"] for row in rows] == ["pipeline_smoke"] * 3
    task = rows[0]["extra_info"]["tools_kwargs"]["task"]
    assert task["name"] == "pipeline_smoke"
    assert task["expected_answer"]
    assert rows[0]["prompt"][0]["role"] == "system"


def test_training_and_validation_examples_do_not_overlap():
    train_rows = build_rows(16)
    val_rows = build_rows(4, offset=16)
    train_prompts = {row["prompt"][-1]["content"] for row in train_rows}
    val_prompts = {row["prompt"][-1]["content"] for row in val_rows}
    assert train_prompts.isdisjoint(val_prompts)


@pytest.mark.parametrize(("size", "offset"), [(21, 0), (4, 17), (1, -1)])
def test_dataset_rejects_requests_that_repeat_bundled_examples(size, offset):
    with pytest.raises(ValueError):
        build_rows(size, offset=offset)


def _training_log(
    *,
    grad_norms: tuple[float, float] = (2.0, 3.0),
    failed_sessions: int = 0,
    reward_mean: float = 0.5,
) -> str:
    lines = []
    for step, grad_norm in enumerate(grad_norms, start=1):
        lines.append(
            "generate_sequences summary: num_input_prompts=2 num_success_sessions=4 "
            "num_success_outputs=4 "
            f"num_failed_sessions={failed_sessions} num_unfinished_episodes=0 num_failed_uids=0"
        )
        lines.append(
            f"step:{step} - training/global_step:{step} - critic/rewards/mean:{reward_mean} "
            f"- actor/grad_norm:{grad_norm} - timing_s/update_actor:1.0 - timing_s/update_weights:1.0"
        )
    return "\n".join(lines)


def test_verify_training_log_accepts_complete_evidence():
    summary = verify_training_log(_training_log(), expected_step=2)
    assert summary["global_step"] == 2
    assert summary["nonzero_grad_steps"] == 2


def test_verify_training_log_accepts_additional_worker_summaries():
    content = _training_log()
    content += "\n" + content.splitlines()[0]
    assert verify_training_log(content, expected_step=2)["rollout_summaries"] == 3


def test_verify_training_log_rejects_zero_gradient():
    with pytest.raises(ValueError, match="no non-zero actor gradient"):
        verify_training_log(_training_log(grad_norms=(0.0, 0.0)), expected_step=2)


def test_verify_training_log_rejects_failed_rollout():
    with pytest.raises(ValueError, match="invalid num_failed_sessions"):
        verify_training_log(_training_log(failed_sessions=1), expected_step=2)


def test_verify_training_log_rejects_missing_success_output():
    content = _training_log().replace("num_success_outputs=4", "num_success_outputs=3", 1)
    with pytest.raises(ValueError, match="session/output counts do not match"):
        verify_training_log(content, expected_step=2)


def test_verify_training_log_rejects_all_zero_rewards():
    with pytest.raises(ValueError, match="no positive task reward"):
        verify_training_log(_training_log(reward_mean=0.0), expected_step=2)


def test_verify_training_log_rejects_incomplete_step_sequence():
    content = _training_log().replace("training/global_step:1", "training/global_step:2")
    with pytest.raises(ValueError, match="training steps are"):
        verify_training_log(content, expected_step=2)


def test_verify_training_log_rejects_non_positive_update_timing():
    content = _training_log().replace("timing_s/update_actor:1.0", "timing_s/update_actor:0.0", 1)
    with pytest.raises(ValueError, match="non-positive actor-update timing"):
        verify_training_log(content, expected_step=2)


def test_task_config_exposes_finish_only():
    config_path = Path(__file__).parents[3] / "examples" / "quickstart" / "training" / "task_config_pipeline_smoke.yaml"
    resolver = TaskConfigResolver.from_file(str(config_path))
    sample = build_rows(1)[0]["extra_info"]["tools_kwargs"]["task"]
    resolved = resolver.resolve(sample)
    assert resolved["sandbox"]["provider"] == "local"
    assert resolved["agent"]["tools"] == [{"name": "finish"}]
    assert resolved["agent"]["model"]["max_total_tokens"] >= 512
    assert get_task(resolved).__class__ is PipelineSmokeTask


class _FakeSandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeAgent:
    async def run(self, **_):
        return AgentResult(
            transcript=[{"role": "tool", "name": "finish", "content": "Observation:\nquartz"}],
            info={"steps": 1, "num_tool_calls": 1},
            finished=True,
        )


class _FakePlainTextAgent:
    async def run(self, **_):
        return AgentResult(
            transcript=[{"role": "assistant", "content": "quartz"}],
            info={"steps": 1, "num_tool_calls": 0},
            finished=True,
        )


class _FakeRecoveredPlainTextAgent:
    async def run(self, **_):
        return AgentResult(
            transcript=[
                {"role": "tool", "name": "finish", "content": "Observation:\nInvalid action: bad arguments"},
                {"role": "assistant", "content": "quartz"},
            ],
            info={"steps": 2, "num_tool_calls": 1},
            finished=True,
        )


@pytest.mark.asyncio
async def test_pipeline_smoke_task_reports_reward(monkeypatch):
    monkeypatch.setattr(PipelineSmokeTask, "build_sandbox", lambda _: _FakeSandbox())
    monkeypatch.setattr(PipelineSmokeTask, "build_agent", lambda _: _FakeAgent())
    task = PipelineSmokeTask(
        PipelineSmokeTaskConfig(
            expected_answer="quartz",
            sandbox={"provider": "local"},
            prompt=[{"role": "user", "content": "Return quartz."}],
        )
    )

    result = await task.run()

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.extra_info["used_finish_tool"] is True


@pytest.mark.asyncio
async def test_pipeline_smoke_task_rewards_correct_plain_text_less(monkeypatch):
    monkeypatch.setattr(PipelineSmokeTask, "build_sandbox", lambda _: _FakeSandbox())
    monkeypatch.setattr(PipelineSmokeTask, "build_agent", lambda _: _FakePlainTextAgent())
    task = PipelineSmokeTask(
        PipelineSmokeTaskConfig(
            expected_answer="quartz",
            sandbox={"provider": "local"},
            prompt=[{"role": "user", "content": "Return quartz."}],
        )
    )

    result = await task.run()

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.extra_info["used_finish_tool"] is False


@pytest.mark.asyncio
async def test_failed_finish_attempt_does_not_earn_tool_bonus(monkeypatch):
    monkeypatch.setattr(PipelineSmokeTask, "build_sandbox", lambda _: _FakeSandbox())
    monkeypatch.setattr(PipelineSmokeTask, "build_agent", lambda _: _FakeRecoveredPlainTextAgent())
    task = PipelineSmokeTask(
        PipelineSmokeTaskConfig(
            expected_answer="quartz",
            sandbox={"provider": "local"},
            prompt=[{"role": "user", "content": "Return quartz."}],
        )
    )

    result = await task.run()

    assert result.reward == 0.5
    assert result.extra_info["used_finish_tool"] is False
