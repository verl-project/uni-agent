"""Interrupted validation has explicit counts without synthetic trajectories."""

import pytest

from uni_agent.integrations.verl_v1.metrics import process_validation_metrics
from verl.trainer.ppo.metric_utils import process_validation_metrics as native_metrics


@pytest.mark.cpu
@pytest.mark.level0
def test_three_scored_two_missing_use_five_observations():
    expected = {("swe", str(i)): 1 for i in range(5)}
    infos = {"reward": [1.0, 1.0, 0.0], "acc": [1.0, 1.0, 0.0], "score": [1.0, 1.0, 0.0]}
    result = process_validation_metrics(["swe"] * 3, ["0", "1", "2"], infos, expected_acc_counts=expected)["swe"]
    for name in infos:
        assert result[name]["mean@1"] == pytest.approx(0.4)
        assert result[name + "_success_only"]["mean@1"] == pytest.approx(2 / 3)
    assert infos["reward"] == [1.0, 1.0, 0.0]
    assert native_metrics(["swe"] * 3, ["0", "1", "2"], infos, expected_acc_counts=expected)["swe"]["reward"][
        "mean@1"
    ] == pytest.approx(2 / 3)


@pytest.mark.cpu
@pytest.mark.level0
def test_all_failed_returns_zero_reward_without_fabricating_accuracy():
    result = process_validation_metrics(
        [], [], {"reward": []}, expected_acc_counts={("swe", str(i)): 1 for i in range(5)}
    )
    assert result == {"swe": {"reward": {"mean@1": 0.0}}}


@pytest.mark.cpu
@pytest.mark.level0
def test_sources_keep_their_metric_definitions():
    result = process_validation_metrics(
        ["scored", "continuous"],
        ["one", "two"],
        {"reward": [1.0, -2.0], "score": [1.0, None]},
        expected_acc_counts={(source, uid): 1 for source in ("scored", "continuous") for uid in ("one", "two")},
    )
    assert result["scored"]["score"]["mean@1"] == 0.5
    assert result["continuous"]["reward"]["mean@1"] == -1.0
    assert result["continuous"]["reward_success_only"]["mean@1"] == -2.0
    assert "score" not in result["continuous"]
    assert "acc" not in result["continuous"]


@pytest.mark.cpu
@pytest.mark.level0
def test_complete_metrics_and_majority_preserve_native_results():
    sources, uids = ["swe"] * 3, ["one"] * 3
    infos = {"reward": [1.0, 0.0, 1.0], "acc": [1.0, 0.0, 1.0], "pred": ["A", "B", "A"], "latency": [1.0, 2.0, 3.0]}
    original = native_metrics(sources, uids, infos)
    result = process_validation_metrics(sources, uids, infos, expected_acc_counts={("swe", "one"): 3})
    for variable, metrics in original["swe"].items():
        assert result["swe"][variable] == metrics


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_predictions_do_not_create_majority_votes():
    result = process_validation_metrics(
        ["swe"], ["one"], {"reward": [1.0], "pred": ["A"]}, expected_acc_counts={("swe", "one"): 2}
    )["swe"]["reward"]
    assert result["mean@2"] == 0.5
    assert not any(name.startswith("maj") for name in result)


@pytest.mark.cpu
@pytest.mark.level0
def test_none_score_keeps_prediction_alignment():
    result = process_validation_metrics(
        ["swe", "swe"],
        ["one", "one"],
        {"reward": [None, 1.0], "pred": ["A", "B"]},
        expected_acc_counts={("swe", "one"): 2},
    )["swe"]["reward"]
    expected = native_metrics(["swe", "swe"], ["one", "one"], {"reward": [0.0, 1.0], "pred": ["A", "B"]})["swe"][
        "reward"
    ]
    assert result == expected


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("expected", [{("swe", "other"): 1}, {("swe", "one"): 0}, {("swe", "one"): True}])
def test_invalid_observation_contract_is_rejected(expected):
    with pytest.raises(ValueError):
        process_validation_metrics(["swe"], ["one"], {"reward": [1.0]}, expected_acc_counts=expected)
