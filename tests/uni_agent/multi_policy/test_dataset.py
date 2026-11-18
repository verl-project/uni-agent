import pytest

from examples.multi_policy.external_mas.dataset import attach_task_metadata

TASK_NAME = "multi_policy_external_mas"


@pytest.mark.cpu
@pytest.mark.level0
def test_attach_task_metadata_forwards_reward_model():
    row = {
        "reward_model": {"ground_truth": ["Paris"]},
        "tools_kwargs": {"tool": "search"},
    }

    result = attach_task_metadata(row)

    assert result["tools_kwargs"]["tool"] == "search"
    task = result["tools_kwargs"]["task"]
    assert task["name"] == TASK_NAME
    assert task["metadata"]["reward_model"] == {"ground_truth": ["Paris"]}
    assert task["ground_truth"] == ["Paris"]


@pytest.mark.cpu
@pytest.mark.level0
def test_attach_task_metadata_preserves_existing_task_metadata():
    row = {
        "reward_model": {"ground_truth": "Paris"},
        "tools_kwargs": {
            "task": {
                "name": TASK_NAME,
                "metadata": {"chunks": [1, 2]},
            }
        },
    }

    result = attach_task_metadata(row)

    task = result["tools_kwargs"]["task"]
    metadata = task["metadata"]
    assert metadata["chunks"] == [1, 2]
    assert metadata["reward_model"] == {"ground_truth": "Paris"}
    assert task["ground_truth"] == ["Paris"]


@pytest.mark.cpu
@pytest.mark.level0
def test_attach_task_metadata_handles_missing_reward_model():
    result = attach_task_metadata({"tools_kwargs": {}})

    task = result["tools_kwargs"]["task"]
    assert task["name"] == TASK_NAME
    assert "reward_model" not in task["metadata"]
    assert task["ground_truth"] == []


@pytest.mark.cpu
@pytest.mark.level0
def test_attach_task_metadata_forwards_ground_truth_raw():
    row = {
        "reward_model": {"ground_truth": ["['NBC' 'Telemundo']"]},
        "extra_info": {"ground_truth_raw": {"target": ["NBC", "Telemundo"]}},
    }

    result = attach_task_metadata(row)

    task = result["tools_kwargs"]["task"]
    assert task["metadata"]["ground_truth_raw"] == {"target": ["NBC", "Telemundo"]}
    assert task["ground_truth"] == ["NBC", "Telemundo"]


@pytest.mark.cpu
@pytest.mark.level0
def test_attach_task_metadata_preserves_existing_ground_truth():
    row = {
        "reward_model": {"ground_truth": "Paris"},
        "tools_kwargs": {"task": {"name": TASK_NAME, "ground_truth": ["London"]}},
    }

    result = attach_task_metadata(row)

    assert result["tools_kwargs"]["task"]["ground_truth"] == ["London"]
