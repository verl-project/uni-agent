from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from uni_agent.tasks import TaskConfig, TaskConfigResolver
from uni_agent.tasks.swe_bench import preprocess as swe_bench_preprocess
from uni_agent.tasks.swe_bench_multilingual import preprocess as multilingual_preprocess
from uni_agent.tasks.swe_rebench import preprocess as swe_rebench_preprocess


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return _FakeDataset([self.rows[index] for index in indices])

    def map(self, function, remove_columns):
        return _FakeDataset([function(deepcopy(row)) for row in self.rows])


@pytest.mark.parametrize(
    ("module", "build_name", "row"),
    [
        (
            swe_bench_preprocess,
            "build_swe_bench_verified",
            {
                "instance_id": "org__repo-1",
                "repo": "org/repo",
                "version": "1",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            },
        ),
        (
            swe_rebench_preprocess,
            "build_swe_rebench",
            {
                "instance_id": "org__repo-2",
                "repo": "org/repo",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "FAIL_TO_FAIL": "[]",
                "PASS_TO_PASS": "[]",
                "PASS_TO_FAIL": "[]",
                "install_config": {"install": "install", "log_parser": "parser", "test_cmd": "test"},
            },
        ),
        (
            multilingual_preprocess,
            "build_swe_bench_multilingual",
            {
                "instance_id": "redis__redis-3",
                "repo": "redis/redis",
                "version": "1",
                "base_commit": "base",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "problem_statement": "Canonical source problem",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            },
        ),
    ],
)
def test_swe_preprocess_emits_source_prompt_without_nested_rendered_prompt(monkeypatch, module, build_name, row):
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: _FakeDataset([row]))

    output = getattr(module, build_name)()[0]

    expected_prompt = [{"role": "user", "content": "Canonical source problem"}]
    assert output["prompt"] == expected_prompt
    task_config = output["extra_info"]["tools_kwargs"]["task"]
    assert "prompt" not in task_config
    assert task_config["metadata"]["problem_statement"] == "Canonical source problem"


@pytest.mark.parametrize(
    ("recipe_path", "expects_submit"),
    [
        ("examples/quickstart/inference/task_config_react.yaml", True),
        ("examples/quickstart/inference/task_config_claude_code.yaml", False),
        ("examples/quickstart/training/task_config_react.yaml", True),
        ("examples/quickstart/training/task_config_claude_code.yaml", False),
    ],
)
def test_swe_recipe_renders_complete_agent_specific_prompt(recipe_path, expects_submit):
    source_problem = "Canonical source problem"
    resolved = TaskConfigResolver.from_file(recipe_path).resolve(
        {
            "name": "swe_bench",
            "prompt": [{"role": "user", "content": source_problem}],
            "prompt_template": [{"role": "user", "content": "STALE {prompt}"}],
            "metadata": {"patch": "SECRET GOLD PATCH", "test_patch": "SECRET TEST PATCH"},
        }
    )

    effective_messages = TaskConfig(**resolved).prompt
    effective_text = "\n".join(str(message["content"]) for message in effective_messages)

    assert [message["role"] for message in effective_messages] == ["system", "user"]
    assert source_problem in effective_text
    assert effective_text.count(source_problem) == 1
    assert "SECRET GOLD PATCH" not in effective_text
    assert "SECRET TEST PATCH" not in effective_text
    assert ("submit" in effective_text.lower()) is expects_submit
    assert "There is no submit tool; exit after validation." not in effective_text


def test_user_recipe_can_replace_complete_prompt_template(tmp_path):
    recipe = yaml.safe_load(Path("examples/quickstart/inference/task_config_react.yaml").read_text())
    recipe[0]["prompt_template"] = [
        {"role": "system", "content": "Custom instructions"},
        {"role": "user", "content": "Custom issue: {prompt}"},
    ]
    config_path = tmp_path / "custom-task.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    resolved = TaskConfigResolver.from_file(str(config_path)).resolve(
        {"name": "swe_bench", "prompt": [{"role": "user", "content": "Source issue"}]}
    )

    assert TaskConfig(**resolved).prompt == [
        {"role": "system", "content": "Custom instructions"},
        {"role": "user", "content": "Custom issue: Source issue"},
    ]
