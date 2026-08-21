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
    if module is multilingual_preprocess:
        assert task_config["metadata"]["language"] == "C"


@pytest.mark.parametrize(
    ("recipe_path", "task_name", "expects_submit", "expects_language"),
    [
        ("examples/quickstart/inference/task_config_react.yaml", "swe_bench", True, False),
        ("examples/quickstart/inference/task_config_react.yaml", "swe_bench_multilingual", True, True),
        ("examples/quickstart/inference/task_config_claude_code.yaml", "swe_bench", False, False),
        ("examples/quickstart/inference/task_config_claude_code.yaml", "swe_bench_multilingual", False, True),
        ("examples/quickstart/training/task_config_react.yaml", "swe_bench", True, False),
        ("examples/quickstart/training/task_config_react.yaml", "swe_rebench", True, False),
        ("examples/quickstart/training/task_config_react.yaml", "swe_bench_multilingual", True, True),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_bench", False, False),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_rebench", False, False),
        ("examples/quickstart/training/task_config_claude_code.yaml", "swe_bench_multilingual", False, True),
    ],
)
def test_swe_recipe_renders_complete_metadata_prompt(recipe_path, task_name, expects_submit, expects_language):
    source_problem = "Dataset source problem"
    metadata_problem = "Metadata problem"
    resolved = TaskConfigResolver.from_file(recipe_path).resolve(
        {
            "name": task_name,
            "prompt": [{"role": "user", "content": source_problem}],
            "prompt_template": [{"role": "user", "content": "STALE {problem_statement}"}],
            "metadata": {
                "problem_statement": metadata_problem,
                "language": "C",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
            },
        }
    )

    effective_messages = TaskConfig(**resolved).prompt
    effective_text = "\n".join(str(message["content"]) for message in effective_messages)

    assert [message["role"] for message in effective_messages] == ["system", "user"]
    assert metadata_problem in effective_text
    assert source_problem not in effective_text
    assert "SECRET GOLD PATCH" not in effective_text
    assert "SECRET TEST PATCH" not in effective_text
    assert ("submit" in effective_text.lower()) is expects_submit
    assert ("primary language: C" in effective_text) is expects_language
    assert "There is no submit tool; exit after validation." not in effective_text


def test_user_recipe_can_replace_complete_prompt_template(tmp_path):
    recipe = yaml.safe_load(Path("examples/quickstart/inference/task_config_react.yaml").read_text())
    recipe[0]["prompt_template"] = [
        {"role": "system", "content": "Custom instructions"},
        {"role": "user", "content": "Custom issue: {problem_statement}"},
    ]
    config_path = tmp_path / "custom-task.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    resolved = TaskConfigResolver.from_file(str(config_path)).resolve(
        {
            "name": "swe_bench",
            "prompt": [{"role": "user", "content": "Source issue"}],
            "metadata": {"problem_statement": "Metadata issue"},
        }
    )

    assert TaskConfig(**resolved).prompt == [
        {"role": "system", "content": "Custom instructions"},
        {"role": "user", "content": "Custom issue: Metadata issue"},
    ]


def test_mini_swe_agent_recipe_without_template_preserves_source_prompt():
    source_prompt = [{"role": "user", "content": "Source issue"}]
    resolved = TaskConfigResolver(
        {
            "swe_bench": {
                "name": "swe_bench",
                "sandbox": {"provider": "local"},
                "agent": {"name": "mini_swe_agent"},
            }
        }
    ).resolve(
        {
            "name": "swe_bench",
            "prompt": source_prompt,
            "metadata": {"problem_statement": "Metadata issue"},
        }
    )

    config = TaskConfig(**resolved)

    assert config.prompt_template is None
    assert config.prompt == source_prompt
