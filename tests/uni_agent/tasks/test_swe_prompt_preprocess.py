from __future__ import annotations

from copy import deepcopy

import pytest

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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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
    language = "TestLanguageSentinel"
    metadata = {
        "problem_statement": metadata_problem,
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET TEST PATCH",
    }
    if expects_language:
        metadata["language"] = language
    image_prefix = "swerebench" if task_name == "swe_rebench" else "swebench"

    resolved = TaskConfigResolver.from_file(recipe_path).resolve(
        {
            "name": task_name,
            "sandbox": {"image": f"{image_prefix}/test-repo:latest"},
            "prompt": [{"role": "user", "content": source_problem}],
            "metadata": metadata,
        }
    )

    rendered_messages = TaskConfig(**resolved).prompt
    rendered_text = "\n".join(str(message["content"]) for message in rendered_messages)

    assert metadata_problem in rendered_text
    assert source_problem not in rendered_text
    assert "SECRET GOLD PATCH" not in rendered_text
    assert "SECRET TEST PATCH" not in rendered_text
    assert ("submit" in rendered_text.lower()) is expects_submit
    assert (language in rendered_text) is expects_language
