from __future__ import annotations

from copy import deepcopy

import pytest

from uni_agent.tasks import TaskConfig, TaskConfigResolver, get_task

_LOCAL_SANDBOX = {"provider": "local"}


@pytest.mark.cpu
@pytest.mark.level0
def test_task_config_has_no_logging_runtime_fields():
    assert "log_dir" not in TaskConfig.model_fields


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_config_overrides_file_defaults_and_runtime_endpoint_wins():
    file_defaults = {
        "name": "swe_bench",
        "sandbox": {
            "provider": "modal",
            "runtime_timeout": 3600,
        },
        "agent": {
            "name": "react",
            "max_steps": 100,
            "tools": [{"name": "stateful_shell"}, {"name": "submit"}],
            "model": {
                "temperature": 0.8,
                "top_p": 0.9,
                "base_url": "http://default.invalid/v1",
            },
        },
    }
    sample_config = {
        "name": "swe_bench",
        "sandbox": {
            "provider": "vefaas",
            "image": "swebench/example:latest",
        },
        "agent": {
            "max_steps": 300,
            "tools": [{"name": "submit"}],
            "model": {
                "temperature": 0.2,
                "base_url": "http://sample.invalid/v1",
                "api_key": "sample-key",
                "model_name": "sample-model",
            },
        },
        "metadata": {"instance_id": "sample-1"},
    }
    original_defaults = deepcopy(file_defaults)
    original_sample = deepcopy(sample_config)

    resolved = TaskConfigResolver({"swe_bench": file_defaults}).resolve(
        sample_config,
        runtime_model={
            "base_url": "http://gateway:8000/sessions/1/v1",
            "api_key": "runtime-key",
            "model_name": "runtime-model",
        },
    )

    assert resolved["sandbox"] == {
        "provider": "vefaas",
        "runtime_timeout": 3600,
        "image": "swebench/example:latest",
    }
    assert resolved["agent"]["max_steps"] == 300
    assert resolved["agent"]["tools"] == [{"name": "submit"}]
    assert resolved["agent"]["model"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "base_url": "http://gateway:8000/sessions/1/v1",
        "api_key": "runtime-key",
        "model_name": "runtime-model",
    }
    assert resolved["metadata"] == {"instance_id": "sample-1"}

    assert file_defaults == original_defaults
    assert sample_config == original_sample

    parsed = get_task(resolved).config
    assert parsed.agent.model.temperature == 0.2
    assert parsed.agent.model.top_p == 0.9
    assert parsed.agent.model.base_url == "http://gateway:8000/sessions/1/v1"


@pytest.mark.cpu
@pytest.mark.level0
def test_model_fallbacks_do_not_override_task_config_defaults():
    resolved = TaskConfigResolver(
        {
            "swe_bench": {
                "name": "swe_bench",
                "sandbox": {"provider": "local"},
                "agent": {
                    "name": "react",
                    "model": {
                        "temperature": 0.3,
                        "top_p": 0.7,
                        "top_k": 42,
                    },
                },
            }
        }
    ).resolve(
        {
            "name": "swe_bench",
            "metadata": {"instance_id": "sample-1"},
        },
        runtime_model={
            "base_url": "http://gateway:8000/sessions/1/v1",
            "api_key": "runtime-key",
            "model_name": "runtime-model",
        },
    )

    model = get_task(resolved).config.agent.model
    assert model.temperature == 0.3
    assert model.top_p == 0.7
    assert model.top_k == 42


@pytest.mark.cpu
@pytest.mark.level0
def test_task_prompt_without_template_passes_through_unchanged():
    messages = [
        {"role": "system", "content": "Existing instructions"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/input.png"}},
            ],
        },
    ]

    config = TaskConfig(sandbox=_LOCAL_SANDBOX, prompt=messages)

    assert config.prompt == messages


@pytest.mark.cpu
@pytest.mark.level0
def test_task_prompt_template_renders_multiple_metadata_fields():
    config = TaskConfig(
        sandbox=_LOCAL_SANDBOX,
        prompt=[{"role": "user", "content": "Dataset source stays separate"}],
        prompt_template=[
            {"role": "system", "content": "Work carefully."},
            {
                "role": "user",
                "content": (
                    "Language: {language}\nIssue: {problem_statement}\nAgain: {language}\nUse {{literal braces}}."
                ),
            },
        ],
        metadata={"problem_statement": "Fix the parser", "language": "Python"},
    )

    assert config.prompt == [
        {"role": "system", "content": "Work carefully."},
        {
            "role": "user",
            "content": "Language: Python\nIssue: Fix the parser\nAgain: Python\nUse {literal braces}.",
        },
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_task_prompt_template_allows_no_placeholders():
    config = TaskConfig(
        sandbox=_LOCAL_SANDBOX,
        prompt=[{"role": "user", "content": "Dataset source stays separate"}],
        prompt_template=[{"role": "user", "content": "Static recipe prompt"}],
        metadata={"problem_statement": "Unused"},
    )

    assert config.prompt == [{"role": "user", "content": "Static recipe prompt"}]


@pytest.mark.cpu
@pytest.mark.level0
def test_prompt_template_is_input_only_across_task_config_serialization():
    config = TaskConfig(
        sandbox=_LOCAL_SANDBOX,
        agent={"name": "react"},
        prompt=[{"role": "user", "content": "Fix the parser"}],
        prompt_template=[{"role": "user", "content": "Issue: {problem_statement}"}],
        metadata={"problem_statement": "Fix the parser"},
    )

    dumped = config.model_dump()

    assert "prompt_template" not in dumped
    assert TaskConfig.model_validate(dumped).prompt == config.prompt


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("prompt_template", "metadata", "error"),
    [
        ([{"role": "user", "content": "{missing}"}], {}, "missing.*missing"),
        ([{"role": "user", "content": "Broken {problem_statement"}], {"problem_statement": "issue"}, "invalid"),
        ([{"role": "user", "content": "{patch}"}], {"patch": {"diff": "secret"}}, "text.*patch"),
        (
            [{"role": "user", "content": "{problem_statement[0]}"}],
            {"problem_statement": "issue"},
            "direct metadata field",
        ),
        (
            [{"role": "user", "content": "{problem_statement!r}"}],
            {"problem_statement": "issue"},
            "conversion",
        ),
        (
            [{"role": "user", "content": "{problem_statement:>10}"}],
            {"problem_statement": "issue"},
            "format spec",
        ),
    ],
)
def test_task_prompt_template_rejects_invalid_metadata_formatting(prompt_template, metadata, error):
    with pytest.raises(ValueError, match=error):
        TaskConfig(
            sandbox=_LOCAL_SANDBOX,
            prompt=[{"role": "user", "content": "Fix the parser"}],
            prompt_template=prompt_template,
            metadata=metadata,
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("prompt_template", "error"),
    [
        (["not-a-message"], "template message"),
        ([{"content": "{problem_statement}"}], "template message.*role"),
        ([{"role": "user", "content": ["{problem_statement}"]}], "template message.*content"),
    ],
)
def test_task_prompt_template_rejects_incompatible_template_messages(prompt_template, error):
    with pytest.raises(ValueError, match=error):
        TaskConfig(
            sandbox=_LOCAL_SANDBOX,
            prompt=[{"role": "user", "content": "Fix the parser"}],
            prompt_template=prompt_template,
            metadata={"problem_statement": "Fix the parser"},
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_recipe_prompt_template_overrides_sample_template_and_uses_metadata():
    recipe_template = [
        {"role": "system", "content": "Recipe instructions"},
        {"role": "user", "content": "Issue: {problem_statement}"},
    ]
    resolver = TaskConfigResolver(
        {
            "swe_bench": {
                "name": "swe_bench",
                "prompt_template": recipe_template,
            }
        }
    )
    resolved = resolver.resolve(
        {
            "name": "swe_bench",
            "prompt": [{"role": "user", "content": "Source problem"}],
            "prompt_template": [{"role": "user", "content": "Sample override: {problem_statement}"}],
            "metadata": {"problem_statement": "Metadata problem"},
        }
    )

    config = TaskConfig(sandbox=_LOCAL_SANDBOX, **resolved)

    assert config.prompt == [
        {"role": "system", "content": "Recipe instructions"},
        {"role": "user", "content": "Issue: Metadata problem"},
    ]
