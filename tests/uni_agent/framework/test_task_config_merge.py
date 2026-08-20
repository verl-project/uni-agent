from __future__ import annotations

from copy import deepcopy

import pytest

from uni_agent.tasks import TaskConfig, TaskConfigResolver, get_task

_LOCAL_SANDBOX = {"provider": "local"}


def test_task_config_has_no_logging_runtime_fields():
    assert "log_dir" not in TaskConfig.model_fields


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


def test_task_prompt_without_template_passes_through_unchanged():
    messages = [
        {"role": "system", "content": "Existing instructions"},
        {"role": "user", "content": "Existing rendered problem"},
    ]

    config = TaskConfig(sandbox=_LOCAL_SANDBOX, prompt=messages)

    assert config.prompt == messages


def test_task_prompt_template_renders_source_user_message():
    config = TaskConfig(
        sandbox=_LOCAL_SANDBOX,
        prompt=[{"role": "user", "content": "Fix the parser"}],
        prompt_template=[
            {"role": "system", "content": "Work carefully."},
            {"role": "user", "content": "Issue:\n{prompt}\nUse {{literal braces}}."},
        ],
    )

    assert config.prompt == [
        {"role": "system", "content": "Work carefully."},
        {"role": "user", "content": "Issue:\nFix the parser\nUse {literal braces}."},
    ]


def test_exact_prompt_substitution_preserves_structured_content():
    structured_content = [
        {"type": "text", "text": "Inspect this image"},
        {"type": "image", "image": {"bytes": b"image-bytes"}},
    ]

    config = TaskConfig(
        sandbox=_LOCAL_SANDBOX,
        prompt=[{"role": "user", "content": structured_content}],
        prompt_template=[{"role": "user", "content": "{prompt}"}],
    )

    assert config.prompt == [{"role": "user", "content": structured_content}]
    assert isinstance(config.prompt[0]["content"], list)


@pytest.mark.parametrize(
    ("prompt_template", "error"),
    [
        ([{"role": "user", "content": "No field"}], "exactly one.*prompt"),
        (
            [
                {"role": "system", "content": "{prompt}"},
                {"role": "user", "content": "{prompt}"},
            ],
            "exactly one.*prompt",
        ),
        ([{"role": "user", "content": "{metadata}"}], "unknown.*metadata"),
        ([{"role": "user", "content": "{prompt!r}"}], "conversion"),
        ([{"role": "user", "content": "{prompt:>20}"}], "format spec"),
    ],
)
def test_task_prompt_template_rejects_invalid_fields(prompt_template, error):
    with pytest.raises(ValueError, match=error):
            TaskConfig(
                sandbox=_LOCAL_SANDBOX,
                prompt=[{"role": "user", "content": "Fix the parser"}],
            prompt_template=prompt_template,
        )


@pytest.mark.parametrize(
    ("source_prompt", "error"),
    [
        ([], "exactly one source user message"),
        (
            [
                {"role": "system", "content": "Legacy instructions"},
                {"role": "user", "content": "Legacy problem"},
            ],
            "exactly one source user message",
        ),
        ([{"role": "assistant", "content": "Wrong role"}], "source user message"),
        ([{"role": "user"}], "source user message.*content"),
    ],
)
def test_task_prompt_template_rejects_incompatible_source_messages(source_prompt, error):
    with pytest.raises(ValueError, match=error):
            TaskConfig(
                sandbox=_LOCAL_SANDBOX,
                prompt=source_prompt,
            prompt_template=[{"role": "user", "content": "{prompt}"}],
        )


@pytest.mark.parametrize(
    ("prompt_template", "error"),
    [
        (["not-a-message"], "template message"),
        ([{"content": "{prompt}"}], "template message.*role"),
        ([{"role": "user"}], "template message.*content"),
        ([{"role": "user", "content": ["{prompt}"]}], "template message.*content"),
    ],
)
def test_task_prompt_template_rejects_incompatible_template_messages(prompt_template, error):
    with pytest.raises(ValueError, match=error):
            TaskConfig(
                sandbox=_LOCAL_SANDBOX,
                prompt=[{"role": "user", "content": "Fix the parser"}],
            prompt_template=prompt_template,
        )


def test_embedded_prompt_substitution_requires_text_source_content():
    with pytest.raises(ValueError, match="embedded.*string"):
        TaskConfig(
            sandbox=_LOCAL_SANDBOX,
            prompt=[{"role": "user", "content": [{"type": "image", "image": "example.png"}]}],
            prompt_template=[{"role": "user", "content": "Issue: {prompt}"}],
        )


def test_recipe_prompt_template_overrides_sample_template_without_metadata_context():
    recipe_template = [
        {"role": "system", "content": "Recipe instructions"},
        {"role": "user", "content": "Issue: {prompt}"},
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
            "prompt_template": [{"role": "user", "content": "Sample override: {prompt}"}],
            "metadata": {"problem_statement": "LEAKED METADATA", "patch": "SECRET PATCH"},
        }
    )

    config = TaskConfig(sandbox=_LOCAL_SANDBOX, **resolved)

    assert config.prompt_template == recipe_template
    assert config.prompt == [
        {"role": "system", "content": "Recipe instructions"},
        {"role": "user", "content": "Issue: Source problem"},
    ]
    assert "LEAKED METADATA" not in str(config.prompt)
    assert "SECRET PATCH" not in str(config.prompt)
