from __future__ import annotations

from copy import deepcopy

from uni_agent.tasks import TaskConfig, TaskConfigResolver, get_task


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
