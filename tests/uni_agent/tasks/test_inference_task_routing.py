from __future__ import annotations

import pytest

from uni_agent.tasks import TaskConfigResolver


def test_resolver_loads_every_named_entry(tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: task_a
  sandbox:
    provider: local
  agent:
    name: react
    model:
      temperature: 0.2
- name: task_b
  sandbox:
    provider: modal
  agent:
    name: claude_code
""".strip()
    )
    resolver = TaskConfigResolver.from_file(str(config_path))

    assert set(resolver.defaults_by_name) == {"task_a", "task_b"}
    assert resolver.defaults_by_name["task_a"]["agent"]["model"]["temperature"] == 0.2


def test_resolver_routes_by_name_and_applies_sample_and_runtime_overrides():
    defaults = {
        "task_a": {
            "name": "task_a",
            "sandbox": {"provider": "local"},
            "agent": {
                "name": "react",
                "max_steps": 50,
                "model": {
                    "base_url": "http://model:8000/v1",
                    "model_name": "policy",
                    "api_key": "key",
                    "temperature": 0.8,
                },
            },
        },
        "task_b": {
            "name": "task_b",
            "sandbox": {"provider": "modal"},
            "agent": {
                "name": "react",
                "model": {
                    "base_url": "http://model:8000/v1",
                    "model_name": "policy",
                    "api_key": "key",
                },
            },
        },
    }
    sample = {
        "name": "task_a",
        "sandbox": {"image": "sample-image"},
        "agent": {"max_steps": 200, "model": {"temperature": 0.3}},
    }

    resolver = TaskConfigResolver(defaults)
    resolved = resolver.resolve(
        sample,
        runtime_model={
            "base_url": "http://runtime:8000/v1",
            "model_name": "runtime-policy",
            "api_key": "runtime-key",
        },
    )

    assert resolved["name"] == "task_a"
    assert resolved["sandbox"] == {"provider": "local", "image": "sample-image"}
    assert resolved["agent"]["max_steps"] == 200
    assert resolved["agent"]["model"]["temperature"] == 0.3
    assert resolved["agent"]["model"]["base_url"] == "http://runtime:8000/v1"
    assert resolved["agent"]["model"]["model_name"] == "runtime-policy"
    assert resolved["agent"]["model"]["api_key"] == "runtime-key"


def test_resolver_rejects_missing_route():
    resolver = TaskConfigResolver({"other": {"name": "other"}})
    with pytest.raises(ValueError, match="no Task Config for sample task 'missing'"):
        resolver.resolve({"name": "missing"})
