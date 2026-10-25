"""CPU tests for external MAS command placeholder formatting."""

from __future__ import annotations

import pytest

from examples.multi_policy.external_mas.external_mas_runner import (
    _build_command,
    _format_values,
)

_CONTEXT = {
    "config_path": "/tmp/mas.yaml",
    "prompt": "hello world",
    "session_group_id": "group-1",
}


@pytest.mark.cpu
@pytest.mark.level0
def test_argv_preserves_literal_braces():
    command, shell = _build_command(
        {"argv": ["awk", "{print $1}", "{prompt}"]}, _CONTEXT
    )

    assert command == ["awk", "{print $1}", "hello world"]
    assert shell is False


@pytest.mark.cpu
@pytest.mark.level0
def test_env_values_preserve_json():
    assert _format_values(
        {"MAS_CONFIG": '{"key": 1}', "PROMPT": "{prompt}"}, _CONTEXT
    ) == {"MAS_CONFIG": '{"key": 1}', "PROMPT": "hello world"}


@pytest.mark.cpu
@pytest.mark.level0
def test_shell_template_preserves_json_and_quotes_substitution():
    command, shell = _build_command(
        {"template": 'echo \'{"a": 1}\' {prompt}', "shell": True},
        _CONTEXT,
    )

    assert command == "echo '{\"a\": 1}' 'hello world'"
    assert shell is True


@pytest.mark.cpu
@pytest.mark.level0
def test_shell_parameter_expansion_is_literal():
    command, shell = _build_command(
        {"template": "echo ${HOME} ${prompt}", "shell": True}, _CONTEXT
    )

    assert command == "echo ${HOME} ${prompt}"
    assert shell is True


@pytest.mark.cpu
@pytest.mark.level0
def test_unsupported_gateway_url_placeholder_is_rejected():
    with pytest.raises(ValueError, match="gateway_url"):
        _build_command({"argv": ["mas", "--gateway", "{gateway_url}"]}, _CONTEXT)
    with pytest.raises(ValueError, match="gateway_url"):
        _format_values({"GATEWAY": "{gateway_url}"}, _CONTEXT)


@pytest.mark.cpu
@pytest.mark.level0
def test_substitution_is_not_recursive():
    context = {**_CONTEXT, "prompt": "{session_group_id}"}

    command, _ = _build_command({"argv": ["{prompt}"]}, context)

    assert command == ["{session_group_id}"]


@pytest.mark.cpu
@pytest.mark.level0
def test_non_shell_template_preserves_json():
    command, shell = _build_command(
        {"template": 'python -c \'print({"a": 1})\' {prompt}'}, _CONTEXT
    )

    assert command == ["python", "-c", 'print({"a": 1})', "hello world"]
    assert shell is False
