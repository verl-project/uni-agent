"""Schema tests for the external-MAS Task Config and its ``launcher_kwargs``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from examples.multi_policy.external_mas.task import (
    ExternalMASLauncherConfig,
    ExternalMASTaskConfig,
)

ROOT = Path(__file__).resolve().parents[3]
TASK_CONFIG = ROOT / "examples" / "multi_policy" / "external_mas" / "task_config.yaml"


def _launcher(**overrides):
    """Valid launcher defaults so each test can invalidate exactly one thing."""
    launcher = {
        "command": {"argv": ["python"]},
        "config": {"template_path": "template.yaml"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(launcher.get(section), dict):
            launcher[section] = {**launcher[section], **values}
        else:
            launcher[section] = values
    return launcher


def _config(**launcher_kwargs):
    return ExternalMASTaskConfig(
        sandbox={"provider": "local"},
        session_group_id="g1",
        role_urls={"answerer": "http://gateway/sessions/g1:answerer/v1"},
        launcher_kwargs=_launcher(**launcher_kwargs),
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_checked_in_task_config_matches_the_schema():
    entries = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(entries, list) and entries

    for entry in entries:
        config = ExternalMASTaskConfig(
            **entry,
            session_group_id="g1",
            role_urls={"answerer": "http://gateway/sessions/g1:answerer/v1"},
        )
        assert config.sandbox is not None
        assert config.sandbox.provider == "local"
        assert isinstance(config.launcher_kwargs, ExternalMASLauncherConfig)
        assert config.launcher_kwargs.command.argv
        assert config.launcher_kwargs.config.template_path
        assert config.launcher_kwargs.config.config_dir == "/tmp"


@pytest.mark.cpu
@pytest.mark.level0
def test_sandbox_is_required():
    with pytest.raises(ValidationError, match="sandbox"):
        ExternalMASTaskConfig(
            session_group_id="g1",
            role_urls={"answerer": "http://gateway/sessions/g1:answerer/v1"},
            launcher_kwargs=_launcher(),
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_removed_execution_block_is_rejected():
    # The backend switch is gone: the sandbox is the only execution path, so a
    # leftover ``execution`` block must fail instead of being silently ignored.
    with pytest.raises(ValidationError, match="execution"):
        _config(execution={"backend": "local_process"})


@pytest.mark.cpu
@pytest.mark.level0
def test_template_path_is_required():
    with pytest.raises(ValidationError, match="template_path"):
        _config(config={"template_path": None})


@pytest.mark.cpu
@pytest.mark.level0
def test_command_needs_argv_or_template():
    with pytest.raises(ValidationError, match="argv or command.template"):
        _config(command={"argv": None, "template": None})


@pytest.mark.cpu
@pytest.mark.level0
def test_command_rejects_argv_and_template_together():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _config(command={"argv": ["python"], "template": "python {config_path}"})


@pytest.mark.cpu
@pytest.mark.level0
def test_command_rejects_argv_with_shell():
    with pytest.raises(ValidationError, match="cannot be combined with shell"):
        _config(command={"argv": ["python"], "shell": True})


@pytest.mark.cpu
@pytest.mark.level0
def test_config_dir_must_be_absolute():
    with pytest.raises(ValidationError, match="absolute sandbox path"):
        _config(config={"template_path": "template.yaml", "config_dir": "relative/dir"})


@pytest.mark.cpu
@pytest.mark.level0
def test_misspelled_launcher_key_is_rejected():
    with pytest.raises(ValidationError, match="commnad"):
        _config(commnad={"argv": ["python"]})


@pytest.mark.cpu
@pytest.mark.level0
def test_misspelled_command_key_is_rejected():
    with pytest.raises(ValidationError, match="argvs"):
        _config(command={"argvs": ["python"]})


@pytest.mark.cpu
@pytest.mark.level0
def test_unknown_result_mode_is_rejected():
    with pytest.raises(ValidationError, match="mode"):
        _config(result={"mode": "yaml_report"})


@pytest.mark.cpu
@pytest.mark.level0
def test_env_values_accept_yaml_scalars():
    # ``yes``/``1`` parse as bool/int in YAML; the runner has always stringified
    # them, so the schema must not reject them.
    config = _config(command={"argv": ["python"], "env": {"DEFAULT": True, "WORKERS": 3}})
    assert config.launcher_kwargs.command.env == {"DEFAULT": "True", "WORKERS": "3"}
