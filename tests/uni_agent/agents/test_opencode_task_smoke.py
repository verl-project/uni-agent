"""Opt-in Task verification with the real CLI and a scripted endpoint."""

import os

import pytest

from examples.opencode.smoke_task import run_smoke

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


@pytest.mark.parametrize("scenario,reward,requests", [("patch", 1.0, 2), ("no-edit", 0.0, 1)])
async def test_smoke_task_scores_files_not_cli_completion(tmp_path, monkeypatch, scenario, reward, requests):
    cli = os.environ.get("OPENCODE_BIN")
    if not cli:
        pytest.skip("set OPENCODE_BIN explicitly to enable the optional OpenCode Task smoke test")
    monkeypatch.setenv("PWD", str(tmp_path))
    result = await run_smoke(cli, scenario)
    assert result.finished is True
    assert result.reward == reward
    assert result.accuracy == reward
    assert result.extra_info["requests"] == requests
    assert result.extra_info["workspace_removed"] is True
    assert result.extra_info["agent_error"] is None
    assert result.extra_info["verifier_exit_code"] == (0 if reward else 1)
    assert not (tmp_path / "calculator.py").exists()


async def test_smoke_task_cleans_up_without_scoring_when_cli_is_missing(tmp_path):
    result = await run_smoke(str(tmp_path / "missing-opencode"))
    assert result.finished is False
    assert result.reward == 0.0
    assert result.extra_info["agent_error"] == "cli_unavailable"
    assert result.extra_info["requests"] == 0
    assert result.extra_info["verifier_exit_code"] is None
    assert result.extra_info["workspace_removed"] is True
