"""Pinned v2 contract; no executable or network required."""

import asyncio
import json

import pytest
from test_opencode_agent import SUCCESS, FakeSandbox

from uni_agent.agents import build_agent
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent
from uni_agent.agents.opencode2 import OpenCode2Agent, OpenCode2Config
from uni_agent.tasks.base import TaskConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def config(**kwargs):
    return OpenCode2Config(
        model=ModelConfig(base_url="http://gateway/sessions/test/v1", model_name="org/policy", api_key="test-key"),
        **kwargs,
    )


def run(**kwargs):
    sandbox = FakeSandbox(version="opencode v2.2.5", **kwargs)
    result = asyncio.run(
        OpenCode2Agent(config()).run(
            sandbox=sandbox, messages=[{"role": "user", "content": "solve"}], workdir="/workspace with 'quote"
        )
    )
    return result, sandbox


def test_v2_registry_and_private_native_launch():
    task = TaskConfig(sandbox={"provider": "local"}, agent={"name": "opencode2"})
    assert isinstance(task.agent, OpenCode2Config)
    assert isinstance(build_agent(task.agent), OpenCode2Agent)
    result, sandbox = run()
    assert result.finished and result.info["cli_version"] == "opencode v2.2.5"
    argv, options = next(item for item in sandbox.commands if item[0][1:2] == ["run"])
    assert "--standalone" in argv and "--pure" not in argv and "--dir" not in argv
    env = options["env"]
    assert env["PWD"] == options["workdir"] == "/workspace with 'quote"
    assert env["OPENCODE_DB"].startswith(config().state_root + "/")
    assert env["OPENCODE_TEST_HOME"] == env["HOME"]
    assert env["OPENCODE_CONFIG_PROJECT_DISABLE"] == "1"
    runtime = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert "provider" not in runtime and "permission" not in runtime
    assert runtime["providers"]["uni-agent"]["package"] == "@opencode/ai/providers/openai-compatible"
    assert "test-key" not in env["OPENCODE_CONFIG_CONTENT"]
    check = next(options for args, options in sandbox.commands if args[-1] == "--version")
    assert "UNI_AGENT_OPENCODE_API_KEY" not in check["env"]


@pytest.mark.parametrize("version", ["2.2.5", "opencode v2.2.6", "opencode v2.2.5-beta", "1.19.3", ""])
def test_other_versions_fail_before_prompt(version):
    sandbox = FakeSandbox(version=version)
    result = asyncio.run(OpenCode2Agent(config()).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))
    assert result.info["error_kind"] == "unsupported_version"
    assert not any(argv[1:2] == ["run"] for argv, _ in sandbox.commands)
    assert not OpenCodeAgent._accept_version("opencode v2.2.5")


@pytest.mark.parametrize(
    "stdout,exit_code,error",
    [
        ("", 0, "protocol_error"),
        ('{"type":"unknown"}', 0, "protocol_error"),
        (SUCCESS.rsplit("\n", 1)[0], 0, "incomplete"),
        (SUCCESS, 1, "agent_failure"),
        (SUCCESS, -1, "timeout"),
    ],
)
def test_failure_cannot_be_reported_as_completion(stdout, exit_code, error):
    result, _ = run(stdout=stdout, exit_code=exit_code)
    assert not result.finished and result.info["error_kind"] == error


def test_v2_model_variant_separator_rejected():
    cfg = config()
    cfg.model.model_name = "model#variant"
    with pytest.raises(ValueError, match="variant"):
        OpenCode2Agent._runtime_config(cfg)


def test_cleanup_and_redaction_are_retained():
    result, _ = run(stdout=SUCCESS.replace("DONE 中文", "test-key"), cleanup_code=1)
    assert result.info["error_kind"] == "cleanup_failure"
    assert not result.finished and "test-key" not in str(result)
