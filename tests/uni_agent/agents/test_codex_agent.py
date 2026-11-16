from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.codex.agent import CodexAgent, CodexConfig, build_agent_command, parse_agent_result
from uni_agent.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self, stdout: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[dict] = []
        self.files: dict[str, str] = {}

    async def write_file(self, path, content):
        self.files[path] = content

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.calls.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self.exit_code, stdout=self.stdout, stderr="")


def make_agent(**kwargs):
    kwargs.setdefault("codex_bin", "/opt/codex/vendor/x86_64-unknown-linux-musl/bin/codex")
    kwargs.setdefault("model", ModelConfig(base_url="http://gateway/v1", model_name="policy"))
    return CodexAgent(CodexConfig(**kwargs))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("inherit_settings", [False, True])
def test_direct_cli_launch_preserves_stdin_flags_and_environment(tmp_path, inherit_settings):
    binary = tmp_path / "codex cli's"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "keys = ['CODEX_HOME', 'OPENAI_API_KEY', 'CODEX_MANAGED_PACKAGE_ROOT', 'CONDA_DEFAULT_ENV', "
        "'CONDA_PREFIX', 'PATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy']\n"
        "home = pathlib.Path(os.environ['CODEX_HOME'])\n"
        "print(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(), "
        "'env': {key: os.environ.get(key) for key in keys}, 'cwd': os.getcwd(), "
        "'home_exists': home.is_dir(), 'config_exists': (home / 'config.toml').exists()}))\n"
    )
    binary.chmod(0o755)
    workdir = tmp_path / "repo's workdir"
    workdir.mkdir()
    home = tmp_path / "codex home's"
    marker = tmp_path / "must-not-exist"
    task = f"fix 'this'\n$(touch {marker})\nUnicode: 中文"
    key = 'key\'s "quoted" value'
    catalog = str(tmp_path / "model catalog's.json")
    env = {
        **os.environ,
        "CONDA_DEFAULT_ENV": "testbed",
        "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
        "PATH": f"/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:{os.environ['PATH']}",
        "HTTP_PROXY": "http://proxy.invalid:8080",
        "HTTPS_PROXY": "http://proxy.invalid:8080",
    }
    for name in ("CODEX_MANAGED_PACKAGE_ROOT", "NO_PROXY", "no_proxy"):
        env.pop(name, None)
    if inherit_settings:
        env.update(
            CODEX_MANAGED_PACKAGE_ROOT="/custom/package root", NO_PROXY="existing-upper", no_proxy="existing-lower"
        )
    command = build_agent_command(
        task_b64=base64.b64encode(task.encode()).decode(),
        codex_bin=str(binary),
        gateway_url="http://127.0.0.1:38197/sessions/s1/v1",
        model_name="policy's model",
        api_key=key,
        project_dir=str(workdir),
        model_catalog_path=catalog,
        codex_home=str(home),
    )
    process = subprocess.run(["bash", "-c", command], cwd=workdir, env=env, text=True, capture_output=True, check=True)
    captured = json.loads(process.stdout)
    assert captured["stdin"] == task
    assert task not in command and task not in captured["argv"]
    assert not marker.exists()
    assert captured["cwd"] == str(workdir)
    assert captured["home_exists"] and not captured["config_exists"]
    assert captured["env"]["CODEX_HOME"] == str(home)
    assert captured["env"]["OPENAI_API_KEY"] == key
    assert captured["env"]["CODEX_MANAGED_PACKAGE_ROOT"] == (
        "/custom/package root" if inherit_settings else "/opt/codex"
    )
    assert captured["env"]["NO_PROXY"] == ("existing-upper,127.0.0.1" if inherit_settings else "127.0.0.1")
    assert captured["env"]["no_proxy"] == ("existing-lower,127.0.0.1" if inherit_settings else "127.0.0.1")
    assert captured["env"]["HTTP_PROXY"] == captured["env"]["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    for name in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PATH"):
        assert captured["env"][name] == env[name]
    assert captured["argv"][:10] == [
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(workdir),
        "--model",
        "policy's model",
        "-c",
    ]
    args = captured["argv"][9:]
    assert all(flag == "-c" for flag in args[::2])
    overrides = {name: json.loads(value) for name, value in (item.split("=", 1) for item in args[1::2])}
    assert overrides == {
        "model_provider": "gateway",
        "model_providers.gateway.name": "Uni-Agent Gateway",
        "model_providers.gateway.base_url": "http://127.0.0.1:38197/sessions/s1/v1",
        "model_providers.gateway.wire_api": "responses",
        "model_providers.gateway.requires_openai_auth": True,
        "model_providers.gateway.env_key": "OPENAI_API_KEY",
        "disable_response_storage": True,
        "check_for_update_on_startup": False,
        "model_catalog_json": catalog,
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_recipe_configures_the_sandbox_environment():
    from uni_agent.sandbox.base import SandboxConfig
    from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox

    recipe = Path(__file__).resolve().parents[3] / "examples/blackbox_recipes/codex_swe_task/task_config_codex.yaml"
    config = yaml.safe_load(recipe.read_text())[0]
    sandbox = OpenyuanrongSandbox.from_config(SandboxConfig(image="swebench/test", **config["sandbox"]))
    assert sandbox.env == {
        "CONDA_DEFAULT_ENV": "testbed",
        "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
    }
    assert sandbox.add_to_path == ("/opt/miniconda3/envs/testbed/bin", "/opt/miniconda3/bin")
    assert sandbox.proxy_port == 38197


@pytest.mark.cpu
@pytest.mark.level0
def test_parse_agent_result_jsonl():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "fixed"}}),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    assert parse_agent_result(stdout, 0) == {
        "exit_status": "ok",
        "ok": True,
        "content": "fixed",
        "event_count": 3,
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_parse_agent_result_timeout_and_failure():
    timeout = parse_agent_result("", -1)
    assert timeout["exit_status"] == "timeout"
    failed = parse_agent_result(json.dumps({"type": "turn.failed", "error": {"message": "bad"}}), 1)
    assert failed["exit_status"] == "error"
    assert failed["error"] == "bad"


@pytest.mark.cpu
@pytest.mark.level0
def test_parse_agent_result_honors_process_exit_code():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}),
            json.dumps({"type": "turn.completed"}),
            json.dumps({"type": "process.completed", "exit_code": 17}),
        ]
    )
    failed = parse_agent_result(stdout, 0)
    assert failed["exit_status"] == "error"
    assert failed["ok"] is False
    assert failed["error"] == "codex exited with code 17"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "diagnostic output",
        json.dumps({"type": "thread.started", "thread_id": "t"}),
    ],
)
def test_parse_agent_result_rejects_missing_completion_events(stdout):
    result = parse_agent_result(stdout, 0)
    assert result["ok"] is False
    assert result["exit_status"] == "error"


@pytest.mark.cpu
@pytest.mark.level0
def test_codex_agent_runs_and_returns_agent_result():
    sandbox = FakeSandbox(
        stdout=json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
        + "\n"
        + json.dumps({"type": "turn.completed"})
    )
    agent = make_agent()
    result = asyncio.run(
        agent.run(
            sandbox=sandbox,
            messages=[
                {"role": "system", "content": "follow repository policy"},
                {"role": "user", "content": "fix bug"},
            ],
        )
    )
    assert isinstance(result, AgentResult)
    assert result.finished is True
    assert result.output["content"] == "done"
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["workdir"] == "/testbed"
    assert sandbox.calls[0]["env"] is None
    assert base64.b64encode(b"fix bug").decode() in sandbox.calls[0]["script"]
    assert base64.b64encode(b"follow repository policy").decode() not in sandbox.calls[0]["script"]


@pytest.mark.cpu
@pytest.mark.level0
def test_build_agent_command_passes_catalog_as_a_quoted_cli_override():
    path = "/tmp/model catalog's.json"
    command = build_agent_command(
        task_b64="",
        codex_bin="/opt/codex/vendor/x86_64-unknown-linux-musl/bin/codex",
        gateway_url="http://gateway/v1",
        model_name="policy",
        api_key="EMPTY",
        model_catalog_path=path,
    )
    assert shlex.split(command)[-2:] == ["-c", "model_catalog_json=" + json.dumps(path)]


@pytest.mark.cpu
@pytest.mark.level0
def test_codex_agent_copies_optional_catalog_into_sandbox(tmp_path):
    catalog = {"models": [{"slug": "policy", "supports_parallel_tool_calls": True}]}
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(catalog))
    sandbox = FakeSandbox(stdout=json.dumps({"type": "turn.completed"}))
    agent = make_agent(model_catalog_path=str(source))
    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "fix bug"}]))
    assert result.finished is True
    assert len(sandbox.files) == 1
    path, content = next(iter(sandbox.files.items()))
    assert path.startswith("/tmp/codex-model-catalog-") and path.endswith(".json")
    assert json.loads(content) == catalog
    assert "model_catalog_json=" + json.dumps(path) in sandbox.calls[0]["script"]
