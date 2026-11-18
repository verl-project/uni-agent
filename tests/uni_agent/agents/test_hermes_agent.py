"""Host-side tests for the Hermes sidecar adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from uni_agent.agents.base import ModelConfig, RequestSamplingConfig
from uni_agent.agents.hermes.agent import (
    RUNNER_FINALIZATION_GRACE_SECONDS,
    HermesAgent,
    HermesConfig,
    build_runner_command,
    parse_result_stdout,
    validate_messages,
    validate_result,
)
from uni_agent.sandbox.base import ExecResult, SandboxConfig


class FakeSandbox:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stderr: str = "",
        result_overrides: dict | None = None,
        result_bytes: bytes | None = None,
    ):
        self.exit_code = exit_code
        self.stderr = stderr
        self.result_overrides = result_overrides or {}
        self.result_bytes = result_bytes
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.files[path] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def exec_shell(self, script: str, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.commands.append(script)
        result_path = re.search(r"--result-path (\S+)", script).group(1).strip("'")
        run_id = result_path.removeprefix("/tmp/hermes-").removesuffix("/result.json")
        if self.exit_code == 0:
            envelope = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "finished": True,
                "final_response": "done",
                "error": None,
                "stop_reason": "text_response",
                "stats": {"api_calls": 3},
            }
            envelope.update(self.result_overrides)
            self.files[result_path] = (
                self.result_bytes if self.result_bytes is not None else json.dumps(envelope).encode()
            )
        return ExecResult(exit_code=self.exit_code, stdout="noise", stderr=self.stderr)


def _agent(**kwargs) -> HermesAgent:
    kwargs.setdefault("model", ModelConfig(base_url="http://gateway/sessions/s/v1", model_name="policy"))
    return HermesAgent(HermesConfig(**kwargs))


@pytest.mark.cpu
@pytest.mark.level0
def test_validate_messages_accepts_system_user_prefix_and_rejects_other_roles():
    validate_messages(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "issue"},
        ]
    )
    with pytest.raises(ValueError, match="at most 2"):
        validate_messages(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "issue"},
                {"role": "user", "content": "details"},
            ]
        )
    with pytest.raises(ValueError, match="only supports"):
        validate_messages([{"role": "assistant", "content": "hidden context"}, {"role": "user", "content": "issue"}])


@pytest.mark.cpu
@pytest.mark.level0
def test_command_does_not_contain_prompt_or_key():
    command = build_runner_command(
        input_path="/tmp/hermes/run/input.json",
        result_path="/tmp/hermes/run/result.json",
        messages_path="/tmp/hermes/run/messages.json",
        log_path="/tmp/hermes/run/runner.log",
        hermes_home="/tmp/hermes/run/home",
        tool_python="/opt/hermes/bin/python",
        runner_script="/opt/hermes/bin/run_hermes.py",
        terminal_timeout=600,
        approval_mode="off",
    )
    assert "/tmp/hermes/run/input.json" in command
    assert "--result-path" in command and "--messages-path" in command and "<" in command
    assert "prompt" not in command and "api_key" not in command
    assert "HERMES_INTERACTIVE=0" in command
    assert "HERMES_YOLO_MODE=1" in command
    assert "umask" not in command
    assert "TERMINAL_TIMEOUT=600" in command
    assert "TERMINAL_TIMEOUT=600.0" not in command


@pytest.mark.cpu
@pytest.mark.level0
def test_runner_command_inherits_sandbox_environment_and_quotes_paths(tmp_path):
    task_bin = tmp_path / "task runtime" / "bin"
    task_bin.mkdir(parents=True)
    (task_bin / "python").symlink_to(sys.executable)
    episode = tmp_path / "episode 'quoted'"
    episode.mkdir()
    runner = episode / "probe runner.py"
    runner.write_text(
        "import json, os, shutil, sys\n"
        "print(json.dumps({\n"
        "    'path': os.environ['PATH'], 'prefix': os.environ['CONDA_PREFIX'],\n"
        "    'env_name': os.environ['CONDA_DEFAULT_ENV'], 'home': os.environ['HERMES_HOME'],\n"
        "    'tool_python': shutil.which('python'),\n"
        "    'payload': json.load(sys.stdin), 'args': sys.argv[1:]\n"
        "}))\n",
        encoding="utf-8",
    )
    input_path = episode / "input.json"
    payload = {"messages": [{"role": "user", "content": "an issue with 'quotes' and 中文"}]}
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    result_path, messages_path, log_path = (episode / name for name in ("result.json", "messages.json", "runner.log"))
    hermes_home = episode / "home"
    config = HermesConfig(tool_python=sys.executable, runner_script=str(runner))
    command = build_runner_command(
        input_path=str(input_path),
        result_path=str(result_path),
        messages_path=str(messages_path),
        log_path=str(log_path),
        hermes_home=str(hermes_home),
        tool_python=config.tool_python,
        runner_script=config.runner_script,
        terminal_timeout=config.terminal_timeout,
        approval_mode=config.approval_mode,
    )
    environment = {
        **os.environ,
        "PATH": f"{task_bin}{os.pathsep}{os.environ['PATH']}",
        "CONDA_PREFIX": str(task_bin.parent),
        "CONDA_DEFAULT_ENV": "sandbox-runtime",
    }

    process = subprocess.run(
        ["bash", "-c", command], env=environment, capture_output=True, text=True, check=True, timeout=10
    )
    observed = json.loads(process.stdout)

    assert observed["path"] == environment["PATH"]
    assert observed["prefix"] == environment["CONDA_PREFIX"]
    assert observed["env_name"] == "sandbox-runtime"
    assert observed["tool_python"] == str(task_bin / "python")
    assert observed["home"] == str(hermes_home)
    assert observed["payload"] == payload
    assert observed["args"] == [
        "--result-path",
        str(result_path),
        "--messages-path",
        str(messages_path),
        "--log-path",
        str(log_path),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_parse_and_validate_marked_result():
    result = {"schema_version": 1, "run_id": "run-1", "status": "completed", "finished": True}
    parsed = parse_result_stdout("untrusted json: {}\nHERMES_RECIPE_RESULT " + json.dumps(result))
    assert validate_result(parsed, run_id="run-1") == result
    assert parse_result_stdout(json.dumps(result)) is None
    with pytest.raises(ValueError, match="mismatch"):
        validate_result(result, run_id="other")
    with pytest.raises(ValueError, match="agree"):
        validate_result({**result, "status": "timeout"}, run_id="run-1")


@pytest.mark.cpu
@pytest.mark.level0
def test_run_reads_result_and_preserves_endpoint_and_messages():
    sandbox = FakeSandbox()
    agent = _agent(
        sampling_params_override=RequestSamplingConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens_per_turn=8192,
        )
    )
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "fix it"}]
    result = asyncio.run(agent.run(sandbox=sandbox, messages=messages, workdir="/testbed"))
    assert result.finished is True
    assert result.output["status"] == "completed"
    assert "error" not in result.info
    assert result.transcript == messages
    assert len(sandbox.commands) == 1
    payload = json.loads(next(iter(sandbox.files.values())))
    assert payload["model"]["endpoint"] == "http://gateway/sessions/s/v1"
    assert payload["messages"] == messages
    assert payload["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    assert payload["limits"]["max_tokens"] == 8192
    assert payload["limits"]["run_budget_seconds"] == 7200 - RUNNER_FINALIZATION_GRACE_SECONDS


@pytest.mark.cpu
@pytest.mark.level0
def test_run_timeout_is_the_single_episode_budget_and_must_leave_finalization_time():
    with pytest.raises(ValueError, match="run_timeout"):
        HermesConfig(run_timeout=RUNNER_FINALIZATION_GRACE_SECONDS)

    agent = _agent(run_timeout=1234)
    sandbox = FakeSandbox()
    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "fix it"}]))
    payload = json.loads(next(iter(sandbox.files.values())))
    assert payload["limits"]["run_budget_seconds"] == 1234 - RUNNER_FINALIZATION_GRACE_SECONDS


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout_is_not_treated_as_success():
    result = asyncio.run(
        _agent().run(
            sandbox=FakeSandbox(exit_code=-1, stderr="exec timed out"),
            messages=[{"role": "user", "content": "fix it"}],
        )
    )
    assert result.finished is False
    assert result.output["status"] == "timeout"
    assert result.info["error"] == "exec timed out"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "sandbox_kwargs,expected_error",
    [
        (
            {"result_overrides": {"status": "error", "finished": False, "error": "API request failed"}},
            "API request failed",
        ),
        ({"result_bytes": b"not JSON"}, "missing or malformed Hermes runner result"),
    ],
    ids=["api-error", "malformed-result"],
)
def test_failed_run_exposes_error_to_task(sandbox_kwargs, expected_error):
    result = asyncio.run(
        _agent().run(
            sandbox=FakeSandbox(**sandbox_kwargs),
            messages=[{"role": "user", "content": "fix it"}],
        )
    )
    assert result.finished is False
    assert result.info["status"] == "error"
    assert result.info["error_kind"] == "agent_failure"
    assert result.info["error"] == expected_error


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("source", ["runner", "timeout"])
@pytest.mark.parametrize("boundary_secret", [None, "endpoint", "api_key"])
def test_error_diagnostics_redact_secrets_before_truncation(source, boundary_secret):
    endpoint = "http://gateway/sessions/private-session/v1"
    api_key = "private-gateway-key"
    bearer = "private-bearer-token"
    error = f"request failed for {endpoint} credential={api_key}; Authorization: Bearer {bearer}\n" + "x" * 5000
    if boundary_secret is not None:
        secret = endpoint if boundary_secret == "endpoint" else api_key
        error = "x" * 3990 + secret + " trailing text"
    sandbox = (
        FakeSandbox(result_overrides={"status": "error", "finished": False, "error": error})
        if source == "runner"
        else FakeSandbox(exit_code=-1, stderr=error)
    )
    result = asyncio.run(
        _agent(model=ModelConfig(base_url=endpoint, model_name="policy", api_key=api_key)).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "fix it"}],
        )
    )
    diagnostic = result.info["error"]
    assert len(diagnostic) <= 4000
    if boundary_secret is None:
        assert diagnostic.startswith("request failed for <redacted> credential=<redacted>;")
    else:
        assert diagnostic == "x" * 3990 + "<redacted>"
        assert secret[:10] not in diagnostic
    for secret in (endpoint, api_key, bearer):
        assert secret not in diagnostic
    assert result.finished is False
    assert result.info["status"] == ("timeout" if source == "timeout" else "error")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("boundary_secret", ["endpoint", "api_key"])
def test_error_diagnostics_redact_secret_prefix_truncated_by_sidecar(boundary_secret):
    import importlib.util
    from pathlib import Path

    runner_path = Path(__file__).resolve().parents[3] / "examples/blackbox_recipes/hermes_swe_task/run_hermes.py"
    spec = importlib.util.spec_from_file_location("hermes_recipe_runner_mapping", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    endpoint = "http://gateway/sessions/private-session/v1"
    api_key = "private-gateway-key"
    secret = endpoint if boundary_secret == "endpoint" else api_key
    mapped = runner.map_hermes_result({"error": "x" * 3990 + secret + " trailing text", "failed": True})
    assert len(mapped["error"]) == 4000
    assert mapped["error"].endswith(secret[:10])
    mapped.pop("run_id")  # The fake sandbox supplies the current invocation's run id.

    result = asyncio.run(
        _agent(model=ModelConfig(base_url=endpoint, model_name="policy", api_key=api_key)).run(
            sandbox=FakeSandbox(result_overrides=mapped),
            messages=[{"role": "user", "content": "fix it"}],
        )
    )

    assert result.info["error"] == "x" * 3990 + "<redacted>"
    assert secret[:10] not in result.info["error"]
    assert result.output["error"] == result.info["error"]
    assert result.finished is False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("agent_error", [None, "API request failed"])
def test_swe_task_keeps_hermes_error_without_changing_reward(monkeypatch, agent_error):
    from uni_agent.tasks.swe_bench import reward
    from uni_agent.tasks.swe_bench.task import SWEBenchTask, SWEBenchTaskConfig

    sandbox = FakeSandbox(
        result_overrides={"status": "error", "finished": False, "error": agent_error} if agent_error else None
    )

    @asynccontextmanager
    async def build_sandbox():
        yield sandbox

    task = SWEBenchTask(
        SWEBenchTaskConfig(
            sandbox=SandboxConfig(provider="docker", image="test-image"),
            agent=_agent().config,
            metadata={"instance_id": "test"},
            prompt=[{"role": "user", "content": "fix it"}],
        )
    )
    monkeypatch.setattr(task, "build_sandbox", build_sandbox)
    scorer_result = {"resolved": True, "eval_report": {"found_eval_status": True}}
    monkeypatch.setattr(reward, "compute_reward", AsyncMock(return_value=scorer_result))

    result = asyncio.run(task.run())

    assert result.reward == result.accuracy == 1.0
    assert result.finished is (agent_error is None)
    assert result.extra_info["eval_report"] == scorer_result["eval_report"]
    assert "agent_error" not in scorer_result
    if agent_error is None:
        assert "agent_error" not in result.extra_info
    else:
        assert result.extra_info["agent_error"] == agent_error
