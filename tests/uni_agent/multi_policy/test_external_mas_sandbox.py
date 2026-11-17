"""Tests for the sandbox execution path of the external MAS Task."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from examples.multi_policy.external_mas import task as task_mod
from examples.multi_policy.external_mas.execution import (
    ExternalMASError,
    ExternalMASRequest,
    ExternalMASTimeout,
    run_external_mas,
)

TASK_NAME = "multi_policy_external_mas"

ROLE_URLS = {
    "verifier": "http://gateway/sessions/g1:verifier/v1",
    "answerer": "http://gateway/sessions/g1:answerer/v1",
}

INJECTION = {
    "global_base_url_path": "llm.base_url",
    "agents_path": "agents",
    "agent_base_url_path": "llm.base_url",
    "agent_model_path": "model",
    "inject_agent_base_url": True,
}

RESULT = {
    "mode": "stdout_json",
    "field_mapping": {
        "final_result": {"fields": ["final_answer", "final_response"]},
        "reward_info": {"final_answer": "final_answer"},
    },
}


class FakeSandbox:
    """Minimal ``Sandbox`` stand-in that records writes and executions."""

    def __init__(self, *, exit_code=0, stdout='{"final_answer": "Paris"}', stderr=""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.written: dict[str, str] = {}
        self.execs: list[dict] = []
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def __aenter__(self) -> FakeSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content

    def _result(self) -> SimpleNamespace:
        return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr=self.stderr)

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        self.execs.append({"argv": list(argv), "timeout": timeout, "workdir": workdir, "env": env})
        if list(argv)[:1] == ["rm"]:
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        return self._result()

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.execs.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return self._result()


def _template(tmp_path, roles=("verifier", "answerer")) -> str:
    path = tmp_path / "inference.yaml"
    path.write_text(
        yaml.safe_dump({"agents": {role: {} for role in roles}, "llm": {}}),
        encoding="utf-8",
    )
    return str(path)


def _request(session_group_id: str = "g1") -> ExternalMASRequest:
    return ExternalMASRequest(
        prompt="who is it?",
        role_urls=ROLE_URLS,
        session_group_id=session_group_id,
    )


def _launcher_kwargs(tmp_path, *, argv=("python", "app.py", "--config", "{config_path}"), work_dir="/app"):
    return {
        "command": {"argv": list(argv), "work_dir": work_dir},
        "config": {"template_path": _template(tmp_path), "injection": INJECTION},
        "result": RESULT,
    }


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_sandbox_backend_writes_config_and_runs_command(tmp_path):
    sandbox = FakeSandbox()

    parsed = await run_external_mas(
        request=_request(),
        timeout={"seconds": 30},
        sandbox=sandbox,
        **_launcher_kwargs(tmp_path),
    )

    assert parsed["final_result"] == "Paris"
    assert parsed["reward_info"]["final_answer"] == "Paris"

    # The rendered config lives in the sandbox, not on the host.
    assert len(sandbox.written) == 1
    config_path, payload = next(iter(sandbox.written.items()))
    assert config_path.startswith("/tmp/uni_mas_g1_")
    injected = yaml.safe_load(payload)
    assert injected["llm"]["base_url"] == ROLE_URLS["verifier"]
    assert injected["agents"]["verifier"]["llm"]["base_url"] == ROLE_URLS["verifier"]
    assert injected["agents"]["answerer"]["llm"]["base_url"] == ROLE_URLS["answerer"]
    assert injected["agents"]["answerer"]["model"] == "answerer"

    # The command receives that sandbox path and the configured work dir.
    run = next(entry for entry in sandbox.execs if entry["argv"][:1] != ["rm"])
    assert run["argv"][3] == config_path
    assert run["workdir"] == "/app"
    assert run["timeout"] == 30.0

    # The per-rollout config is cleaned up inside the sandbox.
    assert sandbox.execs[-1]["argv"][:2] == ["rm", "-f"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_sandbox_backend_uses_shell_when_configured(tmp_path):
    sandbox = FakeSandbox()
    kwargs = _launcher_kwargs(tmp_path)
    kwargs["command"] = {"template": "python app.py --config {config_path}", "shell": True}

    await run_external_mas(request=_request(), sandbox=sandbox, **kwargs)

    shell_runs = [entry for entry in sandbox.execs if "script" in entry]
    assert len(shell_runs) == 1
    config_path = next(iter(sandbox.written))
    assert shell_runs[0]["script"] == f"python app.py --config {config_path}"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_sandbox_backend_non_zero_exit_keeps_output_and_logs(tmp_path):
    sandbox = FakeSandbox(exit_code=3, stdout='{"partial": true}', stderr="boom")
    logs = {"directory": str(tmp_path / "external_logs")}

    with pytest.raises(ExternalMASError) as excinfo:
        await run_external_mas(
            request=_request(),
            logs=logs,
            sandbox=sandbox,
            **_launcher_kwargs(tmp_path),
        )

    assert excinfo.value.exit_code == 3
    assert excinfo.value.stdout == '{"partial": true}'
    assert excinfo.value.stderr == "boom"
    written = list((tmp_path / "external_logs").glob("*"))
    assert written
    assert any("boom" in path.read_text(encoding="utf-8") for path in written)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_sandbox_backend_timeout_is_reported(tmp_path):
    # The base sandbox contract reports a timeout as exit_code == -1.
    sandbox = FakeSandbox(exit_code=-1, stdout="", stderr="local exec timed out")

    with pytest.raises(ExternalMASTimeout):
        await run_external_mas(
            request=_request(),
            timeout={"seconds": 5},
            sandbox=sandbox,
            **_launcher_kwargs(tmp_path),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_rejects_missing_sandbox(tmp_path):
    with pytest.raises(ValueError, match="requires a started sandbox"):
        await run_external_mas(request=_request(), sandbox=None, **_launcher_kwargs(tmp_path))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_task_runs_through_build_sandbox(tmp_path, monkeypatch):
    sandbox = FakeSandbox()
    monkeypatch.setattr(task_mod.ExternalMASTask, "build_sandbox", lambda self: sandbox)

    config = task_mod.ExternalMASTaskConfig(
        name=TASK_NAME,
        prompt=[{"role": "user", "content": "who is it?"}],
        ground_truth=["Paris"],
        session_group_id="g1",
        role_urls=dict(ROLE_URLS),
        sandbox={"provider": "local"},
        launcher_kwargs=_launcher_kwargs(tmp_path),
    )
    result = await task_mod.ExternalMASTask(config).run()

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.extra_info["final_answer"] == "Paris"
    assert sandbox.started == 1
    assert sandbox.stopped == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_task_config_requires_a_sandbox(tmp_path):
    # The sandbox owns execution now, so a Task Config without it is invalid.
    with pytest.raises(ValidationError, match="sandbox"):
        task_mod.ExternalMASTaskConfig(
            name=TASK_NAME,
            prompt=[{"role": "user", "content": "who is it?"}],
            session_group_id="g1",
            role_urls=dict(ROLE_URLS),
            launcher_kwargs=_launcher_kwargs(tmp_path),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_local_sandbox_backend_runs_a_real_command(tmp_path):
    from uni_agent.sandbox import SandboxConfig, build_sandbox

    script = "import json, sys;print(json.dumps({'final_answer': 'Paris', 'config': sys.argv[1]}))"
    argv = [sys.executable, "-c", script, "{config_path}"]

    async with build_sandbox(SandboxConfig(provider="local")) as sandbox:
        parsed = await run_external_mas(
            request=_request(),
            timeout={"seconds": 60},
            sandbox=sandbox,
            **_launcher_kwargs(tmp_path, argv=argv, work_dir=str(tmp_path)),
        )

    assert parsed["final_result"] == "Paris"
