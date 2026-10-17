"""CPU contract tests; FakeSandbox does not run a model or the OpenCode CLI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from uni_agent.agents import build_agent, get_agent_cls
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.agents.opencode.agent import OPENCODE_VERSION
from uni_agent.tasks.base import TaskConfig


def event(kind, **part):
    return json.dumps({"type": kind, "sessionID": "ses_test", "part": part})


SUCCESS = "\n".join(
    [
        event("step_start", type="step-start"),
        event("text", type="text", text="DONE 中文"),
        event("step_finish", type="step-finish", reason="stop"),
    ]
)


class FakeSandbox:
    def __init__(
        self,
        *,
        stdout=SUCCESS,
        exit_code=0,
        version=OPENCODE_VERSION,
        error=None,
        prepare_code=0,
        cleanup_code=0,
        version_code=0,
        cleanup_error=None,
        stderr="",
    ):
        self.stdout, self.exit_code, self.version, self.error = stdout, exit_code, version, error
        self.prepare_code, self.cleanup_code, self.version_code = prepare_code, cleanup_code, version_code
        self.cleanup_error = cleanup_error
        self.stderr = stderr
        self.commands = []

    async def exec(self, argv, **kwargs):
        self.commands.append((argv, kwargs))
        if argv[0] == "mkdir":
            code, stdout = self.prepare_code, ""
        elif argv[0] == "rm":
            if self.cleanup_error:
                raise self.cleanup_error
            code, stdout = self.cleanup_code, ""
        elif argv[-1] == "--version":
            code, stdout = self.version_code, self.version
        else:
            if self.error:
                raise self.error
            code, stdout = self.exit_code, self.stdout
        return SimpleNamespace(exit_code=code, stdout=stdout, stderr=self.stderr if argv[1:2] == ["run"] else "")


def config(**kwargs):
    return OpenCodeConfig(
        model=ModelConfig(
            base_url="http://gateway/sessions/abc/v1/", model_name="org/policy", api_key="test-secret-only"
        ),
        **kwargs,
    )


def run(sandbox=None, cfg=None, messages=None, workdir="/testbed"):
    sandbox = sandbox or FakeSandbox()
    result = asyncio.run(
        OpenCodeAgent(cfg or config()).run(
            sandbox=sandbox,
            messages=messages if messages is not None else [{"role": "user", "content": "solve"}],
            workdir=workdir,
        )
    )
    return result, sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_and_task_config_roundtrip():
    task = TaskConfig(sandbox={"provider": "local"}, agent={"name": "opencode", "run_timeout": 42})
    assert isinstance(task.agent, OpenCodeConfig)
    assert get_agent_cls("opencode") is OpenCodeAgent
    assert isinstance(build_agent(task.agent), OpenCodeAgent)
    assert TaskConfig.model_validate(task.model_dump()).agent.run_timeout == 42


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "override",
    [
        {"run_timeout": 0},
        {"run_timeout": float("inf")},
        {"context_limit": 0},
        {"output_limit": -1},
        {"state_root": "/"},
        {"state_root": "relative"},
        {"state_root": "/tmp/../state"},
        {"unknown": True},
    ],
)
def test_invalid_config(override):
    with pytest.raises(ValidationError):
        config(**override)


@pytest.mark.cpu
@pytest.mark.level0
def test_launch_endpoint_prompt_and_no_secret_in_argv():
    prompt = "--help; $(touch /tmp/not-executed) `id` '\n中文"
    result, sandbox = run(
        messages=[{"role": "user", "content": prompt}], cfg=config(run_timeout=42), workdir="/workspace with spaces"
    )
    argv, options = sandbox.commands[2]
    assert argv == [
        "opencode",
        "run",
        "--pure",
        "--format",
        "json",
        "--model",
        "uni-agent/org/policy",
        "--agent",
        "build",
        "--title",
        "Uni-Agent rollout",
        "--",
        prompt,
    ]
    assert options["workdir"] == "/workspace with spaces"
    assert options["timeout"] == 42
    env = options["env"]
    runtime = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    provider = runtime["provider"]["uni-agent"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://gateway/sessions/abc/v1"
    assert provider["options"]["apiKey"] == "{env:UNI_AGENT_OPENCODE_API_KEY}"
    assert env["UNI_AGENT_OPENCODE_API_KEY"] == "test-secret-only"
    assert "test-secret-only" not in json.dumps(argv) + env["OPENCODE_CONFIG_CONTENT"]
    assert "UNI_AGENT_OPENCODE_API_KEY" not in sandbox.commands[1][1]["env"]
    assert runtime["enabled_providers"] == ["uni-agent"]
    assert runtime["compaction"] == {"auto": False, "prune": False}
    assert runtime["agent"]["title"]["disable"]
    assert runtime["permission"]["task"] == "deny"
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    root = sandbox.commands[0][0][-1]
    assert env["TMPDIR"] == root
    assert env["OPENCODE_DB"] == root + "/opencode.db"
    assert env["OPENCODE_TEST_HOME"] == root + "/home"
    assert env["OPENCODE_WORKSPACE_ID"] == ""
    assert json.loads(env["OPENCODE_PERMISSION"]) == runtime["permission"]
    assert sandbox.commands[-1][0] == ["rm", "-rf", "--", root]
    assert all(
        env[key].startswith(root + "/")
        for key in ("HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME")
    )
    assert result.finished is True
    assert result.output["text"] == "DONE 中文"
    assert result.transcript[-1] == {"role": "assistant", "content": "DONE 中文"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "x"}],
        [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
        [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        [{"role": "user", "content": " "}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        [{"role": "user", "content": "x\x00y"}],
    ],
)
def test_invalid_prompt_before_any_sandbox_operations(messages):
    sandbox = FakeSandbox()
    with pytest.raises(ValueError, match="opencode requires"):
        run(sandbox, messages=messages)
    assert not sandbox.commands


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "model",
    [
        {"base_url": None},
        {"base_url": "file:///tmp/data"},
        {"base_url": "http://user:password@host/v1"},
        {"base_url": "http://host/v1?key=secret"},
        {"base_url": "http://host/v1#x"},
        {"base_url": "http://host/{file:/tmp/x}"},
        {"model_name": None},
        {"model_name": " "},
        {"model_name": "{env:TOKEN}"},
        {"api_key": ""},
    ],
)
def test_invalid_model_before_sandbox_operations(model):
    cfg, sandbox = config(), FakeSandbox()
    cfg.model = cfg.model.model_copy(update=model)
    with pytest.raises(ValueError):
        run(sandbox, cfg=cfg)
    assert not sandbox.commands


@pytest.mark.cpu
@pytest.mark.level0
def test_context_and_workdir_constraints():
    for cfg, workdir in [(config(output_limit=32768), "/workspace"), (config(), "relative")]:
        sandbox = FakeSandbox()
        with pytest.raises(ValueError):
            run(sandbox, cfg=cfg, workdir=workdir)
        assert not sandbox.commands


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "sandbox, kind",
    [
        (FakeSandbox(exit_code=2), "agent_failure"),
        (FakeSandbox(exit_code=-1, stdout=""), "timeout"),
        (FakeSandbox(error=TimeoutError("test-secret-only")), "timeout"),
        (FakeSandbox(error=FileNotFoundError("test-secret-only")), "startup_failure"),
        (FakeSandbox(version="1.18.15"), "unsupported_version"),
        (FakeSandbox(version_code=127), "cli_unavailable"),
        (FakeSandbox(version_code=-1), "timeout"),
        (FakeSandbox(prepare_code=1), "startup_failure"),
        (FakeSandbox(cleanup_code=1), "cleanup_failure"),
        (FakeSandbox(cleanup_error=OSError("test-secret-only")), "cleanup_failure"),
    ],
)
def test_failures_and_cleanup(sandbox, kind):
    result, sandbox = run(sandbox)
    assert result.finished is False
    assert result.info["error_kind"] == kind
    assert sandbox.commands[-1][0][:3] == ["rm", "-rf", "--"]
    assert "test-secret-only" not in json.dumps(dataclasses.asdict(result))
    if kind in {"unsupported_version", "cli_unavailable", "startup_failure"}:
        assert not any(argv[1:2] == ["run"] for argv, _ in sandbox.commands) or sandbox.error


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "stdout,kind",
    [
        ("not json", "protocol_error"),
        ("", "protocol_error"),
        ("[]", "protocol_error"),
        ('{"type":"text","sessionID":"s","part":null}', "protocol_error"),
        ('{"type":"text","part":{}}', "protocol_error"),
        (SUCCESS + '\n{"type":"future","sessionID":"ses_test","part":{}}', "protocol_error"),
        (SUCCESS + '\n{"type":"text","sessionID":"other","part":{}}', "protocol_error"),
        (SUCCESS + "\ntrailing noise", "protocol_error"),
        (event("text", text="partial"), "incomplete"),
        (event("step_finish", reason="length"), "incomplete"),
        (event("step_finish", reason="tool-calls"), "incomplete"),
        (SUCCESS + "\n" + event("step_start", type="step-start"), "incomplete"),
        (SUCCESS + '\n{"type":"error","sessionID":"ses_test","error":{"name":"APIError"}}', "agent_failure"),
    ],
)
def test_event_protocol_and_incomplete_runs(stdout, kind):
    result, _ = run(FakeSandbox(stdout=stdout))
    assert result.finished is False
    assert result.info["error_kind"] == kind


@pytest.mark.cpu
@pytest.mark.level0
def test_tool_roundtrip_can_finish_after_recovered_tool_error():
    stdout = "\n".join(
        [
            event("tool_use", type="tool", state={"status": "error", "error": "retry"}),
            event("step_finish", reason="tool-calls"),
            SUCCESS,
        ]
    )
    result, _ = run(FakeSandbox(stdout=stdout))
    assert result.finished is True
    assert len(result.output["events"]) == 5


@pytest.mark.cpu
@pytest.mark.level0
def test_redaction_before_tail_truncation_and_in_transcript(caplog):
    stdout = "\n".join(
        [
            event(
                "text",
                text="test-secret-only Bearer unknown-secret",
                apiKey="another-secret",
                **{"test-secret-only": "value"},
            ),
            event("step_finish", reason="stop"),
        ]
    )
    result, _ = run(FakeSandbox(stdout=stdout), messages=[{"role": "user", "content": "test-secret-only"}])
    serialized = json.dumps(dataclasses.asdict(result)) + caplog.text
    assert all(secret not in serialized for secret in ("test-secret-only", "unknown-secret", "another-secret"))


@pytest.mark.cpu
@pytest.mark.level0
def test_cancellation_propagates_after_private_state_cleanup():
    sandbox = FakeSandbox(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(sandbox)
    assert sandbox.commands[-1][0][:3] == ["rm", "-rf", "--"]


@pytest.mark.cpu
@pytest.mark.level0
def test_each_invocation_has_private_state():
    _, first = run()
    _, second = run()
    assert first.commands[0][0][-1] != second.commands[0][0][-1]


@pytest.mark.cpu
@pytest.mark.level0
def test_escaped_credential_is_redacted_before_diagnostic_truncation():
    cfg = config()
    cfg.model.api_key = 'test-"quoted\\secret'
    stdout = event("text", text=cfg.model.api_key + "x" * 5000) + "\n" + event("step_finish", reason="stop")
    result, _ = run(FakeSandbox(stdout=stdout), cfg=cfg)
    serialized = json.dumps(dataclasses.asdict(result))
    assert "quoted" not in serialized
    assert result.finished


@pytest.mark.cpu
@pytest.mark.level0
def test_stderr_redaction_precedes_tail_truncation():
    cfg = config()
    cfg.model.api_key = "TEST-KEY-CANARY-SUFFIX"
    stderr = cfg.model.api_key + "x" * 3990
    result, _ = run(FakeSandbox(stderr=stderr), cfg=cfg)
    assert result.info["stderr_tail"] == "<redacted>" + "x" * 3990
    result, _ = run(FakeSandbox(stderr='{"apiKey":"other-key","message":"Bearer another-key"}'))
    assert "other-key" not in result.info["stderr_tail"]
    assert "another-key" not in result.info["stderr_tail"]
