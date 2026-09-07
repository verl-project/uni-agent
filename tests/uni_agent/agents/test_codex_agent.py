from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.codex.agent import CodexAgent, CodexConfig, build_agent_command, parse_agent_result
from uni_agent.gateway.adapters.responses import (
    responses_build_response,
    responses_to_internal,
)
from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.sandbox.base import ExecResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class FakeSandbox:
    def __init__(self, stdout: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[dict] = []

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.calls.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self.exit_code, stdout=self.stdout, stderr="")


def make_agent(**kwargs):
    kwargs.setdefault("tool_script", "/opt/codex/bin/run_agent.sh")
    kwargs.setdefault("model", ModelConfig(base_url="http://gateway/v1", model_name="policy"))
    return CodexAgent(CodexConfig(**kwargs))


def test_build_agent_command_uses_stdin_and_isolates_env():
    task = base64.b64encode(b"fix 'this'").decode()
    command = build_agent_command(
        task_b64=task,
        tool_script="/opt/codex/bin/run_agent.sh",
        gateway_url="http://127.0.0.1:38197/sessions/s1/v1",
        model_name="policy",
        api_key="key",
        project_dir="/testbed",
    )
    assert "| base64 -d |" in command
    assert "CODEX_API_BASE=http://127.0.0.1:38197/sessions/s1/v1" in command
    assert "CODEX_MODEL=policy" in command
    assert "CODEX_HOME=/tmp/codex-home" in command
    assert "fix 'this'" not in command


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


def test_parse_agent_result_timeout_and_failure():
    timeout = parse_agent_result("", -1)
    assert timeout["exit_status"] == "timeout"
    failed = parse_agent_result(json.dumps({"type": "turn.failed", "error": {"message": "bad"}}), 1)
    assert failed["exit_status"] == "error"
    assert failed["error"] == "bad"


def test_parse_agent_result_honors_wrapper_process_exit_event():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}),
            json.dumps({"type": "process.completed", "exit_code": 17}),
        ]
    )
    failed = parse_agent_result(stdout, 0)
    assert failed["exit_status"] == "error"
    assert failed["ok"] is False
    assert failed["error"] == "codex exited with code 17"


def test_codex_agent_runs_and_returns_agent_result():
    sandbox = FakeSandbox(
        stdout=json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
    )
    agent = make_agent()
    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "fix bug"}]))
    assert isinstance(result, AgentResult)
    assert result.finished is True
    assert result.output["content"] == "done"
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["workdir"] == "/testbed"


def test_responses_input_lowering():
    internal = responses_to_internal(
        {
            "instructions": "system rule",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]},
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
            "tools": [
                {"type": "function", "name": "exec_command", "description": "run", "parameters": {"type": "object"}},
                {"type": "web_search", "external_web_access": True},
            ],
            "max_output_tokens": 64,
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset({"max_tokens"}),
    )
    assert internal["messages"][0] == {"role": "system", "content": "system rule"}
    assert internal["messages"][1]["content"] == "fix"
    assert internal["messages"][2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert internal["messages"][3]["role"] == "tool"
    assert internal["sampling_params"]["max_tokens"] == 64
    assert internal["tools"] == [
        {
            "type": "function",
            "function": {"name": "exec_command", "description": "run", "parameters": {"type": "object"}},
        }
    ]


def test_responses_continuation_coalesces_reasoning_and_function_call():
    internal = responses_to_internal(
        {
            "instructions": "system rule",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "inspect first"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"find /testbed"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "file.py"},
            ],
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset(),
    )

    continuation = internal["messages"][2:]
    assert [message["role"] for message in continuation] == ["assistant", "tool"]
    assert continuation[0]["reasoning_content"] == "inspect first"
    assert continuation[0]["tool_calls"][0]["function"]["name"] == "exec_command"
    MessageCodec(FakeTokenizer()).encode_incremental(continuation)


def test_incremental_codec_coalesces_responses_assistant_fragments():
    codec = MessageCodec(FakeTokenizer())
    codec.encode_incremental(
        [
            {"role": "assistant", "content": "", "reasoning_content": "thinking"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
    )


def test_responses_canonicalizes_developer_and_late_system_messages():
    internal = responses_to_internal(
        {
            "instructions": "base system",
            "input": [
                {"type": "message", "role": "user", "content": "inspect"},
                {"type": "message", "role": "developer", "content": "late system"},
            ],
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset(),
    )
    assert [message["role"] for message in internal["messages"]] == ["system", "user"]
    assert internal["messages"][0]["content"] == "base system\nlate system"


def test_responses_build_response_for_text_and_tool_call():
    text = responses_build_response(
        SimpleNamespace(
            assistant_msg={"role": "assistant", "content": "fixed"},
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        model="policy",
    )
    assert text["object"] == "response"
    assert text["output_text"] == "fixed"
    assert text["output"][0]["type"] == "message"

    tool = responses_build_response(
        SimpleNamespace(
            assistant_msg={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": "{}"}}
                ],
            },
            finish_reason="tool_calls",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        model="policy",
    )
    assert tool["output"][0]["type"] == "function_call"
    assert tool["output"][0]["call_id"] == "call_1"
