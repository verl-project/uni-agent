"""Host adapter tests without GPU/runtime dependencies."""
import asyncio
import json
import unittest
from types import SimpleNamespace

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.openclaw.agent import OpenClawAgent, OpenClawConfig, parse_openclaw_result, _redact, _diagnostic


class FakeSandbox:
    def __init__(self, mode="success"):
        self.mode = mode
        self.shells, self.commands, self.files = [], [], {}

    async def exec_shell(self, command, **kwargs):
        self.shells.append(command)
        if command.startswith("rm -f") and self.mode in {"cleanup_failure", "timeout_cleanup"}:
            raise OSError("cleanup unavailable")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write_file(self, path, content):
        self.files[path] = content

    async def exec(self, argv, **kwargs):
        self.commands.append((argv, kwargs))
        if argv[0] == "python3":
            return SimpleNamespace(exit_code=0, stdout=json.dumps({"verified": self.mode != "audit_failure"}), stderr="")
        if self.mode in {"timeout", "timeout_cleanup"}:
            raise TimeoutError("deadline")
        if self.mode == "startup":
            raise FileNotFoundError("openclaw")
        payload = {"payloads": [{"text": "DONE"}], "meta": {
            "aborted": False, "executionTrace": {"fallbackUsed": self.mode == "fallback"}}}
        return SimpleNamespace(exit_code=1 if self.mode == "exit_failure" else 0,
                               stdout="not JSON" if self.mode == "protocol" else json.dumps(payload), stderr="")


class AgentTests(unittest.TestCase):
    def run_agent(self, mode):
        sandbox = FakeSandbox(mode)
        cfg = OpenClawConfig(model=ModelConfig(base_url="http://endpoint/v1", model_name="policy"))
        result = asyncio.run(OpenClawAgent(cfg).run(sandbox=sandbox,
                            messages=[{"role": "user", "content": "solve 中文 ' $(false)"}], workdir="/workspace"))
        self.assertTrue(any(command.startswith("rm -f ") for command in sandbox.shells))
        return result, sandbox

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_success(self):
        result, sandbox = self.run_agent("success")
        self.assertTrue(result.finished)
        self.assertEqual(len(sandbox.commands), 2)
        launch_command = " ".join(sandbox.commands[0][0])
        self.assertIn("--message-file", launch_command)
        message_path = sandbox.commands[0][0][-1]
        message_path = message_path.split("--message-file", 1)[1].split()[0].strip("'\"")
        self.assertIn("solve 中文 ' $(false)", sandbox.files[message_path])
        self.assertNotIn("solve", launch_command)

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_audit_failure(self):
        self.assertFalse(self.run_agent("audit_failure")[0].finished)

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_fallback(self):
        self.assertFalse(self.run_agent("fallback")[0].finished)

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_protocol(self):
        result, _ = self.run_agent("protocol")
        self.assertFalse(result.finished)
        self.assertEqual(result.info["error_kind"], "protocol_error")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_exit_failure(self):
        result, _ = self.run_agent("exit_failure")
        self.assertFalse(result.finished)
        self.assertEqual(result.info["error_kind"], "agent_failure")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_timeout(self):
        result, _ = self.run_agent("timeout")
        self.assertFalse(result.finished)
        self.assertEqual(result.info["error_kind"], "timeout")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_startup(self):
        result, _ = self.run_agent("startup")
        self.assertFalse(result.finished)
        self.assertEqual(result.info["error_kind"], "startup_failure")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_cleanup_failure(self):
        result, _ = self.run_agent("cleanup_failure")
        self.assertFalse(result.finished)
        self.assertEqual(result.info["error_kind"], "cleanup_failure")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_timeout_cleanup_keeps_original_error(self):
        result, _ = self.run_agent("timeout_cleanup")
        self.assertEqual(result.info["error_kind"], "timeout")
        self.assertEqual(result.info["cleanup_error"], "OSError")

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_redacts_payload_and_bounded_diagnostic(self):
        value = {"apiKey": "hidden", "data": ["prefix fake-secret suffix", "Bearer abc123"]}
        rendered = json.dumps(_redact(value, "fake-secret"))
        for secret in ("hidden", "fake-secret", "abc123"):
            self.assertNotIn(secret, rendered)
        self.assertLessEqual(len(_diagnostic("x" * 9000)), 4000)

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_nested_parser(self):
        self.assertEqual(parse_openclaw_result('noise\n{"meta":{"a":1}}'), {"meta": {"a": 1}})

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_last_object(self):
        self.assertEqual(parse_openclaw_result('{"old":1}\n{"new":{"b":2}}'), {"new": {"b": 2}})


if __name__ == "__main__":
    unittest.main()
