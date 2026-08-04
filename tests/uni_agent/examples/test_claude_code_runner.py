from types import SimpleNamespace

import pytest

from examples.blackbox_recipes.claude_code import claude_code_runner as runner_module
from uni_agent.framework import EpisodeResult
from uni_agent.gateway.session import SessionHandle


@pytest.mark.asyncio
async def test_claude_code_runner_returns_typed_episode_result(monkeypatch):
    class _Sandbox:
        def __init__(self):
            self.stopped = False

        async def exec_shell(self, command, timeout):
            return SimpleNamespace(exit_code=0, stdout="done", stderr="")

        async def stop(self):
            self.stopped = True

    sandbox = _Sandbox()

    async def fake_create_sandbox(**kwargs):
        return sandbox

    async def fake_evaluate(env, metadata, eval_timeout):
        return 1.0, {"resolved": True, "report": {"tests": "passed"}}

    monkeypatch.setattr(runner_module, "_create_claude_sandbox", fake_create_sandbox)
    monkeypatch.setattr(runner_module, "evaluate_in_env", fake_evaluate)
    monkeypatch.setattr(runner_module, "build_reward_context", lambda tools_kwargs: ({"case": "x"}, 30))
    monkeypatch.setattr(runner_module, "extract_image", lambda env_config: "sandbox-image")

    result = await runner_module.claude_code_runner(
        raw_prompt=[{"role": "user", "content": "fix it"}],
        session=SessionHandle(session_id="session", base_url="http://gateway/session/v1"),
        sample_index=0,
        tools_kwargs={"env": {}},
    )

    assert result == EpisodeResult(
        reward=1.0,
        metrics={"acc": 1.0},
        episode_finished=True,
        reward_context={
            "claude_code_exit_code": 0,
            "resolved": True,
            "report": {"tests": "passed"},
        },
    )
    assert sandbox.stopped is True
