"""Optional black-box OpenCode + real Gateway integration probe.

Set OPENCODE_BIN explicitly to an installed 1.19.3 CLI to opt in; the test does
not auto-detect a CLI from PATH. The backend is deterministic synthetic CPU
output; it does not run a model or use credentials.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.sandbox.local import LocalSandbox


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_opencode_cli_routes_tool_turns_through_gateway_and_finalizes_trajectory(tmp_path, monkeypatch):
    """Exercise OpenCode/Gateway with synthetic output when OPENCODE_BIN is explicit."""
    cli_setting = os.environ.get("OPENCODE_BIN")
    if not cli_setting:
        pytest.skip("set OPENCODE_BIN explicitly to enable the optional OpenCode CLI probe")
    cli = shutil.which(cli_setting)
    if cli is None and Path(cli_setting).is_file() and os.access(cli_setting, os.X_OK):
        cli = cli_setting
    if cli is None:
        pytest.fail(f"OPENCODE_BIN={cli_setting!r} is not an executable command or path")

    # Python 3.10 exposes NotRequired through typing_extensions; one Gateway
    # dependency imports it from typing as though it were standard there.
    if not hasattr(typing, "NotRequired"):
        from typing_extensions import NotRequired

        monkeypatch.setattr(typing, "NotRequired", NotRequired, raising=False)

    try:
        from tests.uni_agent.support import FakeTokenizer, SequencedBackend
        from uni_agent.gateway.config import GatewayActorConfig
        from uni_agent.gateway.gateway import _GatewayActor
        from uni_agent.gateway.session.codec import MessageCodec
    except ImportError as exc:
        pytest.skip(f"Gateway CPU dependencies are unavailable: {exc}")

    def parse_synthetic_tool_call(self, text, tools, parser_name):
        if not text.startswith("<tool_call>\n") or not text.endswith("\n</tool_call>"):
            return text, []
        call = json.loads(text[len("<tool_call>\n") : -len("\n</tool_call>")])
        return "", [SimpleNamespace(name=call["name"], arguments=call["arguments"])]

    # Keep the real Gateway actor/session/codec path while replacing only the
    # engine-specific vLLM parser with a deterministic synthetic tool parser.
    monkeypatch.setattr(MessageCodec, "_process_tool_calls_vllm", parse_synthetic_tool_call)

    workspace = tmp_path / "workspace with 'quote"
    workspace.mkdir()
    marker = workspace / "gateway-tool-result.txt"
    tool_call = {
        "name": "bash",
        "arguments": {
            "command": f"printf 'gateway-probe\\n' > {shlex.quote(str(marker))}",
            "description": "Write a harmless marker in the temporary workspace",
        },
    }
    tool_text = f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"
    backend = SequencedBackend([tool_text, "GATEWAY_PROBE_DONE"])
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            tool_parser_name="synthetic-hermes",
            rollout_backend="vllm",
        ),
        backend,
    )
    await actor.start()
    session_id = "opencode-cli-integration"
    try:
        session = await actor.create_session(session_id, metadata={"probe": "synthetic-opencode"})
        config = OpenCodeConfig(
            opencode_bin=cli,
            run_timeout=60,
            state_root=str(tmp_path / "opencode-state"),
            model=ModelConfig(
                base_url=session.base_url,
                model_name="synthetic-policy",
                api_key="synthetic-probe-key",
            ),
        )
        result = await OpenCodeAgent(config).run(
            sandbox=LocalSandbox(),
            messages=[{"role": "user", "content": "Write the probe marker, then finish."}],
            workdir=str(workspace),
        )
        assert result.finished is True, result.info
        assert result.output["text"] == "GATEWAY_PROBE_DONE"
        assert marker.read_text(encoding="utf-8") == "gateway-probe\n"
        assert len(backend.calls) == 2

        session_state = await actor.get_session_state(session_id)
        assert session_state["session_id"] == session_id
        trajectories = await actor.finalize_session(session_id)
        assert len(trajectories) == 1
        assert trajectories[0].finished is None
        assert trajectories[0].num_turns == 4
        assert any(mask == 1 for mask in trajectories[0].response_mask)
    finally:
        if session_id in actor._sessions:
            await actor.abort_session(session_id)
        await actor.shutdown()
