"""Run the contributor Agent template through the real local sandbox API."""

import asyncio
import sys
from pathlib import Path

import pytest

from examples.sidecar.agent_template import ToolAgent, ToolConfig
from uni_agent.agents import ModelConfig
from uni_agent.sandbox import LocalSandbox


@pytest.mark.parametrize("finished", [True, False, "true"])
def test_agent_template_preserves_model_config_and_requires_boolean_completion(tmp_path, finished):
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import json,sys; request=json.load(sys.stdin); "
        f"print(json.dumps({{'output':request, 'finished':{finished!r}}}))\n"
    )
    agent = ToolAgent(
        ToolConfig(
            command=[sys.executable, str(wrapper)],
            model=ModelConfig(base_url="http://gateway.example/sessions/sample/v1", model_name="my-model"),
        )
    )
    sandbox = LocalSandbox()
    created = []
    write_file = sandbox.write_file

    async def capture_write(path, content):
        created.append(Path(path))
        await write_file(path, content)

    sandbox.write_file = capture_write
    result = asyncio.run(
        agent.run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "sample"}],
            workdir=str(tmp_path),
        )
    )
    assert result.output["model"]["base_url"].endswith("/sessions/sample/v1")
    assert result.output["model"]["model_name"] == "my-model"
    assert result.finished is (finished is True)
    assert created and not created[0].parent.exists()


def test_agent_template_rejects_invalid_stdout(tmp_path):
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("print('unstructured CLI events')\n")
    result = asyncio.run(
        ToolAgent(ToolConfig(command=[sys.executable, str(wrapper)])).run(
            sandbox=LocalSandbox(),
            messages=[],
            workdir=str(tmp_path),
        )
    )
    assert not result.finished
    assert "error" in result.output
