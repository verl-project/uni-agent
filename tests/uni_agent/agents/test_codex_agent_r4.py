from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from uni_agent.agents.codex.agent import build_agent_command

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

ROOT = Path(__file__).parents[3]
RUN_AGENT = ROOT / "examples" / "codex" / "run_agent.sh"


def test_build_command_and_wrapper_preserve_extra_args(tmp_path):
    tool_root = tmp_path / "tool"
    tool_bin = tool_root / "bin"
    tool_bin.mkdir(parents=True)
    wrapper = tool_bin / "run_agent.sh"
    shutil.copy2(RUN_AGENT, wrapper)
    fake_codex = tool_bin / "codex"
    fake_codex.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake_codex.chmod(0o755)

    marker = tmp_path / "command-substitution-ran"
    extra_args = ["--config", "value with spaces", "quote'arg", f"$(touch {marker})"]
    command = build_agent_command(
        task_b64="Zml4IHRoaXM=",
        tool_script=str(wrapper),
        gateway_url="http://127.0.0.1:38197/sessions/s1/v1",
        model_name="policy",
        api_key="key",
        project_dir=str(tmp_path),
        codex_home=str(tmp_path / "codex-home"),
        extra_args=extra_args,
    )
    env = os.environ.copy()
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.strip())
    assert argv[-len(extra_args) :] == extra_args
    assert argv[:2] == ["exec", "--json"]
    assert not marker.exists()
