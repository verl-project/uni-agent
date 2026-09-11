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


@pytest.mark.parametrize("via_agent", [False, True])
def test_run_agent_preserves_stdin_arguments_and_inherited_home(tmp_path, via_agent):
    tool_root = tmp_path / "tool"
    tool_bin = tool_root / "bin"
    tool_bin.mkdir(parents=True)
    run_agent = tool_bin / "run_agent.sh"
    shutil.copy2(RUN_AGENT, run_agent)
    vendor_bin = tool_root / "vendor" / "x86_64-unknown-linux-musl"
    vendor_bin.mkdir(parents=True)
    fake_codex = vendor_bin / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(), "
        "'managed_package_root': __import__('os').environ.get('CODEX_MANAGED_PACKAGE_ROOT')}))\n"
    )
    fake_codex.chmod(0o755)

    marker = tmp_path / "command-substitution-ran"
    extra_args = ["--config", "value with spaces", "quote'arg", f"$(touch {marker})"]
    env = os.environ.copy()
    home = tmp_path / "codex-home"
    env.update(
        CODEX_HOME=str(home), CODEX_MODEL="policy", CODEX_API_BASE="http://gateway/v1", CODEX_PROJECT_DIR=str(tmp_path)
    )
    argv = ["bash", str(run_agent), *extra_args]
    if via_agent:
        command = build_agent_command(
            task_b64="Zml4IHRoaXM=",
            tool_script=str(run_agent),
            gateway_url="http://gateway/v1",
            model_name="policy",
            api_key="key",
            project_dir=str(tmp_path),
        )
        argv = ["bash", "-c", command]
    result = subprocess.run(
        argv,
        input="fix this",
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    argv = payload["argv"]
    assert payload["stdin"] == "fix this"
    assert payload["managed_package_root"] == str(tool_root)
    if not via_agent:
        assert argv[-len(extra_args) :] == extra_args
    assert argv[:2] == ["exec", "--json"]
    assert not marker.exists()

    assert (home / "config.toml").is_file()


def test_run_agent_propagates_codex_exit_code(tmp_path):
    tool_root = tmp_path / "tool"
    tool_bin = tool_root / "bin"
    tool_bin.mkdir(parents=True)
    run_agent = tool_bin / "run_agent.sh"
    shutil.copy2(RUN_AGENT, run_agent)
    vendor_bin = tool_root / "vendor" / "x86_64-unknown-linux-musl"
    vendor_bin.mkdir(parents=True)
    fake_codex = vendor_bin / "codex"
    fake_codex.write_text("#!/usr/bin/env python3\nraise SystemExit(23)\n")
    fake_codex.chmod(0o755)

    env = os.environ.copy()
    env.update(CODEX_HOME=str(tmp_path / "codex-home"), CODEX_MODEL="policy", CODEX_API_BASE="http://gateway/v1")
    result = subprocess.run(
        ["bash", str(run_agent)],
        input="fix this",
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 23


@pytest.mark.parametrize("inherited", [None, "", "/tmp/custom-codex-state"])
def test_run_agent_home_default_expression(inherited):
    # Evaluate the actual assignment without creating shared /tmp state.
    assignment = next(line for line in RUN_AGENT.read_text().splitlines() if line.startswith("CODEX_HOME="))
    env = os.environ.copy()
    env.pop("CODEX_HOME", None)
    if inherited is not None:
        env["CODEX_HOME"] = inherited
    result = subprocess.run(
        ["bash", "-c", assignment + '\nprintf "%s" "$CODEX_HOME"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == (inherited or "/tmp/codex-home")
