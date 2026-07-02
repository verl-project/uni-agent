"""Routing tests for SimulatedRuntime: bash command string -> route key.

SimulatedRuntime is the CPU-only sandbox stub used for performance testing
(LLM real, sandbox simulated). The router decides which representative
observation template a given tool command maps to.
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from uni_agent.deployment.simulated.deployment import SimulatedRuntime  # noqa: E402


def _make_runtime():
    """Build a SimulatedRuntime without going through a real swerex session.

    The router is a pure function of the command string; it must be
    exercisable without start()/network/docker.
    """
    return SimulatedRuntime(run_id="test-route")


@pytest.mark.parametrize(
    "command, expected",
    [
        # submit/finish tool emits this fixed string
        ("echo '<<<Finished>>>'", "finish"),
        # str_replace_editor subcommands (CLI argv, --command <X>)
        ("str_replace_editor --command view --path /testbed", "editor:view"),
        ("str_replace_editor --command create --path /testbed/x.py --file_text a=1", "editor:create"),
        ("str_replace_editor --command str_replace --path /testbed/x.py", "editor:str_replace"),
        ("str_replace_editor --command insert --path /testbed/x.py", "editor:insert"),
        ("str_replace_editor --command undo_edit --path /testbed/x.py", "editor:undo_edit"),
        # install_tools phase commands must route to install (no-op success)
        ("which str_replace_editor", "install"),
        ("export PATH=/usr/local/bin:$PATH", "install"),
        ("chmod +x /usr/local/bin/execute_bash", "install"),
        ("mkdir -p /opt/uni-agent/skills", "install"),
        ("python -m pip install tree-sitter", "install"),
        # execute_bash sub-routing by first token
        ("python -m pytest tests/test_card.py", "test_output"),
        ("pytest -x tests/", "test_output"),
        ("python /testbed/reproduce_issue.py", "python_script"),
        ("python3 /testbed/repro.py", "python_script"),
        ("find /testbed -name *.py", "listing"),
        ("ls -la /testbed", "listing"),
        ("grep -rn Count /testbed/x.py", "search"),
        ("cat /testbed/x.py", "file_view"),
        ("head -50 /testbed/x.py", "file_view"),
        ("git diff", "default"),
        ("git log --oneline", "default"),
        ("cd /testbed && ls", "default"),
    ],
)
def test_route_maps_command_to_key(command: str, expected: str) -> None:
    rt = _make_runtime()
    assert rt._route(command) == expected, f"command={command!r}"
