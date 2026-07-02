"""Render + determinism tests for SimulatedRuntime.run_in_session.

Covers the core performance-test guarantees:
- returns an Observation (routed by the command)
- finish / install produce their fixed canned outputs
- with a seed, the same command sequence reproduces identical observations
- with a different seed (or None), sampling may differ
- observation_scale stretches the output length
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from swerex.runtime.abstract import BashAction, BashInterruptAction  # noqa: E402

from uni_agent.deployment.simulated.deployment import SimulatedRuntime  # noqa: E402


def _has_attrs(obj, *names) -> bool:
    """swerex's Observation is an Annotated union alias, not isinstance-able;
    assert on the duck-typed attributes the loop relies on instead."""
    return all(hasattr(obj, n) for n in names)


def _bash(command: str, timeout: int = 10) -> BashAction:
    return BashAction(command=command, timeout=timeout)


@pytest.mark.asyncio
async def test_finish_command_returns_terminal_signal() -> None:
    rt = SimulatedRuntime(run_id="t")
    obs = await rt.run_in_session(_bash("echo '<<<Finished>>>'"))
    assert _has_attrs(obs, "output", "exit_code")
    assert obs.exit_code == 0
    assert "<<<Finished>>>" in obs.output


@pytest.mark.asyncio
async def test_install_commands_succeed_with_empty_output() -> None:
    rt = SimulatedRuntime(run_id="t")
    for cmd in ("which str_replace_editor", "export PATH=/x:$PATH", "chmod +x /x", "mkdir -p /opt/s"):
        obs = await rt.run_in_session(_bash(cmd))
        assert obs.exit_code == 0, cmd
        assert obs.output == "", cmd


@pytest.mark.asyncio
async def test_interrupt_action_returns_nonzero_exit() -> None:
    rt = SimulatedRuntime(run_id="t")
    obs = await rt.run_in_session(BashInterruptAction(timeout=5))
    assert _has_attrs(obs, "output", "exit_code")
    assert obs.exit_code == 130


@pytest.mark.asyncio
async def test_non_install_command_produces_representative_output() -> None:
    """A real tool call (editor view) must return a non-empty canned
    observation, not empty like install."""
    rt = SimulatedRuntime(run_id="t")
    obs = await rt.run_in_session(_bash("str_replace_editor --command view --path /testbed/x.py"))
    assert obs.exit_code == 0
    assert len(obs.output) > 0


@pytest.mark.asyncio
async def test_same_seed_reproduces_identical_sequence() -> None:
    """Reproducibility: two runtimes with the same seed run the same command
    sequence -> identical outputs, byte for byte."""
    cmds = [
        "str_replace_editor --command view --path /testbed/a.py",
        "python /testbed/repro.py",
        "grep -rn foo /testbed",
        "str_replace_editor --command view --path /testbed/b.py",
    ]
    rt1 = SimulatedRuntime(run_id="t1", seed=42)
    rt2 = SimulatedRuntime(run_id="t2", seed=42)
    out1 = [await rt1.run_in_session(_bash(c)) for c in cmds]
    out2 = [await rt2.run_in_session(_bash(c)) for c in cmds]
    assert [o.output for o in out1] == [o.output for o in out2]


@pytest.mark.asyncio
async def test_different_seed_likely_differs() -> None:
    """With many distinct templates, two different seeds should not produce
    an identical full sequence (probabilistic, but robust with enough pulls)."""
    cmd = "python /testbed/repro.py"
    pulls = 30
    a = [await SimulatedRuntime(run_id="a", seed=1).run_in_session(_bash(cmd)) for _ in range(pulls)]
    b = [await SimulatedRuntime(run_id="b", seed=2).run_in_session(_bash(cmd)) for _ in range(pulls)]
    assert [o.output for o in a] != [o.output for o in b]


@pytest.mark.asyncio
async def test_observation_scale_stretches_length() -> None:
    """observation_scale multiplies the template length (>=1x grows it)."""
    rt_small = SimulatedRuntime(run_id="t", seed=7, observation_scale=1.0)
    rt_big = SimulatedRuntime(run_id="t", seed=7, observation_scale=2.0)
    cmd = "str_replace_editor --command view --path /testbed/x.py"
    a = await rt_small.run_in_session(_bash(cmd))
    b = await rt_big.run_in_session(_bash(cmd))
    assert len(b.output) > len(a.output)
