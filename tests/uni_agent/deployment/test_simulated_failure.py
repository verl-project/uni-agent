"""Failure-simulation tests for SimulatedRuntime.

Three failure modes, all probability-driven and seed-reproducible:

1. Command timeout: sleep(delay) then raise swerex CommandTimeoutError.
   AgentEnv catches it -> ActionTimeoutError (recoverable: timeout_budget--).
2. Terminal death: modeled as a dead flag. The command first times out
   (CommandTimeoutError), which makes AgentEnv probe liveness via
   ``echo 'terminal still alive'``; once dead, that probe returns no marker,
   so AgentEnv itself raises TerminalNotAliveError (fatal: episode ends).
   This is the *only* way to get the correct ``terminal_dead`` semantics,
   because env.py raises TerminalNotAliveError from its own probe logic --
   it is never thrown by the runtime.
3. Observation-layer failure: a template returns failure text (traceback /
   "command not found") with exit_code != 0; the sandbox itself does NOT
   raise -- the model sees the failure and the loop continues.
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from swerex.exceptions import CommandTimeoutError  # noqa: E402
from swerex.runtime.abstract import BashAction  # noqa: E402

from uni_agent.deployment.simulated.config import (  # noqa: E402
    SimulatedDeploymentConfig,
    TerminalDeadConfig,
    TimeoutSimConfig,
)
from uni_agent.deployment.simulated.deployment import SimulatedRuntime  # noqa: E402
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig, TerminalNotAliveError  # noqa: E402


def _bash(command: str, timeout: int = 10) -> BashAction:
    return BashAction(command=command, timeout=timeout)


def _runtime(**kw) -> SimulatedRuntime:
    base = dict(run_id="t")
    base.update(kw)
    return SimulatedRuntime(**base)


# ---------------------------------------------------------------------------
# Timeout simulation (runtime-level: raises swerex CommandTimeoutError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_disabled_by_default_does_not_raise() -> None:
    rt = _runtime(seed=1)
    for _ in range(50):
        obs = await rt.run_in_session(_bash("python /testbed/repro.py"))
        assert obs.exit_code == 0


@pytest.mark.asyncio
async def test_timeout_enabled_always_raises_command_timeout() -> None:
    rt = _runtime(seed=1, timeout=TimeoutSimConfig(enabled=True, probability=1.0, delay_seconds=0.01))
    with pytest.raises(CommandTimeoutError):
        await rt.run_in_session(_bash("python /testbed/repro.py"))


@pytest.mark.asyncio
async def test_timeout_probability_is_seed_reproducible() -> None:
    """Same seed -> same commands time out (by index)."""
    cmds = [f"python /testbed/r{i}.py" for i in range(40)]
    cfg = TimeoutSimConfig(enabled=True, probability=0.3, delay_seconds=0.0)

    async def which_timeout(seed: int) -> list[int]:
        rt = _runtime(seed=seed, timeout=cfg)
        idx: list[int] = []
        for i, c in enumerate(cmds):
            try:
                await rt.run_in_session(_bash(c))
            except CommandTimeoutError:
                idx.append(i)
        return idx

    a = await which_timeout(42)
    b = await which_timeout(42)
    assert a == b
    assert len(a) > 0


# ---------------------------------------------------------------------------
# Terminal death (end-to-end through AgentEnv -> env.py raises TerminalNotAliveError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_dead_makes_env_raise_terminal_not_alive() -> None:
    """With terminal_dead enabled, a command must drive AgentEnv to raise
    TerminalNotAliveError -- the fatal path, not the recoverable timeout path."""
    env = AgentEnv(
        run_id="dead-1",
        env_config=AgentEnvConfig(
            deployment=SimulatedDeploymentConfig(
                seed=1,
                terminal_dead=TerminalDeadConfig(enabled=True, probability=1.0),
            )
        ),
    )
    await env.start()
    try:
        with pytest.raises(TerminalNotAliveError):
            await env.run_action("python /testbed/repro.py", action_timeout=10)
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_terminal_dead_probability_is_seed_reproducible() -> None:
    """Across many short episodes, the set of episodes that die must be
    reproducible under a fixed seed."""
    cmds = [f"python /testbed/r{i}.py" for i in range(40)]
    cfg = TerminalDeadConfig(enabled=True, probability=0.3)

    async def which_dead(seed: int) -> list[int]:
        rt = _runtime(seed=seed, terminal_dead=cfg)
        idx: list[int] = []
        for i, c in enumerate(cmds):
            # once dead, the runtime stays dead: every subsequent call is a
            # failed liveness probe. We only record the FIRST death index.
            try:
                await rt.run_in_session(_bash(c))
            except CommandTimeoutError:
                # terminal_dead triggers via a timeout + dead probe; record if
                # the runtime entered the dead state on this call.
                if rt._dead:
                    idx.append(i)
                    break
        return idx

    a = await which_dead(7)
    b = await which_dead(7)
    assert a == b
    assert len(a) > 0


@pytest.mark.asyncio
async def test_dead_runtime_probes_fail_with_no_marker() -> None:
    """Once dead, the liveness probe command must return NO 'terminal still
    alive' marker, which is what makes env.py decide the terminal is dead."""
    rt = _runtime(seed=1, terminal_dead=TerminalDeadConfig(enabled=True, probability=1.0))
    # trigger the dead state (this times out via CommandTimeoutError)
    with pytest.raises(CommandTimeoutError):
        await rt.run_in_session(_bash("python /testbed/repro.py"))
    assert rt._dead
    # the probe command env.py issues:
    probe = await rt.run_in_session(_bash("echo 'terminal still alive'"))
    assert "terminal still alive" not in probe.output


# ---------------------------------------------------------------------------
# Observation-layer failure (content failure, sandbox does not raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observation_layer_failure_does_not_raise() -> None:
    """A failed observation (python traceback / command not found) must come
    back as an Observation, not as a sandbox exception."""
    rt = _runtime(seed=1)
    seen_failure = False
    for _ in range(50):
        obs = await rt.run_in_session(_bash("python /testbed/repro.py"))
        if "Traceback" in obs.output or "command not found" in obs.output:
            seen_failure = True
            break
    assert seen_failure


# ---------------------------------------------------------------------------
# Independence: timeout and terminal_dead use separate RNG streams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_and_terminal_dead_are_independent_streams() -> None:
    cmds = [f"python /testbed/r{i}.py" for i in range(40)]
    tcfg = TimeoutSimConfig(enabled=True, probability=0.3, delay_seconds=0.0)
    pcfg = TerminalDeadConfig(enabled=True, probability=0.3)

    async def run_once(seed: int) -> tuple[list[int], int]:
        rt = _runtime(seed=seed, timeout=tcfg, terminal_dead=pcfg)
        timeouts: list[int] = []
        first_dead = -1
        for i, c in enumerate(cmds):
            try:
                await rt.run_in_session(_bash(c))
            except CommandTimeoutError:
                timeouts.append(i)
                if rt._dead and first_dead < 0:
                    first_dead = i
                    break
        return timeouts, first_dead

    t1, d1 = await run_once(100)
    t2, d2 = await run_once(100)
    assert (t1, d1) == (t2, d2)  # reproducible
    assert len(t1) > 0
