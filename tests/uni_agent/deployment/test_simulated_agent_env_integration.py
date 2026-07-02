"""Integration: drive a real AgentEnv end-to-end with a simulated deployment.

This is the proof that ``type: simulated`` plugs into the production path with
zero changes above AgentEnv. The env's start -> install_tools -> run_action
-> close lifecycle must complete without docker/swe-rex, the install-phase
commands (which/chmod/export/mkdir) must all succeed, and a real tool call
(str_replace_editor view) must come back as a representative observation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from uni_agent.deployment import SimulatedDeploymentConfig  # noqa: E402
from uni_agent.interaction.env import AgentEnv, AgentEnvConfig  # noqa: E402


@pytest.mark.asyncio
async def test_agent_env_lifecycle_with_simulated_deployment() -> None:
    """Full AgentEnv lifecycle on a simulated sandbox installs tools and returns a
    representative observation for a real tool command."""
    env_config = AgentEnvConfig(
        deployment=SimulatedDeploymentConfig(seed=1, observation_scale=1.0),
        env_variables={"FOO": "bar"},
    )
    env = AgentEnv(run_id="it-1", env_config=env_config)

    await env.start()  # must not touch docker/swe-rex
    try:
        # install-phase commands the AgentEnv would issue (PATH/which/chmod)
        # must all succeed (exit 0) -- the simulated routes them to install.
        out = await env.communicate("which str_replace_editor", check="raise")
        assert isinstance(out, str)
        out = await env.communicate("export PATH=/usr/local/bin:$PATH", check="raise")

        # A real tool call routed to editor:view returns a representative
        # (non-empty) observation, not an install-style empty one.
        obs = await env.run_action(
            "str_replace_editor --command view --path /testbed/x.py",
            action_timeout=10,
        )
        assert "Observation:" in obs
        assert len(obs) > len("Observation:")
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_agent_env_finish_action_via_simulated() -> None:
    """The submit tool's fixed ``echo '<<<Finished>>>'`` command surfaces the
    terminal signal through the real run_action wrapper."""
    env = AgentEnv(
        run_id="it-2",
        env_config=AgentEnvConfig(deployment=SimulatedDeploymentConfig()),
    )
    await env.start()
    try:
        obs = await env.run_action("echo '<<<Finished>>>'", action_timeout=10)
        assert "<<<Finished>>>" in obs
    finally:
        await env.close()
