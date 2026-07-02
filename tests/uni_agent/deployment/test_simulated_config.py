"""Config + discriminated-union registration tests for the simulated deployment.

Verifies that ``type: simulated`` routes through the ``DeployConfig`` discriminated
union to ``SimulatedDeploymentConfig`` / ``SimulatedDeployment``, the same way every
other deployment (host/local/modal/...) does. This is what makes the simulated
selectable from ``agent_config`` with zero changes above ``AgentEnv``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from pydantic import BaseModel, Field  # noqa: E402

from uni_agent.deployment import DeployConfig, SimulatedDeployment, SimulatedDeploymentConfig  # noqa: E402


class _EnvLikeConfig(BaseModel):
    """Mirror of AgentEnvConfig's deployment field: the discriminated union
    only takes effect when it's a *field* of a pydantic model, not as a bare
    Annotated alias."""

    deployment: DeployConfig = Field(default_factory=SimulatedDeploymentConfig)


def test_simulated_config_defaults() -> None:
    """The discriminator and tunables have the agreed defaults."""
    cfg = SimulatedDeploymentConfig()
    assert cfg.type == "simulated"
    assert cfg.seed is None  # default = random
    assert cfg.observation_scale == 1.0


def test_simulated_config_tolerates_docker_only_overrides() -> None:
    """The swebench dataset carries per-sample docker-only fields (image,
    command, container_runtime) in tools_kwargs, which the agent loop deep-
    merges onto the deployment block. SimulatedDeploymentConfig must IGNORE these
    (not reject) so the same dataset drives both local and simulated deployments.

    Regression for the ValidationError that killed the first simulated perf run:
    ``deployment.simulated.image: Extra inputs are not permitted``.
    """
    cfg = SimulatedDeploymentConfig.model_validate(
        {
            "type": "simulated",
            "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033",
            "command": "python3 -m swerex.server --auth-token {token}",
            "container_runtime": "docker",
            "extra_run_args": ["-v", "/wheels:/wheels:ro"],
        }
    )
    assert cfg.type == "simulated"
    assert cfg.seed is None  # docker-only keys silently dropped


def test_simulated_config_accepts_seed_and_scale() -> None:
    cfg = SimulatedDeploymentConfig(seed=42, observation_scale=2.0)
    assert cfg.seed == 42
    assert cfg.observation_scale == 2.0


def test_discriminated_union_routes_type_simulated() -> None:
    """A plain dict with ``type: simulated`` must parse via the union into the
    SimulatedDeploymentConfig subclass -- this is the wire format coming from YAML
    once it lands in an AgentEnvConfig.deployment field."""
    cfg = _EnvLikeConfig.model_validate({"deployment": {"type": "simulated"}})
    assert isinstance(cfg.deployment, SimulatedDeploymentConfig)


def test_get_deployment_returns_simulated() -> None:
    """The factory hands back a SimulatedDeployment carrying the seeded SimulatedRuntime."""
    cfg = SimulatedDeploymentConfig(seed=7, observation_scale=1.5)
    dep = cfg.get_deployment(run_id="run-1")
    assert isinstance(dep, SimulatedDeployment)
    assert dep.run_id == "run-1"


@pytest.mark.asyncio
async def test_simulated_deployment_exposes_seeded_runtime() -> None:
    """SimulatedDeployment must construct its SimulatedRuntime with seed/scale so the
    route/render behavior is configured before start()."""
    cfg = SimulatedDeploymentConfig(seed=7, observation_scale=1.5)
    dep = cfg.get_deployment(run_id="run-1")
    await dep.start()
    try:
        rt = dep.runtime
        assert rt._seed == 7
        assert rt.observation_scale == 1.5
    finally:
        await dep.stop()
