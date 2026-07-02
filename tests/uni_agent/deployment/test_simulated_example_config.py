"""Validate the shipped agent_config_simulated.yaml example.

A bad example config (typo, wrong field, missing type) is the kind of thing
that only blows up at perf-run time. This test loads the real YAML file and
asserts its deployment block parses into a SimulatedDeploymentConfig with sane
defaults, so the example can't silently rot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("swerex")

from uni_agent.deployment import SimulatedDeploymentConfig  # noqa: E402
from uni_agent.interaction.env import AgentEnvConfig  # noqa: E402


def _example_config_path() -> Path:
    """Locate scripts/agent_config_simulated.yaml relative to the repo root.

    Prefer the test file's own location; fall back to cwd (tests are always
    run from the repo root). __file__ alone is unreliable here because the
    uni_agent package is a namespace package (no __init__.py), which makes
    pytest surface a cwd-relative module path that .resolve() mishandles.
    """
    candidates = [
        Path(__file__).resolve().parents[3],
        Path.cwd(),
    ]
    for root in candidates:
        p = root / "scripts" / "agent_config_simulated.yaml"
        if p.is_file():
            return p
    raise FileNotFoundError("agent_config_simulated.yaml not found under repo root")


def test_example_config_parses_as_simulated() -> None:
    raw = yaml.safe_load(_example_config_path().read_text())
    # the agent config is a one-element list whose first item has `env.deployment`
    deployment_dict = raw[0]["env"]["deployment"]
    env_cfg = AgentEnvConfig.model_validate({"deployment": deployment_dict})
    assert isinstance(env_cfg.deployment, SimulatedDeploymentConfig)
    assert env_cfg.deployment.type == "simulated"


def test_example_config_disabled_failure_sims_by_default() -> None:
    """The shipped example must NOT enable timeout/terminal_dead by default --
    a perf baseline run should measure the LLM serving layer, not the
    failure-slow paths (which are opt-in)."""
    raw = yaml.safe_load(_example_config_path().read_text())
    dep = raw[0]["env"]["deployment"]
    assert dep["timeout"]["enabled"] is False
    assert dep["terminal_dead"]["enabled"] is False
    assert dep["observation_scale"] == 1.0
