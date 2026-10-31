"""Keep the core independent of the recipe's concrete MAS types."""

import ast
from pathlib import Path

import pytest


@pytest.mark.cpu
@pytest.mark.level0
def test_core_does_not_import_multi_agent_recipe():
    root = Path(__file__).resolve().parents[3]
    for path in (root / "uni_agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert not any(
                module == "uni_agent.multi_agent"
                or module.startswith("uni_agent.multi_agent.")
                or module.startswith("examples.multi_policy")
                for module in modules
            ), str(path)


@pytest.mark.cpu
@pytest.mark.level0
def test_recipe_config_points_to_local_adapter():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "examples/multi_policy/config/multi_policy.yaml")
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class == (
        "examples.multi_policy.framework.multi_agent_entry.MultiPolicyRolloutAdapter"
    )
