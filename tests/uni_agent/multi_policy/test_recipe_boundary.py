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
def test_recipe_uses_custom_task_runner_not_standard_manager_class():
    """The adapter needs per-policy clients, so the recipe task runner builds it.

    verl's standard TaskRunner supplies a single llm_client/teacher_client and
    cannot satisfy MultiPolicyRolloutAdapter.create. Keep the recipe config from
    advertising an extension point that it does not (and cannot) use.
    """
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "examples/multi_policy/config/multi_policy.yaml")
    agent_cfg = config.actor_rollout_ref.rollout.get("agent")
    assert agent_cfg is None or "agent_loop_manager_class" not in agent_cfg

    main_path = root / "examples" / "multi_policy" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "MultiPolicyTaskRunner" in class_names
    adapter_create_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "MultiPolicyRolloutAdapter"
    ]
    assert adapter_create_calls
