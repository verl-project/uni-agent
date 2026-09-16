"""Pure protocol and status tests for the in-image Hermes runner."""

from __future__ import annotations

import pytest

from examples.blackbox_recipes.hermes.run_hermes import _recipe_config, map_hermes_result

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_recipe_config_disables_non_episode_state():
    config = _recipe_config(approval_mode="off")
    assert config["compression"] == {
        "enabled": False,
        "micro_compact": False,
        "proactive_prune_tokens": 0,
        "idle_compact_after_seconds": 0,
    }
    assert config["memory"]["memory_enabled"] is False
    assert config["delegation"]["orchestrator_enabled"] is False
    assert config["approvals"]["mode"] == "off"


def test_map_hermes_result_keeps_budget_and_errors_unfinished():
    budget = map_hermes_result({"completed": False, "final_response": "summary", "turn_exit_reason": "max_iterations"})
    assert budget["status"] == "budget_exhausted"
    assert budget["finished"] is False

    error = map_hermes_result(
        {"completed": True, "final_response": "partial", "failed": True, "turn_exit_reason": "text_response"}
    )
    assert error["status"] == "error"
    assert error["finished"] is False


def test_map_hermes_result_only_marks_unambiguous_text_complete():
    result = map_hermes_result({"completed": True, "final_response": "fixed", "turn_exit_reason": "text_response"})
    assert result["status"] == "completed"
    assert result["finished"] is True
