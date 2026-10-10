import pytest

from uni_agent.gateway.agent_hint import AgentRuntimeHint
from uni_agent.gateway.kv_offload.hints import compute_dynamic_priority

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _inputs(**overrides):
    values = {
        "tools_available": False,
        "has_active_chain": False,
        "received_tool_result": False,
        "context_tokens": 0,
        "trajectory_capacity": 65_536,
        "remaining_capacity": 65_536,
        "rollback_applied": False,
        "active_chain_count": 0,
    }
    values.update(overrides)
    return AgentRuntimeHint(**values)


def test_new_tool_request_scores_as_normal():
    decision = compute_dynamic_priority(_inputs(tools_available=True))

    assert decision.priority == 35
    assert decision.continuation_score == 25
    assert decision.remaining_capacity_score == 10
    assert decision.band == "normal"


def test_long_tool_result_continuation_scores_hot():
    decision = compute_dynamic_priority(
        _inputs(
            tools_available=True,
            has_active_chain=True,
            received_tool_result=True,
            context_tokens=40_000,
            remaining_capacity=20_000,
            rollback_applied=True,
        )
    )

    assert decision.priority == 92
    assert decision.continuation_score == 55
    assert decision.recompute_cost_score == 22
    assert decision.remaining_capacity_score == 10
    assert decision.chain_state_score == 5
    assert decision.band == "hot"


def test_nearly_exhausted_branched_chain_is_penalized():
    decision = compute_dynamic_priority(
        _inputs(
            has_active_chain=True,
            context_tokens=64_000,
            remaining_capacity=900,
            active_chain_count=2,
        )
    )

    assert decision.priority == 42
    assert decision.remaining_capacity_score == -15
    assert decision.chain_state_score == -5


def test_runtime_hint_roundtrips_without_policy_results():
    runtime = _inputs(trajectory_id="trajectory")
    raw = runtime.to_dict()
    assert AgentRuntimeHint.from_dict(raw) == runtime
    assert raw["schema_version"] == 2
    assert "kv_priority" not in raw and "lease_until" not in raw


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("tools_available", 1),
        ("has_active_chain", "false"),
        ("received_tool_result", None),
        ("rollback_applied", 0),
        ("context_tokens", -1),
        ("context_tokens", True),
        ("context_tokens", 1.5),
        ("active_chain_count", -1),
        ("trajectory_capacity", 0),
        ("remaining_capacity", -1),
        ("remaining_capacity", float("nan")),
        ("remaining_capacity", 70_000),
        ("trajectory_id", None),
    ],
)
def test_runtime_hint_rejects_malformed_fields(field, value):
    raw = _inputs().to_dict()
    raw[field] = value
    assert AgentRuntimeHint.from_dict(raw) is None


@pytest.mark.parametrize("field", list(AgentRuntimeHint.__dataclass_fields__) + ["schema_version"])
def test_runtime_hint_rejects_incomplete_snapshot(field):
    raw = _inputs().to_dict()
    del raw[field]
    assert AgentRuntimeHint.from_dict(raw) is None


def test_runtime_hint_accepts_unknown_capacity():
    raw = _inputs(trajectory_capacity=None, remaining_capacity=None).to_dict()
    assert AgentRuntimeHint.from_dict(raw) is not None
    raw["remaining_capacity"] = 10
    assert AgentRuntimeHint.from_dict(raw) is None
