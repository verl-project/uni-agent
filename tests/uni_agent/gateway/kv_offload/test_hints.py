from uni_agent.gateway.kv_offload.hints import DynamicPriorityInputs, compute_dynamic_priority


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
    return DynamicPriorityInputs(**values)


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
