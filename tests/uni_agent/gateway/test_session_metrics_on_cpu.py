"""Gateway session-local metrics: request counts and phase timings."""

from __future__ import annotations

import pytest

from tests.uni_agent.gateway.test_session_multiple_chains_on_cpu import (
    _prompt_length,
    _run,
    _session,
)
from tests.uni_agent.support import SequencedBackend


@pytest.mark.asyncio
async def test_gateway_session_records_request_metrics_on_last_trajectory():
    session = _session("metrics-session")
    backend = SequencedBackend(["ONE", "TWO"])

    await _run(session, backend, [{"role": "user", "content": "first"}])
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ONE"},
            {"role": "user", "content": "second"},
        ],
    )
    trajectories = await session.finalize()

    assert len(trajectories) == 1
    metrics = trajectories[0].extra_fields["agent_metrics"]
    assert metrics["gateway.requests"]["aggregation"] == "sum"
    assert metrics["gateway.requests"]["values"] == [1.0, 1.0]
    for name in (
        "gateway.request_s",
        "gateway.encode_s",
        "gateway.backend_generate_s",
        "gateway.decode_s",
    ):
        assert metrics[name]["aggregation"] == "sum"
        assert len(metrics[name]["values"]) == 2
        assert all(value >= 0.0 for value in metrics[name]["values"])


@pytest.mark.asyncio
async def test_gateway_session_puts_metrics_only_on_last_of_multiple_chains():
    first_messages = [{"role": "user", "content": "fill the response budget"}]
    session = _session(
        "metrics-multi-chain",
        prompt_length=_prompt_length(first_messages),
        response_length=len("FULL"),
    )
    backend = SequencedBackend(["FULL", "NEW"])

    await _run(session, backend, first_messages)
    await _run(session, backend, [*first_messages, {"role": "assistant", "content": "FULL"}])
    await _run(session, backend, [{"role": "user", "content": "start a fresh chain"}])
    trajectories = await session.finalize()

    assert len(trajectories) == 2
    assert "agent_metrics" not in trajectories[0].extra_fields
    metrics = trajectories[-1].extra_fields["agent_metrics"]
    assert metrics["gateway.requests"]["aggregation"] == "sum"
    assert sum(metrics["gateway.requests"]["values"]) >= 2.0
