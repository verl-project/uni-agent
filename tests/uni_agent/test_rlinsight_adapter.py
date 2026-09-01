from __future__ import annotations

from types import SimpleNamespace

import pytest

from uni_agent import rlinsight_adapter


@pytest.fixture(autouse=True)
def _reset_rlinsight_adapter_state():
    rlinsight_adapter._warned_compatibility_features.clear()
    rlinsight_adapter.RolloutTraceConfig.reset()
    yield
    rlinsight_adapter._warned_compatibility_features.clear()
    rlinsight_adapter.RolloutTraceConfig.reset()


def _capture_trace_span(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict] = []

    def trace_span(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(rlinsight_adapter.RLInsightLogger, "trace_span", staticmethod(trace_span))
    return captured


@pytest.mark.cpu
@pytest.mark.level0
def test_report_span_merges_identity_and_normalizes_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_trace_span(monkeypatch)
    token = rlinsight_adapter._set_trace_identity({"uid": "u1", "sample": "7"})
    try:
        rlinsight_adapter._report_span(
            "test_span",
            start_time_ns=10,
            attributes={"payload": {"turn": 1}},
        )
    finally:
        rlinsight_adapter._reset_trace_identity(token)

    attributes = captured[0]["attributes"]
    assert attributes["uid"] == "u1"
    assert attributes["sample"] == "7"
    assert attributes["payload"] == '{"turn": 1}'


@pytest.mark.cpu
@pytest.mark.level0
def test_task_span_reports_identity_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_trace_span(monkeypatch)
    tools_kwargs = {
        "_trace_identity": {"uid": "u1", "session_id": "s1"},
        "task": {"sandbox": {"image": "image:v1"}},
    }

    with rlinsight_adapter.task_span(
        tools_kwargs, task_name="hotpotqa", prompt=[{"role": "user", "content": "q"}]
    ) as span:
        span.record_result(
            SimpleNamespace(reward=0.5, accuracy=1.0, finished=True),
            reward_posted=True,
        )

    attributes = captured[0]["attributes"]
    assert attributes["uid"] == "u1"
    assert attributes["session_id"] == "s1"
    assert attributes["task_name"] == "hotpotqa"
    assert attributes["image_ref"] == "image:v1"
    assert attributes["reward"] == 0.5
    assert attributes["accuracy"] == 1.0
    assert attributes["finished"] is True
    assert attributes["reward_posted"] is True


@pytest.mark.cpu
@pytest.mark.level0
def test_generation_span_success_uses_chain_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_trace_span(monkeypatch)
    monkeypatch.setattr(
        rlinsight_adapter,
        "agent_loop_lane_id",
        lambda experiment_name, sample, session, traj: f"{experiment_name}/{sample}/{session}/{traj}",
    )
    span = rlinsight_adapter.start_generation_span(
        {
            "experiment_name": "run",
            "sample": "1",
            "session": "0",
            "state_lane_id": "run/1/0/0",
        }
    )
    span.success(
        prompt_tokens=10,
        completion_tokens=5,
        chain_id=2,
        turn=3,
        assistant_msg={
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"function": {"name": "search"}}],
        },
        finish_reason="stop",
    )
    span.report()

    attributes = captured[0]["attributes"]
    assert attributes["state_lane_id"] == "run/1/0/1"
    assert attributes["traj"] == "1"
    assert attributes["type"] == "tool"
    assert attributes["tools"] == '["search"]'
    assert attributes["turn"] == 3
    assert attributes["finish_reason"] == "stop"


@pytest.mark.cpu
@pytest.mark.level0
def test_old_verl_missing_optional_apis_degrades_to_warnings(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    class OldVerlLogger:
        pass

    monkeypatch.setattr(rlinsight_adapter, "RLInsightLogger", OldVerlLogger)
    monkeypatch.setattr(rlinsight_adapter, "agent_loop_lane_id", None)
    rlinsight_adapter.RolloutTraceConfig.init(
        project_name="project",
        experiment_name="experiment",
        backend=None,
    )

    with caplog.at_level("WARNING", logger="uni_agent.rlinsight_adapter"):
        rlinsight_adapter._report_span("agent_task", start_time_ns=1, attributes={})
        session = rlinsight_adapter.agent_loop_session(sample=7, session=1, global_steps=2)

    assert "does not provide RLInsightLogger.trace_span" in caplog.text
    assert "does not provide RLInsightLogger.agent_loop_session" in caplog.text
    assert session.identity["project"] == "project"
    assert session.identity["state_lane_id"] == "experiment=experiment/sample=7/session=1/traj=0"
    session.finish(status="success", trajectories=[])
