"""Tests for the per-role wrapping of multi_policy select functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.framework import multi_agent_framework as mod
from uni_agent.framework.framework import _RunnerConfig
from uni_agent.framework.trajectory_selection import _CustomSelectFunction
from uni_agent.gateway.session import Trajectory


def _trajectory(role: str) -> Trajectory:
    return Trajectory(
        prompt_ids=[1],
        response_ids=[1],
        response_mask=[1],
        extra_fields={"role": role},
    )


def _framework(select_fn, monkeypatch):
    monkeypatch.setattr(
        "uni_agent.framework.framework._load_select_function",
        lambda _custom: select_fn,
    )
    runner = _RunnerConfig(
        runner_fqn="unused",
        runner_kwargs={},
        dispatch_mode="ray_task",
        max_concurrent_sessions=0,
        custom_select_function=_CustomSelectFunction(path="selectors.py", name="rule"),
    )
    return mod.MultiAgentGatewayFramework(
        gateway_manager=SimpleNamespace(),
        runner_registry={"multi_agent": runner},
        role_policy_mapping={"verifier": "p1", "searcher": "p2", "answerer": "p3"},
        policy_processors={"p1": None, "p2": None, "p3": None},
        policy_rollout_configs={
            name: OmegaConf.create({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "calculate_log_probs": True})
            for name in ("p1", "p2", "p3")
        },
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_select_function_is_called_once_per_role(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def rule(session_id, trajectories):
        calls.append((session_id, [trajectory.extra_fields["role"] for trajectory in trajectories]))
        return trajectories

    framework = _framework(rule, monkeypatch)
    wrapped = framework._select_functions["multi_agent"]
    trajectories = [
        _trajectory("verifier"),
        _trajectory("verifier"),
        _trajectory("answerer"),
    ]

    kept = await wrapped("session-1", trajectories)

    assert calls == [
        ("session-1", ["verifier", "verifier"]),
        ("session-1", ["answerer"]),
    ]
    assert len(kept) == 3


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_async_select_function_is_supported(monkeypatch):
    calls: list[str] = []

    async def rule(session_id, trajectories):
        calls.append(trajectories[0].extra_fields["role"])
        return trajectories

    framework = _framework(rule, monkeypatch)
    wrapped = framework._select_functions["multi_agent"]

    kept = await wrapped("session-1", [_trajectory("verifier"), _trajectory("answerer")])

    assert calls == ["verifier", "answerer"]
    assert len(kept) == 2


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_role_keeps_its_trajectories(monkeypatch):
    def rule(session_id, trajectories):
        if trajectories[0].extra_fields["role"] == "searcher":
            raise RuntimeError("boom")
        return trajectories[:1]

    framework = _framework(rule, monkeypatch)
    wrapped = framework._select_functions["multi_agent"]
    verifier = [_trajectory("verifier"), _trajectory("verifier")]
    searcher = [_trajectory("searcher"), _trajectory("searcher")]

    kept = await wrapped("session-1", [*verifier, *searcher])

    # verifier is filtered to one; searcher falls back to both.
    assert len(kept) == 3
    assert kept[0] is verifier[0]
    assert kept[1] is searcher[0]
    assert kept[2] is searcher[1]
