"""Configuration and routing contracts for MAS framework extension points."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.framework.multi_agent_framework import (
    MultiAgentGatewayFramework,
)
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult


def _build_framework(settings, *, reward_config=None):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "unused",
                                    "dispatch_mode": "ray_task",
                                }
                            },
                            "role_policy_mapping": {"planner": "p1", "answerer": "p2"},
                            **settings,
                        },
                    },
                },
            },
        }
    )
    if reward_config is not None:
        config.reward = reward_config
    return MultiAgentGatewayFramework.from_config(
        config=config,
        gateway_manager=SimpleNamespace(),
        policy_processors={"p1": None, "p2": None},
        policy_rollout_configs={},
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("settings", "environment", "expected"),
    [
        ({"log_dir": "mas_logs"}, "environment_logs", "mas_logs"),
        ({"log_dir": None}, "environment_logs", None),
        ({"log_dir": ""}, "environment_logs", None),
        ({}, "environment_logs", "environment_logs"),
        ({}, None, "/tmp/uni_agent_logs"),
    ],
)
def test_log_directory_matches_parent_precedence(monkeypatch, settings, environment, expected):
    if environment is None:
        monkeypatch.delenv("UNI_AGENT_LOG_DIR", raising=False)
    else:
        monkeypatch.setenv("UNI_AGENT_LOG_DIR", environment)
    assert _build_framework(settings)._log_dir == expected


def _reverse_trajectories(trajectories, *, task_result, options):
    assert isinstance(trajectories, tuple)
    assert options == {"enabled": True}
    return [
        replace(trajectory, extra_fields={**trajectory.extra_fields, "processed": True})
        for trajectory in reversed(trajectories)
    ]


async def _reverse_trajectories_async(trajectories, *, task_result, options):
    return _reverse_trajectories(trajectories, task_result=task_result, options=options)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("function_name", ["_reverse_trajectories", "_reverse_trajectories_async"])
@pytest.mark.asyncio
async def test_yaml_postprocessor_runs_and_preserves_policy_routing(function_name):
    framework = _build_framework(
        {
            "log_dir": None,
            "trajectory_postprocessor_fqn": f"{__name__}.{function_name}",
            "trajectory_postprocessor_kwargs": {"options": {"enabled": True}},
        }
    )
    trajectories = [
        Trajectory(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            finished=True,
            reward_score=1.0,
            reward_metrics={"acc": 1.0},
            extra_fields={
                "role": role,
                "policy_name": policy,
                "session_group_id": "group-1",
            },
        )
        for role, policy in (("planner", "p1"), ("answerer", "p2"))
    ]
    result = await framework._apply_trajectory_postprocessor(
        trajectories,
        TaskResult(reward=1.0, finished=True),
    )
    assert [trajectory.extra_fields["policy_name"] for trajectory in result] == [
        "p2",
        "p1",
    ]
    for trajectory in result:
        fields, tags = framework._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields={"uid": "prompt-1", "raw_prompt": "question"},
            session_index=0,
            global_steps=1,
            uid="prompt-1",
        )
        assert fields["extra_fields"]["processed"] is True
        for key in ("policy_name", "role", "session_group_id"):
            assert tags[key] == trajectory.extra_fields[key]
        assert trajectory.reward_score == 1.0
        assert trajectory.reward_metrics == {"acc": 1.0}
    assert all("processed" not in trajectory.extra_fields for trajectory in trajectories)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("settings", "error_type", "message"),
    [
        ({"trajectory_postprocessor_fqn": "  "}, ValueError, "non-empty string"),
        ({"trajectory_postprocessor_fqn": 123}, ValueError, "non-empty string"),
        (
            {"trajectory_postprocessor_kwargs": {"enabled": True}},
            ValueError,
            "requires",
        ),
        (
            {
                "trajectory_postprocessor_fqn": f"{__name__}._reverse_trajectories",
                "trajectory_postprocessor_kwargs": [],
            },
            TypeError,
            "must be a mapping",
        ),
        (
            {"trajectory_postprocessor_fqn": f"{__name__}._NOT_CALLABLE"},
            TypeError,
            "callable",
        ),
    ],
)
def test_invalid_postprocessor_config_fails_during_construction(settings, error_type, message):
    with pytest.raises(error_type, match=message):
        _build_framework(settings)


_NOT_CALLABLE = 42


@pytest.mark.cpu
@pytest.mark.level0
def test_absent_postprocessor_keeps_default_behavior():
    framework = _build_framework({"log_dir": None})
    assert framework._trajectory_postprocessor is None
    assert framework._trajectory_postprocessor_kwargs == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("runner_names", [["search_mas"], ["search_mas", "coding_mas"]])
def test_from_config_accepts_parent_runner_registry(runner_names):
    runners = {runner_name: {"runner_fqn": "unused", "dispatch_mode": "ray_task"} for runner_name in runner_names}
    framework = _build_framework({"agent_runners": runners})
    assert set(framework.runner_registry) == set(runner_names)


@pytest.mark.cpu
@pytest.mark.level0
def test_null_runner_config_uses_shared_parser_error():
    with pytest.raises(AttributeError, match="get"):
        _build_framework({"agent_runners": {"multi_agent": None}})


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("selection", ["all", "longest"])
def test_shared_runner_parser_preserves_selection_and_timeout(selection):
    framework = _build_framework(
        {
            "agent_runners": {
                "multi_agent": {
                    "runner_fqn": "unused",
                    "dispatch_mode": "ray_task",
                    "trajectory_selection": selection,
                    "session_timeout_seconds": 12,
                    "max_concurrent_sessions": 3,
                }
            },
        }
    )
    runner = framework.runner_registry["multi_agent"]
    assert runner.trajectory_selection == selection
    assert runner.session_timeout_seconds == 12
    assert runner.max_concurrent_sessions == 3


@pytest.mark.cpu
@pytest.mark.level0
def test_invalid_trajectory_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown trajectory selection"):
        _build_framework(
            {
                "agent_runners": {
                    "multi_agent": {
                        "runner_fqn": "unused",
                        "trajectory_selection": "per_policy",
                    }
                }
            }
        )


async def _inline_runner(**kwargs):
    return None


@pytest.mark.cpu
@pytest.mark.level0
def test_omitted_dispatch_mode_uses_mainline_inline_default():
    framework = _build_framework(
        {
            "agent_runners": {
                "multi_agent": {
                    "runner_fqn": f"{__name__}._inline_runner",
                }
            }
        }
    )
    assert framework.runner_registry["multi_agent"].dispatch_mode == "inline_async"
    assert framework._inline_runners["multi_agent"] is _inline_runner


@pytest.mark.cpu
@pytest.mark.level0
def test_selection_is_over_the_whole_session_group():
    from uni_agent.framework.framework import _select_session_trajectories

    trajectories = [
        Trajectory([1], [2], [1], extra_fields={"policy_name": "p1", "role": "planner"}),
        Trajectory([1], [2, 3], [1, 1], extra_fields={"policy_name": "p2", "role": "answerer"}),
    ]
    assert _select_session_trajectories("group-1", trajectories, "all") == trajectories
    selected = _select_session_trajectories("group-1", trajectories, "longest")
    assert len(selected) == 1
    assert selected[0] is trajectories[1]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("reward_config", "expect_warning"),
    [
        ({}, False),
        (
            {
                "reward_model": {"enable": True, "enable_resource_pool": False},
                "custom_reward_function": {"path": "custom_reward.py"},
            },
            True,
        ),
        (
            {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "custom_reward_function": {"path": "custom_reward.py"},
            },
            False,
        ),
        (
            {
                "reward_model": {"enable": True, "enable_resource_pool": True},
                "custom_reward_function": {"path": "custom_reward.py"},
            },
            False,
        ),
        (
            {
                "reward_model": {"enable": True},
                "custom_reward_function": {"path": "custom_reward.py"},
            },
            False,
        ),
        ({"reward_model": {"enable": True, "enable_resource_pool": False}}, False),
    ],
)
def test_reward_model_configuration_warning_matches_parent(caplog, reward_config, expect_warning):
    logger_name = "uni_agent.framework.framework"
    with caplog.at_level("WARNING", logger=logger_name):
        framework = _build_framework({"log_dir": None}, reward_config=reward_config)

    warnings = [
        record.getMessage() for record in caplog.records if record.name == logger_name and record.levelname == "WARNING"
    ]
    if expect_warning:
        assert len(warnings) == 1
        assert "colocated reward model" in warnings[0]
        assert "TaskResult/runner_reward_info" in warnings[0]
        assert "rm_scores" in warnings[0]
    else:
        assert warnings == []
    assert framework._custom_reward_function_configured is (
        reward_config.get("custom_reward_function", {}).get("path") is not None
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("selection", ["all", "longest"])
def test_shared_framework_parser_matches_parent_and_mas_factories(selection):
    from uni_agent.framework.framework import GatewayAgentFramework

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "unused",
                                    "dispatch_mode": "ray_task",
                                    "trajectory_selection": selection,
                                    "session_timeout_seconds": 12,
                                    "max_concurrent_sessions": 3,
                                }
                            },
                            "role_policy_mapping": {"planner": "p1"},
                            "log_dir": None,
                            "mask_unfinished_episode": True,
                            "trajectory_postprocessor_fqn": f"{__name__}._reverse_trajectories",
                            "trajectory_postprocessor_kwargs": {"options": {"enabled": True}},
                        }
                    }
                }
            },
            "reward": {"custom_reward_function": {"path": "custom_reward.py"}},
        }
    )
    parsed = GatewayAgentFramework._parse_framework_config(config)
    parent = GatewayAgentFramework.from_config(config=config, gateway_manager=SimpleNamespace())
    mas = MultiAgentGatewayFramework.from_config(
        config=config,
        gateway_manager=SimpleNamespace(),
        policy_processors={"p1": None},
        policy_rollout_configs={},
    )
    for framework in (parent, mas):
        assert framework.runner_registry == parsed["runner_registry"]
        assert framework._log_dir is None
        assert framework._mask_unfinished_episode is True
        assert framework._custom_reward_function_configured is True
        assert framework._trajectory_postprocessor is _reverse_trajectories
        assert framework._trajectory_postprocessor_kwargs == {"options": {"enabled": True}}
