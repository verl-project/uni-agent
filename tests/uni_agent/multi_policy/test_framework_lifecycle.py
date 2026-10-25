import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

from omegaconf import OmegaConf
import pytest

from examples.multi_policy.framework import SessionGroupHandle
from examples.multi_policy.framework import multi_agent_entry as entry_module
from examples.multi_policy.framework import multi_agent_framework as mod
from examples.multi_policy.framework.multi_agent_entry import MultiPolicyRolloutAdapter
from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig


def _framework(*, policy_rollout_configs=None, **kwargs):
    return mod.MultiAgentGatewayFramework(
        gateway_manager=kwargs.pop("gateway_manager", SimpleNamespace()),
        runner_registry={
            "multi_agent": _RunnerConfig(
                runner_fqn="unused",
                runner_kwargs={},
                dispatch_mode="ray_task",
                max_concurrent_sessions=kwargs.pop("max_concurrent_sessions", 0),
                session_timeout_seconds=kwargs.pop("session_timeout_seconds", None),
            ),
        },
        role_policy_mapping={"a": "p1", "b": "p2"},
        policy_processors=kwargs.pop("policy_processors", {"p1": None, "p2": None}),
        policy_rollout_configs=policy_rollout_configs
        or {
            name: OmegaConf.create(
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": True,
                }
            )
            for name in ("p1", "p2")
        },
        **kwargs,
    )


def _from_config_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "unused",
                                    "dispatch_mode": "ray_task",
                                },
                            },
                            "role_policy_mapping": {"a": "p1"},
                        }
                    }
                }
            }
        }
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("partition", ["train", "val"])
@pytest.mark.parametrize("do_sample", [None, True, False])
def test_sampling_defaults_are_built_per_policy(partition, do_sample):
    policy_rollout_configs = {
        "p1": OmegaConf.create(
            {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "calculate_log_probs": True}
        ),
        "p2": OmegaConf.create(
            {
                "temperature": 1.2,
                "top_p": 0.8,
                "top_k": 40,
                "calculate_log_probs": False,
            }
        ),
    }
    for policy_config in policy_rollout_configs.values():
        policy_config.val_kwargs = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 10,
            "do_sample": True,
        }
    framework = _framework(policy_rollout_configs=policy_rollout_configs)
    sample = {"raw_prompt": "q"}
    if do_sample is not None:
        sample["__do_sample__"] = do_sample
    sampling = framework._build_session_sampling_params(
        partition_id=partition, sample_fields=sample
    )
    for policy_name, config in policy_rollout_configs.items():
        expected = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if partition == "val":
            expected.update(temperature=0.3, top_p=0.95, top_k=10)
        if do_sample is False:
            expected.update(temperature=0, top_p=1.0, top_k=-1)
        assert sampling[policy_name] == expected
    assert policy_rollout_configs["p1"].temperature == 0.7
    assert "temperature" not in sample


@pytest.mark.cpu
@pytest.mark.level0
def test_sampling_dicts_are_independent_per_policy():
    framework = _framework()
    sampling = framework._build_session_sampling_params(
        partition_id="train", sample_fields={}
    )
    sampling["p1"]["temperature"] = 9
    assert sampling["p2"]["temperature"] == 1.0


@pytest.mark.cpu
@pytest.mark.level0
def test_multi_agent_framework_constructor_exposes_supported_parent_arguments():
    parameters = inspect.signature(mod.MultiAgentGatewayFramework).parameters
    assert "gateway_manager" in parameters
    assert "runner_registry" in parameters
    assert "runner_config" not in parameters
    assert "reward_loop_worker_handles" in parameters
    assert "rollout_config" in parameters
    assert "kwargs" not in parameters


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_preserves_parent_signature():
    parent = inspect.signature(GatewayAgentFramework.from_config).parameters
    child = inspect.signature(mod.MultiAgentGatewayFramework.from_config).parameters

    for name, parent_param in parent.items():
        assert name in child, f"from_config is missing parent parameter {name!r}"
        child_param = child[name]
        if parent_param.default is not inspect.Parameter.empty:
            assert child_param.default is not inspect.Parameter.empty, (
                f"from_config made parent parameter {name!r} required"
            )


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_accepts_parent_processor_argument():
    framework = mod.MultiAgentGatewayFramework.from_config(
        config=_from_config_config(),
        gateway_manager=SimpleNamespace(),
        processor=object(),
        policy_rollout_configs={},
        policy_processors={"p1": None},
    )
    assert framework._policy_processors == {"p1": None}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("missing", ["policy_rollout_configs", "policy_processors"])
def test_from_config_requires_per_policy_configuration(missing):
    kwargs = {
        "policy_rollout_configs": {},
        "policy_processors": {"p1": None},
    }
    kwargs[missing] = None
    with pytest.raises(ValueError, match="multi_agent_entry"):
        mod.MultiAgentGatewayFramework.from_config(
            config=_from_config_config(),
            gateway_manager=SimpleNamespace(),
            processor=object(),
            **kwargs,
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_uses_shared_runner_config():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "example.runner",
                                    "runner_kwargs": {
                                        "execution": {"backend": "local_process"}
                                    },
                                    "max_concurrent_sessions": 3,
                                    "dispatch_mode": "ray_task",
                                    "trajectory_selection": "all",
                                    "session_timeout_seconds": 12,
                                },
                            },
                            "role_policy_mapping": {"a": "p1"},
                        }
                    }
                }
            }
        }
    )
    framework = mod.MultiAgentGatewayFramework.from_config(
        config=config,
        gateway_manager=SimpleNamespace(),
        policy_rollout_configs={},
        policy_processors={"p1": None},
    )
    runner_config = framework.runner_registry["multi_agent"]
    assert runner_config.runner_fqn == "example.runner"
    assert runner_config.max_concurrent_sessions == 3
    assert runner_config.session_timeout_seconds == 12
    assert runner_config.dispatch_mode == "ray_task"
    assert runner_config.trajectory_selection == "all"
    assert framework._role_policy_mapping == {"a": "p1"}
    assert framework._rollout_config is config.actor_rollout_ref.rollout


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_allows_inline_runner_dispatch():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "uni_agent.framework.task_runner.run_task",
                                    "dispatch_mode": "inline_async",
                                },
                            },
                            "role_policy_mapping": {"a": "p1"},
                        }
                    }
                }
            }
        }
    )
    framework = mod.MultiAgentGatewayFramework.from_config(
        config=config,
        gateway_manager=SimpleNamespace(),
        policy_rollout_configs={},
        policy_processors={"p1": None},
    )
    assert framework.runner_registry["multi_agent"].dispatch_mode == "inline_async"
    from uni_agent.framework.task_runner import run_task

    assert framework._inline_runners["multi_agent"] is run_task


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_rejects_unknown_runner_dispatch_mode():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {
                                    "runner_fqn": "example.runner",
                                    "dispatch_mode": "process",
                                },
                            },
                            "role_policy_mapping": {"a": "p1"},
                        }
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="Unknown dispatch mode"):
        mod.MultiAgentGatewayFramework.from_config(
            config=config,
            gateway_manager=SimpleNamespace(),
            policy_rollout_configs={},
            policy_processors={"p1": None},
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("af_cfg", [{}, {"agent_runners": None}, {"agent_runners": {}}])
def test_from_config_requires_runner_node(af_cfg):
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": af_cfg}}}}
    )
    with pytest.raises(ValueError) as exc_info:
        mod.MultiAgentGatewayFramework.from_config(
            config=config,
            gateway_manager=SimpleNamespace(),
            policy_rollout_configs={},
            policy_processors={"p1": None},
        )
    assert "agent_runners is required" in str(exc_info.value)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("runner_fqn", [None, ""])
def test_from_config_requires_runner_fqn(runner_fqn):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "multi_agent": {"runner_fqn": runner_fqn}
                            },
                            "role_policy_mapping": {"a": "p1"},
                        }
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="runner_fqn is required"):
        mod.MultiAgentGatewayFramework.from_config(
            config=config,
            gateway_manager=SimpleNamespace(),
            policy_rollout_configs={},
            policy_processors={"p1": None},
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_session_group_creation_uses_group_manager_lifecycle():
    handle = SessionGroupHandle(
        "r",
        {
            "a": "http://gateway/sessions/r:a/v1",
            "b": "http://gateway/sessions/r:b/v1",
        },
    )
    gateway = SimpleNamespace(
        create_session=AsyncMock(return_value=handle),
        finalize_session=AsyncMock(return_value=[]),
        abort_session=AsyncMock(),
    )
    framework = _framework(gateway_manager=gateway)

    opened = await framework._create_session(
        "r",
        metadata={"global_steps": 1},
        sampling_params={"p1": {"temperature": 0.1}},
    )
    assert opened is handle
    gateway.create_session.assert_awaited_once_with(
        "r",
        role_policy_mapping={"a": "p1", "b": "p2"},
        sampling_params={"p1": {"temperature": 0.1}},
        metadata={"global_steps": 1},
    )

    assert await framework.gateway_manager.finalize_session("r") == []
    gateway.finalize_session.assert_awaited_once_with("r")
    await framework.gateway_manager.abort_session("r")
    gateway.abort_session.assert_awaited_once_with("r")


@pytest.mark.cpu
@pytest.mark.level0
def test_multi_agent_framework_does_not_override_manager_lifecycle_hooks():
    assert "_finalize_session" not in mod.MultiAgentGatewayFramework.__dict__
    assert "_abort_session" not in mod.MultiAgentGatewayFramework.__dict__


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_session_group_sampling_is_forwarded_to_manager():
    gateway = SimpleNamespace(
        create_session=AsyncMock(
            return_value=SessionGroupHandle(
                "r", {"a": "http://gateway/sessions/r:a/v1"}
            )
        )
    )
    framework = _framework(gateway_manager=gateway)
    sampling = {"p1": {"temperature": 0.1}}
    await framework._create_session("r", sampling_params=sampling)
    assert gateway.create_session.call_args.kwargs["sampling_params"] is sampling


@pytest.mark.cpu
@pytest.mark.level0
def test_extra_tags_expose_routing_metadata():
    framework = _framework()
    trajectory = SimpleNamespace(
        extra_fields={"role": "answerer", "policy_name": "p1", "session_group_id": "r"}
    )
    assert framework._extra_tags(trajectory) == {
        "role": "answerer",
        "policy_name": "p1",
        "session_group_id": "r",
    }


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_worker_initializes_tq_and_delegates_generate(monkeypatch):
    framework = SimpleNamespace(generate_sequences=AsyncMock())
    captured = {}

    def build_framework(**kwargs):
        captured.update(kwargs)
        return framework

    monkeypatch.setattr(entry_module, "build_agent_framework", build_framework)
    monkeypatch.setattr(entry_module, "tq", SimpleNamespace(init=lambda: None))
    worker_class = (
        entry_module.MultiAgentFrameworkWorker.__ray_metadata__.modified_class
    )
    policy_rollout_configs = {"p1": OmegaConf.create({"temperature": 0.7})}
    worker = worker_class(
        config=None,
        gateway_manager=None,
        policy_gateway_configs={},
        policy_rollout_configs=policy_rollout_configs,
    )
    assert captured["policy_rollout_configs"] is policy_rollout_configs
    await worker.generate_sequences("batch")
    framework.generate_sequences.assert_awaited_once_with("batch")
    assert not hasattr(worker, "shutdown")


@pytest.mark.cpu
@pytest.mark.level0
def test_adapter_submits_and_waits():
    submitted = []
    worker = SimpleNamespace(
        generate_sequences=SimpleNamespace(
            remote=lambda prompts: submitted.append(prompts) or "ref"
        )
    )
    adapter = MultiPolicyRolloutAdapter(
        framework_worker=worker,
        gateway_manager=SimpleNamespace(),
    )
    assert adapter.generate_sequences("batch") is None
    assert submitted == ["batch"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("runner_names", "agent_name", "expected_runner"),
    [
        (["search_mas"], None, "search_mas"),
        (["search_mas", "coding_mas"], "coding_mas", "coding_mas"),
    ],
)
@pytest.mark.asyncio
async def test_constructor_uses_parent_runner_selection(
    runner_names, agent_name, expected_runner
):
    runner_registry = {
        name: _RunnerConfig(
            runner_fqn="unused",
            runner_kwargs={},
            dispatch_mode="ray_task",
            max_concurrent_sessions=0,
        )
        for name in runner_names
    }
    framework = mod.MultiAgentGatewayFramework(
        gateway_manager=SimpleNamespace(),
        runner_registry=runner_registry,
        role_policy_mapping={"a": "p1"},
        policy_processors={"p1": None},
    )
    framework._run_agent_episode = AsyncMock(return_value=([], {}))
    sample_fields = {} if agent_name is None else {"agent_name": agent_name}
    await framework._run_agent_episode_with_concurrency_limit(
        sample_fields=sample_fields,
        sample_index=0,
        session_index=0,
        global_steps=1,
        sampling_params={},
    )
    assert (
        framework._run_agent_episode.await_args.kwargs["runner_name"] == expected_runner
    )
    assert (
        framework._run_agent_episode.await_args.kwargs["runner_config"]
        is runner_registry[expected_runner]
    )
