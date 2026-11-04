from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.framework import multi_agent_entry as entry_module
from examples.multi_policy.framework.multi_agent_entry import (
    MultiPolicyRolloutAdapter,
    build_policy_gateway_configs,
)
from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer


@pytest.mark.cpu
@pytest.mark.level0
def test_recipe_composes_shared_counts_and_policy_validation_sampling():
    recipe_config = Path(__file__).parents[3] / "examples" / "multi_policy" / "config" / "multi_policy.yaml"
    outer_config = OmegaConf.load(recipe_config)
    outer_config.actor_rollout_ref.rollout.n = 6
    outer_config.actor_rollout_ref.rollout.val_kwargs.n = 2
    policy_2_validation = outer_config.policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.val_kwargs
    policy_2_validation.do_sample = True
    policy_2_validation.temperature = 0.4
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = outer_config
    trainer.total_training_steps = int(outer_config.trainer.total_training_steps)
    trainer.parameter_sync_step = int(outer_config.trainer.v1.separate_async.parameter_sync_step)
    policy_configs = trainer._compose_policy_configs()
    for policy_name, policy_config in policy_configs.items():
        rollout = policy_config.actor_rollout_ref.rollout
        assert rollout.n == 6
        assert rollout.val_kwargs.n == 2
        assert rollout.val_kwargs.do_sample == (policy_name == "policy_2")
        assert rollout.val_kwargs.temperature == (0.4 if policy_name == "policy_2" else 0.0)
    assert set(outer_config.actor_rollout_ref.rollout.val_kwargs) == {"n"}


def config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "gateway_count": 2,
                            "enable_tool_parser_cache": False,
                            "allowed_request_sampling_param_keys": ["temperature"],
                        }
                    }
                }
            },
        }
    )


def policy_config(
    policy_name,
    *,
    backend="vllm",
    prompt=100,
    response=20,
    apply_chat_template_kwargs=None,
    mm_processor_kwargs=None,
):
    return OmegaConf.create(
        {
            "data": {
                "apply_chat_template_kwargs": apply_chat_template_kwargs or {},
                "mm_processor_kwargs": mm_processor_kwargs or {},
            },
            "actor_rollout_ref": {
                "model": {"_kind": "model", "policy_name": policy_name},
                "rollout": {
                    "_kind": "rollout",
                    "name": backend,
                    "prompt_length": prompt,
                    "response_length": response,
                    "multi_turn": {"format": "hermes"},
                },
            },
        }
    )


@pytest.fixture(autouse=True)
def patch_gateway_dataclass_conversion(monkeypatch):
    def convert(section):
        if section["_kind"] == "model":
            policy_name = section["policy_name"]
            return SimpleNamespace(
                tokenizer=SimpleNamespace(policy_name=policy_name),
                processor=SimpleNamespace(policy_name=policy_name),
                hf_config=SimpleNamespace(model_type=policy_name),
            )
        return SimpleNamespace(
            multi_turn=SimpleNamespace(format=section["multi_turn"]["format"]),
            name=section["name"],
            prompt_length=section["prompt_length"],
            response_length=section["response_length"],
        )

    monkeypatch.setattr(entry_module, "omega_conf_to_dataclass", convert)


@pytest.mark.cpu
@pytest.mark.level0
def test_policy_configs_use_per_policy_runtime_and_input_processing_kwargs():
    cfg = config()
    policy_configs = {
        "alpha": policy_config(
            "qwen",
            backend="vllm",
            prompt=100,
            apply_chat_template_kwargs={"enable_thinking": True},
            mm_processor_kwargs={"max_pixels": 100},
        ),
        "beta": policy_config(
            "llama",
            backend="sglang",
            prompt=200,
            response=30,
            apply_chat_template_kwargs={"enable_thinking": False},
            mm_processor_kwargs={"max_pixels": 200},
        ),
    }
    result = build_policy_gateway_configs(cfg, policy_configs)
    assert set(result) == {"alpha", "beta"}
    assert result["alpha"].tokenizer.policy_name == "qwen"
    assert result["alpha"].hf_model_type == "qwen"
    assert result["alpha"].rollout_backend == "vllm"
    assert result["alpha"].prompt_length == 100
    assert result["alpha"].apply_chat_template_kwargs == {"enable_thinking": True}
    assert result["alpha"].mm_processor_kwargs == {"max_pixels": 100}
    assert result["alpha"].allowed_request_sampling_param_keys == {"temperature"}
    assert result["alpha"].enable_tool_parser_cache is False
    assert result["beta"].rollout_backend == "sglang"
    assert result["beta"].prompt_length == 200
    assert result["beta"].response_length == 30
    assert result["beta"].apply_chat_template_kwargs == {"enable_thinking": False}
    assert result["beta"].mm_processor_kwargs == {"max_pixels": 200}
    assert result["beta"].allowed_request_sampling_param_keys == {"temperature"}


@pytest.mark.cpu
@pytest.mark.level0
def test_invalid_sampling_keys_fail():
    cfg = config()
    cfg.actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys = "temperature"
    with pytest.raises(ValueError, match="allowed_request_sampling_param_keys"):
        build_policy_gateway_configs(cfg, {"alpha": policy_config("qwen")})


@pytest.mark.cpu
@pytest.mark.level0
def test_adapter_rejects_mismatched_clients_before_creating_manager():
    with pytest.raises(ValueError, match="identical policy names"):
        MultiPolicyRolloutAdapter.create(
            config=config(),
            policy_clients={"other": object()},
            policy_configs={"alpha": policy_config("qwen")},
        )


@pytest.mark.cpu
@pytest.mark.level0
def test_adapter_passes_full_policy_gateway_configs_to_shared_manager(monkeypatch):
    captured = {}
    worker_kwargs = {}

    def manager_init(self, **kwargs):
        captured.update(kwargs)

    framework_worker = SimpleNamespace()
    fake_manager = type("FakeManager", (), {"__init__": manager_init})

    def create_worker(**kwargs):
        worker_kwargs.update(kwargs)
        return framework_worker

    monkeypatch.setattr(entry_module, "MultiAgentGatewayManager", fake_manager)
    monkeypatch.setattr(
        entry_module.MultiAgentFrameworkWorker,
        "remote",
        create_worker,
    )
    composed_config = policy_config("qwen", prompt=321)
    adapter = MultiPolicyRolloutAdapter.create(
        config=config(),
        policy_clients={"alpha": object()},
        policy_configs={"alpha": composed_config},
    )
    assert adapter.framework_worker is framework_worker
    assert isinstance(adapter.gateway_manager, fake_manager)
    assert captured["gateway_count"] == 2
    assert set(captured["llm_client"]) == {"alpha"}
    assert captured["gateway_actor_config"]["alpha"].prompt_length == 321
    assert captured["gateway_actor_config"]["alpha"].hf_model_type == "qwen"
    assert worker_kwargs["policy_rollout_configs"]["alpha"] is composed_config.actor_rollout_ref.rollout


@pytest.mark.cpu
@pytest.mark.level0
def test_trainer_exposes_composed_policy_configs():
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.policy_configs = {"alpha": policy_config("qwen")}
    configs = trainer.get_policy_configs()
    assert configs == trainer.policy_configs
    assert configs is not trainer.policy_configs
