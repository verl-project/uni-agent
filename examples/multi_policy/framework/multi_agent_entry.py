from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import ray
from omegaconf import DictConfig, OmegaConf

from examples.multi_policy.framework.multi_agent_framework import (
    MultiAgentGatewayFramework,
)
from examples.multi_policy.framework.multi_agent_gateway_manager import (
    MultiAgentGatewayManager,
)
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.rl_insight.adapter import init_rollout_trace_config
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.transferqueue_utils import tq
from verl.workers.config import HFModelConfig, RolloutConfig

if TYPE_CHECKING:
    from verl.workers.rollout.llm_server import LLMServerClient


def build_policy_gateway_configs(
    config: DictConfig,
    policy_configs: Mapping[str, DictConfig],
) -> dict[str, GatewayActorConfig]:
    """Build one GatewayActorConfig from each composed policy config.

    Model, rollout, chat-template, and multimodal processing fields come from
    each policy config. Shared Gateway/session switches and sampling-key
    settings come from the outer multi-policy config.
    """
    if not policy_configs:
        raise ValueError("policy_configs must not be empty")

    agent_framework_config = config.actor_rollout_ref.rollout.custom.agent_framework
    allowed_sampling_keys = agent_framework_config.get("allowed_request_sampling_param_keys")
    if allowed_sampling_keys is not None:
        if not isinstance(allowed_sampling_keys, list | tuple) and not OmegaConf.is_list(allowed_sampling_keys):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
        if any(not isinstance(key, str) for key in allowed_sampling_keys):
            raise ValueError("allowed_request_sampling_param_keys must be a list of strings or null")
        allowed_sampling_keys = set(allowed_sampling_keys)

    policy_gateway_configs: dict[str, GatewayActorConfig] = {}
    for policy_name, policy_config in policy_configs.items():
        data_config = policy_config.data
        model_config: HFModelConfig = omega_conf_to_dataclass(policy_config.actor_rollout_ref.model)
        rollout_config: RolloutConfig = omega_conf_to_dataclass(policy_config.actor_rollout_ref.rollout)
        apply_chat_template_kwargs = dict(data_config.get("apply_chat_template_kwargs", {}))
        mm_processor_kwargs = dict(data_config.get("mm_processor_kwargs", {}))
        tokenizer = model_config.tokenizer
        if tokenizer is None:
            raise ValueError(f"policy {policy_name!r} has no tokenizer")

        policy_gateway_configs[policy_name] = GatewayActorConfig(
            tokenizer=tokenizer,
            processor=model_config.processor,
            tool_parser_name=rollout_config.multi_turn.format,
            rollout_backend=rollout_config.name,
            enable_tool_parser_cache=agent_framework_config.get("enable_tool_parser_cache", True),
            hf_model_type=getattr(model_config.hf_config, "model_type", None),
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            prompt_length=rollout_config.prompt_length,
            response_length=rollout_config.response_length,
            enable_last_assistant_rollback=agent_framework_config.get("enable_last_assistant_rollback", True),
            allowed_request_sampling_param_keys=allowed_sampling_keys,
            coalesce_reserved_exact_requests=agent_framework_config.get("coalesce_reserved_exact_requests", True),
        )
    return policy_gateway_configs


def build_gateway_manager(
    *,
    config,
    policy_clients: Mapping[str, LLMServerClient],
    policy_gateway_configs: Mapping[str, GatewayActorConfig],
) -> MultiAgentGatewayManager:
    """Build the shared Gateway manager from per-policy Gateway configs."""
    if not policy_gateway_configs:
        raise ValueError("policy_gateway_configs must not be empty")
    if set(policy_clients) != set(policy_gateway_configs):
        raise ValueError("policy_clients and policy_gateway_configs must have identical policy names")

    agent_framework_config = config.actor_rollout_ref.rollout.custom.agent_framework
    return MultiAgentGatewayManager(
        llm_client=policy_clients,
        gateway_count=int(agent_framework_config["gateway_count"]),
        gateway_actor_config=policy_gateway_configs,
    )


def build_agent_framework(
    *,
    config,
    gateway_manager: MultiAgentGatewayManager,
    policy_gateway_configs: Mapping[str, GatewayActorConfig],
    policy_rollout_configs: Mapping[str, DictConfig],
    reward_loop_worker_handles=None,
) -> MultiAgentGatewayFramework:
    """Build the multi-agent framework over an injected Gateway manager."""
    policy_processors = {
        policy_name: gateway_config.processor for policy_name, gateway_config in policy_gateway_configs.items()
    }
    return MultiAgentGatewayFramework.from_config(
        config=config,
        gateway_manager=gateway_manager,
        policy_processors=policy_processors,
        policy_rollout_configs=policy_rollout_configs,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


@ray.remote
class MultiAgentFrameworkWorker:
    """Ray actor that owns one multi-agent framework instance."""

    def __init__(
        self,
        *,
        config: DictConfig,
        gateway_manager: MultiAgentGatewayManager,
        policy_gateway_configs: Mapping[str, GatewayActorConfig],
        policy_rollout_configs: Mapping[str, DictConfig],
        reward_loop_worker_handles: Sequence[Any] | None = None,
    ) -> None:
        tq.init()
        if config is not None:
            init_rollout_trace_config(config)
        self.framework = build_agent_framework(
            config=config,
            gateway_manager=gateway_manager,
            policy_gateway_configs=policy_gateway_configs,
            policy_rollout_configs=policy_rollout_configs,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    async def generate_sequences(self, prompts) -> None:
        await self.framework.generate_sequences(prompts)


class MultiPolicyRolloutAdapter:
    """Recipe-local trainer adapter for multi-policy MAS rollouts.

    During verl v1 training, the adapter and its Gateway actors are owned by
    the Ray task-runner actor and are reclaimed with that owner.
    """

    def __init__(
        self,
        *,
        framework_worker: ray.actor.ActorHandle,
        gateway_manager: MultiAgentGatewayManager,
    ) -> None:
        self.framework_worker = framework_worker
        self.gateway_manager = gateway_manager

    @classmethod
    def create(
        cls,
        *,
        config: DictConfig,
        policy_clients: Mapping[str, LLMServerClient],
        policy_configs: Mapping[str, DictConfig],
        reward_loop_worker_handles: Sequence[Any] | None = None,
        **_,
    ) -> MultiPolicyRolloutAdapter:
        policy_gateway_configs = build_policy_gateway_configs(config, policy_configs)
        policy_rollout_configs = {
            policy_name: policy_config.actor_rollout_ref.rollout
            for policy_name, policy_config in policy_configs.items()
        }
        gateway_manager = build_gateway_manager(
            config=config,
            policy_clients=policy_clients,
            policy_gateway_configs=policy_gateway_configs,
        )
        try:
            framework_worker = MultiAgentFrameworkWorker.remote(
                config=config,
                gateway_manager=gateway_manager,
                policy_gateway_configs=policy_gateway_configs,
                policy_rollout_configs=policy_rollout_configs,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
        except Exception:
            asyncio.run(gateway_manager.shutdown())
            raise
        return cls(
            framework_worker=framework_worker,
            gateway_manager=gateway_manager,
        )

    def generate_sequences(self, prompts) -> None:
        self.framework_worker.generate_sequences.remote(prompts)
        return None

    def generate_sequences_and_wait(self, prompts) -> None:
        ray.get(self.framework_worker.generate_sequences.remote(prompts))
