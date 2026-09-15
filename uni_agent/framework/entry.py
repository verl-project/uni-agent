"""Factory entry + trainer-facing adapter for the agent framework stack.

`build_gateway_manager` owns gateway-universal wiring (driver-side); the trainer
adapter creates the manager and injects it so the framework only handles its own
agent runner, reward dispatch, and framework-specific config fields.

`AgentFrameworkRolloutAdapter` satisfies the trainer's
`agent_loop_manager_class` extension-point contract; recipes wire it in via
yaml without authoring per-recipe glue:

    actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        uni_agent.framework.entry.AgentFrameworkRolloutAdapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ray
from omegaconf import OmegaConf

from uni_agent.framework.base import AgentFramework
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.manager import GatewayManager
from uni_agent.rl_insight.adapter import init_rollout_trace_config
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.transferqueue_utils import tq
from verl.workers.config import HFModelConfig, RolloutConfig

if TYPE_CHECKING:
    from tensordict import TensorDict

_DEFAULT_FRAMEWORK_CLASS = "uni_agent.framework.framework.GatewayAgentFramework"


def build_gateway_manager(*, config, llm_client) -> GatewayManager:
    """Spawn the gateway actor pool (driver-side, driver-owned) and return its manager."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    data_cfg = config.data
    model_cfg = config.actor_rollout_ref.model
    rollout_cfg = config.actor_rollout_ref.rollout
    af_cfg = rollout_cfg.custom.agent_framework

    apply_chat_template_kwargs = data_cfg.get("apply_chat_template_kwargs", {})
    mm_processor_kwargs = data_cfg.get("mm_processor_kwargs", {})

    # Match AgentLoopWorker pattern: self-load tokenizer/processor via HFModelConfig.
    rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_cfg)
    model_config: HFModelConfig = omega_conf_to_dataclass(model_cfg)
    # TODO: Gateway session capacity is prompt_length + response_length, which can
    # disagree with actor_rollout_ref.rollout.max_model_len (engine context window
    # or extra slack). Forward that knob into GatewayActorConfig once the override
    # path is restored so session clipping matches the engine budget.
    gateway_actor_config = GatewayActorConfig(
        tokenizer=model_config.tokenizer,
        processor=model_config.processor,
        tool_parser_name=rollout_config.multi_turn.format,
        rollout_backend=rollout_config.name,
        enable_tool_parser_cache=af_cfg.get("enable_tool_parser_cache", True),
        hf_model_type=getattr(model_config.hf_config, "model_type", None),
        apply_chat_template_kwargs=dict(apply_chat_template_kwargs),
        mm_processor_kwargs=dict(mm_processor_kwargs),
        prompt_length=rollout_config.prompt_length,
        response_length=rollout_config.response_length,
        enable_last_assistant_rollback=af_cfg.get("enable_last_assistant_rollback", True),
    )

    return GatewayManager(
        llm_client=llm_client,
        gateway_count=int(af_cfg["gateway_count"]),
        gateway_actor_config=gateway_actor_config,
    )


def build_agent_framework(
    *,
    config,
    gateway_manager,
    reward_loop_worker_handles=None,
) -> AgentFramework:
    """Wire the configured framework subclass over an injected gateway manager."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
    model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)

    framework_cls = load_class_from_fqn(str(af_cfg.get("framework_class_fqn", _DEFAULT_FRAMEWORK_CLASS)))
    return framework_cls.from_config(
        config=config,
        gateway_manager=gateway_manager,
        processor=model_config.processor,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


@ray.remote
class AgentFrameworkWorker:
    """Ray actor host: initializes TQ in this process and owns one AgentFramework.

    Construction is synchronous (no async setup round-trip); the gateway manager
    is created driver-side and injected so its actors are not owned by this worker.
    """

    def __init__(self, *, config, gateway_manager, reward_loop_worker_handles=None) -> None:
        tq.init()
        init_rollout_trace_config(config)
        self.framework = build_agent_framework(
            config=config,
            gateway_manager=gateway_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    async def generate_sequences(self, prompts) -> None:
        await self.framework.generate_sequences(prompts)

    async def submit_sessions(self, prompts: TensorDict):
        """Require the Framework's admission-only submission interface."""
        await self.framework.submit_sessions(prompts)


class AgentFrameworkRolloutAdapter:
    """Trainer-facing adapter satisfying the `agent_loop_manager_class` contract.

    Holds zero recipe-specific logic; every agent-framework recipe wires the
    same class in yaml. The adapter owns the gateway manager (driver-side) and
    injects it into the framework worker. The caller owns version availability;
    the adapter forwards the selected version unchanged. Per-runner
    ``max_concurrent_sessions`` is the worker-side concurrency cap.
    Laminar training explicitly uses ``submit_sessions`` for admission backpressure.
    Ordinary generation only requires ``generate_sequences`` on the framework;
    there is no completion-based admission fallback.
    """

    def __init__(self) -> None:
        self.framework_worker = None
        # Driver-owned so the gateway actors outlive the framework worker; also
        # the handle through which teardown can be driven once a call site exists.
        self.gateway_manager = None

    @classmethod
    def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        **_,
    ) -> AgentFrameworkRolloutAdapter:
        if teacher_client is not None:
            raise ValueError(
                "AgentFrameworkRolloutAdapter does not support teacher_client yet; "
                "disable teacher policy/distillation or use an AgentLoopManager that supports it."
            )

        agent_runners = (
            OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners", default={}) or {}
        )
        runner_limits = [int(runner.get("max_concurrent_sessions", 0) or 0) for runner in agent_runners.values()]
        # Cover the session semaphore budgets without lowering Ray's async default.
        # A batch can retain its RPC slot for just one straggling session,
        # so dividing by batch size or rollout.n would throttle useful work.
        worker_concurrency = max(1000, sum(limit for limit in runner_limits if limit > 0))

        gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
        framework_worker = AgentFrameworkWorker.options(max_concurrency=worker_concurrency).remote(
            config=config,
            gateway_manager=gateway_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework_worker = framework_worker
        instance.gateway_manager = gateway_manager
        return instance

    def generate_sequences(self, prompts) -> None:
        """Submit a TQ batch generation task without waiting for rollout results."""
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")

        self.framework_worker.generate_sequences.remote(prompts)
        return None

    def submit_sessions(self, prompts: TensorDict) -> None:
        """Wait for session admission; require framework support for this capability."""
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before submit_sessions")

        ray.get(self.framework_worker.submit_sessions.remote(prompts))
        return None

    def generate_sequences_and_wait(self, prompts) -> None:
        """Blocking variant of :meth:`generate_sequences` for standalone (non-trainer) runs.

        :meth:`generate_sequences` is fire-and-forget (the trainer consumes TQ asynchronously
        via its ReplayBuffer); this awaits the framework worker so the caller knows every
        session's trajectory has landed in TQ, and re-raises any worker-side error.
        """
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")

        ray.get(self.framework_worker.generate_sequences.remote(prompts))
        return None
