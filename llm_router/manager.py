"""LLMRouter: drop-in replacement for verl's AgentLoopManager.

The trainer talks to this class via the exact same public surface as
AgentLoopManager. Internally it owns a LoadBalancer wrapping a RouterPolicy.
Replica creation and per-trajectory AgentLoopWorker logic is reused from verl.
"""
from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig, open_dict

from llm_router.config import parse_config
from llm_router.load_balancer import LoadBalancer
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.utils.ray_utils import auto_await

if TYPE_CHECKING:
    from verl.protocol import DataProto
    from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup


class LLMRouter:
    """Drop-in replacement for verl.experimental.agent_loop.AgentLoopManager."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup | None = None,
        rollout_resource_pool: RayResourcePool | None = None,
        teacher_model_manager=None,
        reward_loop_worker_handles=None,
    ):
        # Reuse verl helpers — they live alongside AgentLoopManager.
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopWorker,
            _get_rollout_and_model_config,
        )
        from verl.workers.rollout.replica import get_rollout_replica_class

        self.config = config
        self.rollout_config, self.model_config = _get_rollout_and_model_config(config)
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.teacher_model_manager = teacher_model_manager
        # Plan A: distillation streaming is not yet supported by LLMRouter.
        # Pin attrs to False so the delegated generate_sequences() path doesn't
        # AttributeError, and reject teacher_model_manager explicitly so the
        # regression is loud rather than silent.
        # TODO(plan-followup): wire teacher_model_manager through worker init.
        self.distillation_enabled = False
        self.stream_teacher_with_rollout = False
        if teacher_model_manager is not None:
            raise NotImplementedError(
                "LLMRouter does not yet support teacher_model_manager streaming. "
                "Use the stock AgentLoopManager for distillation runs until a "
                "future Plan extends LLMRouter."
            )
        self.reward_loop_worker_handles = reward_loop_worker_handles

        self._llm_router_config = parse_config(self._router_config_section())
        self.rollout_replica_class = get_rollout_replica_class(self.rollout_config.name)
        self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

        self.rollout_replicas: list = []
        self.server_handles: list = []
        self.server_ids: list[str] = []
        self.server_addresses: list[str] = []
        self.agent_loop_workers: list = []
        self.load_balancer: ray.actor.ActorHandle | None = None
        self._load_balancer_actor_name = f"llm_router_lb_{uuid4().hex}"

    def _router_config_section(self):
        """Return llm_router config without polluting verl RolloutConfig."""
        actor_rollout_ref = self.config.get("actor_rollout_ref", {}) or {}
        if "llm_router" in actor_rollout_ref:
            return actor_rollout_ref.get("llm_router", {}) or {}
        return self.rollout_config.get("llm_router", {}) or {}

    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup | None = None,
        rollout_resource_pool: RayResourcePool | None = None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
    ) -> LLMRouter:
        instance = cls(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        await instance._initialize_llm_servers()
        await instance._init_load_balancer()
        await instance._init_agent_loop_workers()
        return instance

    async def _initialize_llm_servers(self) -> None:
        rcfg = self.rollout_config
        rollout_world_size = (
            rcfg.tensor_model_parallel_size
            * rcfg.data_parallel_size
            * rcfg.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else rcfg.n_gpus_per_node * rcfg.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.server_ids = [f"replica-{r}" for r in range(num_replicas)]
        self.rollout_replicas = []
        for r in range(num_replicas):
            self.rollout_replicas.append(
                self.rollout_replica_class(
                    replica_rank=r,
                    config=self._rollout_config_for_replica(r),
                    model_config=self.model_config,
                    gpus_per_node=rcfg.n_gpus_per_node,
                )
            )
        if self.worker_group and rcfg.name != "trtllm":
            await asyncio.gather(*[s.init_hybrid(self.worker_group) for s in self.rollout_replicas])
        elif self.worker_group and rcfg.name == "trtllm":
            await asyncio.gather(
                *[
                    s.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for s in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[s.init_standalone() for s in self.rollout_replicas])

        self.server_handles = [s._server_handle for s in self.rollout_replicas]
        self.server_addresses = [s._server_address for s in self.rollout_replicas]

        print(f"LLMRouter: {dict(zip(self.server_ids, self.server_addresses, strict=True))}")

        # Update Prometheus configuration with server addresses (parity with AgentLoopManager).
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(
                self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
            )

    async def _init_load_balancer(self) -> None:
        self.load_balancer = LoadBalancer.options(
            name=self._load_balancer_actor_name,
        ).remote(
            server_ids=self.server_ids,
            policy_name=self._llm_router_config.policy,
            routing_cache_size=self._llm_router_config.routing_cache_size,
            hit_threshold=self._llm_router_config.hit_threshold,
            gpu_hit_threshold=self._llm_router_config.gpu_hit_threshold,
            cpu_hit_threshold=self._llm_router_config.cpu_hit_threshold,
            load_threshold=self._llm_router_config.load_threshold,
            max_prefix_entries_per_server=self._llm_router_config.max_prefix_entries_per_server,
            server_aliases=self._server_aliases(),
        )

    async def _init_agent_loop_workers(self) -> None:
        num_workers = self.rollout_config.agent.num_workers
        servers = list(zip(self.server_ids, self.server_handles, strict=True))
        node_ids = [
            n["NodeID"]
            for n in ray.nodes()
            if n["Alive"] and n["Resources"].get("CPU", 0) > 0
        ]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    servers,
                    self.load_balancer,
                    None,  # teacher_servers
                    None,  # teacher_load_balancer_handle
                    self.reward_loop_worker_handles,
                )
            )

    def _rollout_config_for_replica(self, replica_rank: int) -> DictConfig:
        if not self._uses_mooncake_connector(self.rollout_config):
            return self.rollout_config
        replica_config = copy.deepcopy(self.rollout_config)
        server_id = f"replica-{replica_rank}"
        with open_dict(replica_config):
            kv_transfer_config = replica_config.get("kv_transfer_config")
            if kv_transfer_config is None:
                replica_config.kv_transfer_config = {}
                kv_transfer_config = replica_config.kv_transfer_config
            extra = kv_transfer_config.get("kv_connector_extra_config") or {}
            extra = dict(extra)
            extra.setdefault("server_id", server_id)
            extra.setdefault("load_balancer_actor_name", self._load_balancer_actor_name)
            kv_transfer_config.kv_connector_extra_config = extra
        return replica_config

    def _uses_mooncake_connector(self, rollout_config: DictConfig) -> bool:
        kv_transfer_config = rollout_config.get("kv_transfer_config", {}) or {}
        connector_name = str(kv_transfer_config.get("kv_connector", ""))
        return connector_name == "MooncakeKVConnector"

    def _server_aliases(self) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {sid: [sid] for sid in self.server_ids}
        for server_id, address in zip(self.server_ids, self.server_addresses, strict=True):
            for alias in self._address_aliases(address):
                aliases.setdefault(alias, [])
                if server_id not in aliases[alias]:
                    aliases[alias].append(server_id)
        return aliases

    def _address_aliases(self, address: str) -> list[str]:
        text = str(address)
        aliases = [text]
        parsed = urlparse(text if "://" in text else f"//{text}")
        if parsed.hostname:
            aliases.append(parsed.hostname.strip("[]"))
            if parsed.port is not None:
                aliases.append(f"{parsed.hostname.strip('[]')}:{parsed.port}")
        return list(dict.fromkeys(aliases))

    # Copied verbatim from verl.experimental.agent_loop.agent_loop.AgentLoopManager
    # (lines 1478-1505) so the delegated generate_sequences() can compute its
    # timing breakdown without having to reach back into verl for it.
    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    # ---- Public protocol parity with AgentLoopManager ----

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Mirror AgentLoopManager.generate_sequences: chunk prompts across workers
        # and concat results. Delegated entirely to AgentLoopWorker — same as verl.
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopManager as _Ref,
        )

        # Reuse the original method by binding self into a thin shim that exposes
        # the attributes _Ref.generate_sequences expects.
        # NOTE: _Ref.generate_sequences is wrapped by @auto_await; unwrap to the
        # underlying async function to keep this call awaitable in our own
        # @auto_await-wrapped method.
        impl = getattr(_Ref.generate_sequences, "__wrapped__", _Ref.generate_sequences)
        return await impl(self, prompts)  # type: ignore[arg-type]

    @auto_await
    async def clear_kv_cache(self) -> None:
        await asyncio.gather(*[r.clear_kv_cache() for r in self.rollout_replicas])

    @auto_await
    async def start_profile(self, **kwargs) -> None:
        await asyncio.gather(*[r.start_profile(**kwargs) for r in self.rollout_replicas])

    @auto_await
    async def stop_profile(self) -> None:
        await asyncio.gather(*[r.stop_profile() for r in self.rollout_replicas])
