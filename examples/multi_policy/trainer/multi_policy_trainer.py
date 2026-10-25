from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import ray
import torch
import transfer_queue as tq
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.skip import SkipManager
from verl.utils.tracking import Tracking

from examples.multi_policy.trainer.multi_policy_checkpoint_manager import (
    MultiPolicyCheckpointManager,
    supports_transfer_queue_checkpoint,
)

if TYPE_CHECKING:
    from verl.workers.rollout.llm_server import LLMServerClient

    from examples.multi_policy.framework.multi_agent_entry import (
        MultiPolicyRolloutAdapter,
    )
    from examples.multi_policy.trainer.single_async_policy_trainer import (
        SingleAsyncPPOTrainer,
    )
    from examples.multi_policy.trainer.single_policy_trainer import SinglePPOTrainer

    PolicyTrainer = SinglePPOTrainer | SingleAsyncPPOTrainer


logger = logging.getLogger(__name__)


class MultiPolicyTrainer:
    """PPO orchestration layer for multi-agent blackbox training.

    The class owns per-policy v1 PPO trainer runtimes and drives one shared
    multi-policy rollout adapter for multi-agent rollout collection.
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        # Capture guaranteed outer config values once (verl-native style): the
        # trainer loop and helper methods read these instance attributes instead
        # of re-walking the config tree on every access.
        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(
            self.trainer_mode, {}
        ).get("parameter_sync_step", 1)
        self.train_batch_size = self.config.data.train_batch_size
        self.save_freq = self.config.trainer.save_freq
        self.total_training_steps = self._require_total_training_steps(
            self.config.trainer.total_training_steps
        )

        self.policy_configs = self._compose_policy_configs()
        self.role_policy_mapping = self._resolve_role_policy_mapping()
        self._validate_role_policy_mapping()
        self.checkpoint_manager = MultiPolicyCheckpointManager(
            config=self.config,
            policy_configs=self.policy_configs,
            trainer_mode=self.trainer_mode,
        )

        self.policy_trainers: dict[str, PolicyTrainer] = {}
        # Set before any training phase so cleanup is safe during partial setup.
        self.agent_loop_manager = None
        self.global_steps = 0
        self.timing_raw: dict[str, Any] = {}

        self._create_policy_trainers()
        # Different policies use disjoint GPU sets, so their training phases
        # run concurrently in this pool.
        self._policy_pool = ThreadPoolExecutor(
            max_workers=max(1, len(self.policy_trainers))
        )

    @staticmethod
    def _require_total_training_steps(
        total_training_steps: int | None,
    ) -> int:
        """Return a positive int, or raise a clear error."""
        if (
            isinstance(total_training_steps, bool)
            or not isinstance(total_training_steps, int)
            or total_training_steps <= 0
        ):
            raise ValueError(
                "trainer.total_training_steps must be a positive integer; "
                "epoch-based derivation from trainer.total_epochs is not supported. "
                f"Got {total_training_steps!r} "
                f"({type(total_training_steps).__name__})."
            )
        return total_training_steps

    def _compose_policy_configs(self) -> dict[str, DictConfig]:
        policies = self.config.get("policies")
        # The PPO trainer base is an outer-level runtime choice shared by
        # every policy. Policy entries may customize the composed config
        # through ``ppo_trainer_overrides`` but cannot select a different
        # trainer implementation.
        config_name = self.config.get("ppo_trainer_config_name")
        if not config_name:
            raise ValueError(
                "config.ppo_trainer_config_name is required "
                "(e.g. 'ppo_trainer' or 'ppo_megatron_trainer')"
            )

        resolved = {}
        for policy_key, policy_entry in (
            policies.items() if policies is not None else []
        ):
            policy_name = policy_key
            resolved[policy_name] = self._compose_policy_ppo_config(
                policy_name=policy_name,
                config_name=config_name,
                policy_entry=policy_entry,
            )

        if not resolved:
            raise ValueError("MultiPolicyTrainer requires config.policies")

        # Runtime schedule is applied after composition, before trainer construction.
        optimizer_horizon_steps = (
            self.total_training_steps * self.parameter_sync_step
        )
        for policy_config in resolved.values():
            self._set_policy_optimizer_horizon(
                policy_config, optimizer_horizon_steps
            )
        return resolved

    def _compose_policy_ppo_config(
        self,
        *,
        policy_name: str,
        config_name: str,
        policy_entry: DictConfig,
    ) -> DictConfig:
        # Policy trainer configs come from one outer-selected Hydra config
        # module (e.g. verl's ``verl.trainer.config``). Policy entries cannot
        # replace this source; they only provide config overrides below.
        source = self.config.get("ppo_trainer_config_source") or "verl.trainer.config"
        with initialize_config_module(config_module=source, version_base=None):
            policy_config = compose(config_name=config_name)

        overrides = policy_entry.get("ppo_trainer_overrides")
        if overrides is not None:
            OmegaConf.set_struct(policy_config, False)
            if isinstance(overrides, DictConfig):
                # Resolve policy overrides in the outer Hydra context. This
                # preserves existing ${...} projections (for example,
                # data.train_batch_size) before merging them into the
                # resolved per-policy PPO config.
                overrides = OmegaConf.to_container(overrides, resolve=True)
            policy_config = OmegaConf.merge(policy_config, OmegaConf.create(overrides))

        # ``algorithm``, ``reward``, and ``trainer.v1`` are outer-owned in the
        # multi-agent trainer. Merge (rather than replace) these sections so
        # every policy receives identical shared semantics while verl defaults
        # not declared by the outer config remain available.
        outer_owned_sections = {}
        for section_name in ("algorithm", "reward"):
            section = OmegaConf.select(self.config, section_name, default=None)
            if section is not None:
                outer_owned_sections[section_name] = OmegaConf.to_container(
                    section, resolve=True
                )
        outer_v1 = OmegaConf.select(self.config, "trainer.v1", default=None)
        if outer_v1 is not None:
            outer_owned_sections["trainer"] = {
                "v1": OmegaConf.to_container(outer_v1, resolve=True),
            }
        if outer_owned_sections:
            policy_config = OmegaConf.merge(
                policy_config, OmegaConf.create(outer_owned_sections)
            )

        with open_dict(policy_config):
            policy_config.policy_name = policy_name
        return policy_config

    @staticmethod
    def _set_policy_optimizer_horizon(
        policy_config: DictConfig, optimizer_horizon_steps: int
    ) -> None:
        """Set the scheduler horizon; the outer trainer owns the schedule."""
        with open_dict(policy_config):
            if OmegaConf.select(policy_config, "actor_rollout_ref.actor.optim"):
                policy_config.actor_rollout_ref.actor.optim.total_training_steps = (
                    optimizer_horizon_steps
                )
            if OmegaConf.select(policy_config, "critic.optim"):
                policy_config.critic.optim.total_training_steps = (
                    optimizer_horizon_steps
                )

    def _resolve_role_policy_mapping(self) -> dict[str, str]:
        config_path = (
            "actor_rollout_ref.rollout.custom.agent_framework.role_policy_mapping"
        )
        mapping = OmegaConf.select(self.config, config_path, default=None)
        if mapping is None:
            raise ValueError(f"{config_path} is required")

        mapping = OmegaConf.to_container(mapping, resolve=True)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{config_path} must be a non-empty mapping")

        resolved_mapping: dict[str, str] = {}
        for role, policy_name in mapping.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"{config_path} contains an invalid role: {role!r}")
            if not isinstance(policy_name, str) or not policy_name.strip():
                raise ValueError(
                    f"{config_path}[{role!r}] must be a non-empty policy name"
                )
            resolved_mapping[role] = policy_name
        return resolved_mapping

    def _validate_role_policy_mapping(self) -> None:
        configured_policies = set(self.policy_configs)
        mapped_policies = set(self.role_policy_mapping.values())
        unknown_policies = mapped_policies - configured_policies
        if unknown_policies:
            raise ValueError(
                "role_policy_mapping references unknown policies: "
                f"{sorted(unknown_policies)}. "
                f"Configured policies: {sorted(configured_policies)}"
            )

        unused_policies = configured_policies - mapped_policies
        if unused_policies:
            logger.warning(
                "Configured policies are not referenced by role_policy_mapping: %s",
                sorted(unused_policies),
            )

    def _create_policy_trainers(self) -> dict[str, PolicyTrainer]:
        for policy_name, policy_config in self.policy_configs.items():
            # The outer trainer owns the checkpoint lifecycle (load/save under
            # checkpoints/.../policies/<policy>), so a per-policy trainer must
            # never self-resume; verl's default resume_mode is "auto", which
            # would conflict with the outer ownership.
            with open_dict(policy_config.trainer):
                policy_config.trainer.resume_mode = "disable"

            if self.trainer_mode == "sync":
                from examples.multi_policy.trainer.single_policy_trainer import (
                    SinglePPOTrainer,
                )

                trainer_cls = SinglePPOTrainer
            elif self.trainer_mode == "separate_async":
                from examples.multi_policy.trainer.single_async_policy_trainer import (
                    SingleAsyncPPOTrainer,
                )

                trainer_cls = SingleAsyncPPOTrainer
            else:
                raise ValueError(
                    f"Unsupported trainer.v1.trainer_mode: {self.trainer_mode!r}"
                )
            self.policy_trainers[policy_name] = trainer_cls(config=policy_config)
        return self.policy_trainers

    def init(self) -> None:
        """Initialize all components of the multi-agent trainer.

        1. Initialize every policy's trainer/FSDP runtime concurrently before
           any standalone rollout placement group is created.
        2. Shared outer dataloader: single prompt stream for all policies.
        3. Shared outer replay buffer over the TransferQueue.
        4. Outer checkpoint phase; the outer trainer owns the shared checkpoint
           lifecycle and establishes actor/critic state and ``global_steps``.
        5. Initialize standalone rollout runtimes in descending per-replica
           node GPU footprint waves; policies in the same wave run concurrently.
        6. Invoke every policy's ``on_init_end()`` concurrently after the outer
           initialization phase to synchronize current actor weights to that
           policy's rollout replicas.
        """
        self._validate_stage_resources("training")
        futures = [
            self._policy_pool.submit(policy_trainer.init_training_runtime)
            for policy_trainer in self.policy_trainers.values()
        ]
        wait(futures)
        for future in futures:
            future.result()

        self._build_dataloader()
        self._build_replay_buffer()
        self._load_checkpoint()

        if self.trainer_mode == "separate_async":
            self._validate_stage_resources("rollout")
            for gpus_per_replica_node, policy_names in self._get_rollout_init_waves():
                logger.info(
                    "Initializing standalone rollout wave: "
                    "gpus_per_replica_node=%d policies=%s",
                    gpus_per_replica_node,
                    policy_names,
                )
                futures = [
                    self._policy_pool.submit(
                        self.policy_trainers[
                            policy_name
                        ].init_standalone_rollout_runtime
                    )
                    for policy_name in policy_names
                ]
                wait(futures)
                for future in futures:
                    future.result()

        futures = [
            self._policy_pool.submit(policy_trainer.on_init_end)
            for policy_trainer in self.policy_trainers.values()
        ]
        wait(futures)
        for future in futures:
            future.result()

    def _get_stage_resource_requests(self, stage: str) -> list[tuple[str, str, int]]:
        """Return ``(policy, pool, GPUs-per-node)`` for every stage placement group."""
        if stage not in {"training", "rollout"}:
            raise ValueError(f"Unsupported resource preflight stage: {stage!r}")

        requests: list[tuple[str, str, int]] = []

        def add_requests(
            policy_name: str, pool_name: str, nnodes: Any, gpus_per_node: Any
        ) -> None:
            nnodes = int(nnodes)
            gpus_per_node = int(gpus_per_node)
            if nnodes <= 0 or gpus_per_node <= 0:
                raise ValueError(
                    f"Invalid {stage} resource request for {policy_name}/{pool_name}: "
                    f"nnodes={nnodes}, n_gpus_per_node={gpus_per_node}"
                )
            requests.extend(
                (policy_name, pool_name, gpus_per_node) for _ in range(nnodes)
            )

        for policy_name, policy_config in self.policy_configs.items():
            if stage == "training":
                add_requests(
                    policy_name,
                    "global_pool",
                    OmegaConf.select(policy_config, "trainer.nnodes"),
                    OmegaConf.select(policy_config, "trainer.n_gpus_per_node"),
                )

                if OmegaConf.select(
                    policy_config,
                    "reward.reward_model.enable_resource_pool",
                    default=False,
                ):
                    add_requests(
                        policy_name,
                        "reward_pool",
                        OmegaConf.select(policy_config, "reward.reward_model.nnodes"),
                        OmegaConf.select(
                            policy_config, "reward.reward_model.n_gpus_per_node"
                        ),
                    )

                if OmegaConf.select(
                    policy_config, "distillation.enabled", default=False
                ):
                    add_requests(
                        policy_name,
                        "teacher_pool",
                        OmegaConf.select(policy_config, "distillation.nnodes"),
                        OmegaConf.select(policy_config, "distillation.n_gpus_per_node"),
                    )
                continue

            rollout_config = OmegaConf.select(
                policy_config, "actor_rollout_ref.rollout"
            )
            rollout_nnodes = int(OmegaConf.select(rollout_config, "nnodes"))
            rollout_gpus_per_node = int(
                OmegaConf.select(rollout_config, "n_gpus_per_node")
            )
            total_gpus = rollout_nnodes * rollout_gpus_per_node
            tensor_parallel_size = int(
                OmegaConf.select(
                    rollout_config, "tensor_model_parallel_size", default=1
                )
            )
            data_parallel_size = int(
                OmegaConf.select(rollout_config, "data_parallel_size", default=1)
            )
            pipeline_parallel_size = int(
                OmegaConf.select(
                    rollout_config, "pipeline_model_parallel_size", default=1
                )
            )

            if OmegaConf.select(
                rollout_config, "disaggregation.enabled", default=False
            ):
                decode_parallel_size = OmegaConf.select(
                    rollout_config,
                    "disaggregation.decode_tensor_model_parallel_size",
                    default=None,
                )
                if decode_parallel_size is None:
                    decode_parallel_size = tensor_parallel_size
                rollout_world_size = (
                    (
                        tensor_parallel_size
                        * int(
                            OmegaConf.select(
                                rollout_config, "disaggregation.prefill_replicas"
                            )
                        )
                        + int(decode_parallel_size)
                        * int(
                            OmegaConf.select(
                                rollout_config, "disaggregation.decode_replicas"
                            )
                        )
                    )
                    * data_parallel_size
                    * pipeline_parallel_size
                )
            else:
                rollout_world_size = (
                    tensor_parallel_size * data_parallel_size * pipeline_parallel_size
                )

            if rollout_world_size <= 0 or total_gpus % rollout_world_size != 0:
                raise ValueError(
                    f"Invalid rollout resource request for {policy_name}: total_gpus={total_gpus} "
                    f"must be divisible by rollout_world_size={rollout_world_size}"
                )

            num_replicas = total_gpus // rollout_world_size
            gpus_per_replica_node = min(rollout_gpus_per_node, rollout_world_size)
            if rollout_world_size % gpus_per_replica_node != 0:
                raise ValueError(
                    f"Invalid rollout resource request for {policy_name}: rollout_world_size="
                    f"{rollout_world_size} must be divisible by replica GPUs per node="
                    f"{gpus_per_replica_node}"
                )
            replica_nnodes = rollout_world_size // gpus_per_replica_node
            requests.extend(
                (policy_name, "standalone_rollout", gpus_per_replica_node)
                for _ in range(num_replicas * replica_nnodes)
            )

        return requests

    def _get_rollout_init_waves(self) -> list[tuple[int, list[str]]]:
        """Group policies by replica-node footprint, largest footprint first."""
        policy_footprints: dict[str, int] = {}
        for policy_name, pool_name, gpus_per_node in self._get_stage_resource_requests(
            "rollout"
        ):
            if pool_name != "standalone_rollout":
                continue
            policy_footprints[policy_name] = max(
                policy_footprints.get(policy_name, 0),
                gpus_per_node,
            )

        missing_policies = set(self.policy_trainers) - set(policy_footprints)
        if missing_policies:
            raise ValueError(
                "Missing standalone rollout resource requests for policies: "
                f"{sorted(missing_policies)}"
            )

        policies_by_footprint: dict[int, list[str]] = defaultdict(list)
        for policy_name in self.policy_trainers:
            policies_by_footprint[policy_footprints[policy_name]].append(policy_name)

        return [
            (footprint, policies_by_footprint[footprint])
            for footprint in sorted(policies_by_footprint, reverse=True)
        ]

    def _validate_stage_resources(self, stage: str) -> None:
        """Fail fast when the current Ray snapshot cannot place a whole stage.

        This is a feasibility check, not a reservation: Ray remains the source
        of truth when the policy runtimes create their placement groups.
        """
        requests = self._get_stage_resource_requests(stage)
        node_resources = ray._private.state.available_resources_per_node()
        node_capacities = {
            node_id: int(resources.get("GPU", resources.get("NPU", 0)))
            for node_id, resources in node_resources.items()
        }
        required_gpus = sum(gpus for _, _, gpus in requests)
        available_gpus = sum(node_capacities.values())

        request_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        for request in requests:
            request_counts[request] += 1
        request_summary = ", ".join(
            f"{policy_name}/{pool_name}={count}x{gpus}GPU"
            for (policy_name, pool_name, gpus), count in sorted(request_counts.items())
        )
        node_summary = ", ".join(
            f"{node_id}={gpus}GPU" for node_id, gpus in sorted(node_capacities.items())
        )

        if available_gpus < required_gpus:
            raise ValueError(
                f"Insufficient Ray GPU resources for multi-policy {stage} initialization: "
                f"required {required_gpus}, available {available_gpus}; "
                f"requests=[{request_summary}]; nodes=[{node_summary}]"
            )

        demands = tuple(sorted((gpus for _, _, gpus in requests), reverse=True))
        capacities = tuple(sorted(node_capacities.values(), reverse=True))

        @lru_cache(maxsize=None)
        def can_strict_pack(request_index: int, remaining: tuple[int, ...]) -> bool:
            if request_index == len(demands):
                return True

            demand = demands[request_index]
            tried_capacities = set()
            for node_index, capacity in enumerate(remaining):
                if capacity < demand or capacity in tried_capacities:
                    continue
                tried_capacities.add(capacity)
                next_remaining = list(remaining)
                next_remaining[node_index] -= demand
                next_remaining.sort(reverse=True)
                if can_strict_pack(request_index + 1, tuple(next_remaining)):
                    return True
            return False

        if not can_strict_pack(0, capacities):
            raise ValueError(
                f"Ray GPU topology for multi-policy {stage} initialization cannot satisfy "
                f"STRICT_PACK placement; requests=[{request_summary}]; nodes=[{node_summary}]"
            )

    def get_policy_clients(self) -> dict[str, LLMServerClient]:
        """Return the LLM client for each policy."""
        return {
            policy_name: trainer.get_llm_client()
            for policy_name, trainer in self.policy_trainers.items()
        }

    def get_policy_configs(self) -> dict[str, DictConfig]:
        """Return the composed PPO config for each policy."""
        return dict(self.policy_configs)

    def get_reward_handles(self) -> list[Any] | None:
        """Aggregate reward-worker handles from all policy trainers.

        All policies currently share the same rule-based reward function, so any
        worker can score any trajectory; aggregating lets the framework load-
        balance across all workers instead of only the first policy's. If
        policies ever diverge in reward functions, this must become per-policy
        routing.
        """
        handles: list[Any] = []
        for trainer in self.policy_trainers.values():
            trainer_handles = trainer.get_reward_handles()
            if trainer_handles:
                handles.extend(list(trainer_handles))
        return handles or None

    def fit(self, agent_loop_manager: MultiPolicyRolloutAdapter) -> None:
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        succeeded = False
        try:
            # Native v1 validates the restored policy before advancing to the
            # first training step. Validation is outer-owned because one MAS
            # rollout can produce records for multiple policies.
            if self.config.trainer.get("val_before_train", True):
                self.on_validate_begin()
                try:
                    val_metrics = self._validate()
                finally:
                    self.on_validate_end()
                if not val_metrics:
                    raise RuntimeError("Validation produced no metrics")
                self.logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    succeeded = True
                    return

            # verl semantics: global_steps is 1-based and is advanced before
            # on_train_begin, so warmup batches and update_weights/rollout version
            # tags all use the current step number.
            self.global_steps += 1
            self._sync_policy_runtime_context()
            self._reissue_inflight_prompts()
            self.on_train_begin()
            while self.global_steps <= self.total_training_steps:
                step_metrics = self.train_step()
                is_last_step = self.global_steps >= self.total_training_steps
                test_freq = self.config.trainer.get("test_freq", -1)
                if test_freq > 0 and (
                    is_last_step or self.global_steps % test_freq == 0
                ):
                    self.on_validate_begin()
                    try:
                        step_metrics.update(self._validate())
                    finally:
                        self.on_validate_end()
                # Mirror verl v1 trainer.fit(): record per-step metrics
                # (loss/adv/grad_norm, prefixed per policy) to the configured
                # backend. Sorting keeps the two policies' metrics grouped.
                self.logger.log(
                    data=dict(sorted(step_metrics.items())), step=self.global_steps
                )
                self.global_steps += 1
                self._sync_policy_runtime_context()
            succeeded = True
        finally:
            self.on_train_end()
            self.logger.finish(exit_code=0 if succeeded else 1)

    def train_step(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        self.timing_raw = {}
        with marked_timer("step", self.timing_raw):
            self._sync_policy_runtime_context()
            self.on_step_begin()
            batch = self.step(metrics=metrics, timing_raw=self.timing_raw)
            if self._should_save_checkpoint():
                with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                    self._save_checkpoint()
            self.on_step_end()
            # Collect separate_async weight-sync metrics from the policy trainers.
            metrics.update(self._consume_sync_metrics())
        self._add_data_metrics(batch, metrics, timing_raw=self.timing_raw)
        # Mirror verl v1 trainer.fit(): evict the sampled trajectory records
        # from TransferQueue at the end of each step. ReplayBuffer.sample()
        # only clears the prompt uids, so without this the per-trajectory
        # {uid}_{sample_idx}_{record_idx} records accumulate in SimpleStorage
        # and eventually exceed total_storage_size on long runs.
        if batch is not None:
            tq.kv_clear(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
            )
        metrics["training/global_step"] = self.global_steps
        return dict(metrics)

    def _consume_sync_metrics(self) -> dict[str, Any]:
        """Consume each policy trainer's pending weight-sync metrics.

        separate_async's per-policy ``on_step_end`` stores the standalone
        checkpoint manager's sync metrics (engine-side per-sync stats) in
        ``trainer._pending_sync_metrics``. The outer trainer merges them,
        prefixed per policy, so the console/wandb output stays namespaced.
        """
        metrics: dict[str, Any] = {}
        for policy_name, trainer in self.policy_trainers.items():
            pending = trainer._consume_sync_metrics()
            if not pending:
                continue
            for key, value in pending.items():
                metrics[f"{policy_name}/sync/{key}"] = value
        return metrics

    def step(
        self,
        metrics: dict[str, Any] | None = None,
        timing_raw: dict[str, Any] | None = None,
    ) -> KVBatchMeta:
        metrics = metrics if metrics is not None else {}
        timing_raw = timing_raw if timing_raw is not None else {}
        self.timing_raw = timing_raw
        train_batch_size = self.train_batch_size
        parameter_sync_step = self.parameter_sync_step
        if train_batch_size % parameter_sync_step != 0:
            raise ValueError(
                f"train_batch_size ({train_batch_size}) must be divisible by "
                f"parameter_sync_step ({parameter_sync_step})"
            )
        sample_batch_size = train_batch_size // parameter_sync_step

        self._add_batch_to_generate()

        # Decoupled PPO uses a stable pi_old within one parameter-sync cycle.
        # A local MAS batch may not contain trajectories for every policy, so
        # each policy advances its trigger step only when it participates.
        policy_local_trigger_steps = {
            policy_name: 0 for policy_name in self.policy_trainers
        }
        if parameter_sync_step == 1:
            if self.trainer_mode == "separate_async":
                for policy_name, trainer in self.policy_trainers.items():
                    trainer.local_trigger_step = policy_local_trigger_steps[policy_name]
            return self._step_once(metrics, timing_raw, sample_batch_size)

        metrics_aggregator = MetricsAggregator()
        policy_metrics_aggregators = {
            policy_name: MetricsAggregator() for policy_name in self.policy_trainers
        }
        step_batches = []
        for _ in range(parameter_sync_step):
            if self.trainer_mode == "separate_async":
                for policy_name, trainer in self.policy_trainers.items():
                    trainer.local_trigger_step = policy_local_trigger_steps[policy_name]
            iter_metrics: dict[str, Any] = {}
            batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
            step_batches.append(batch)

            non_padding_mask = np.array(
                [not tag.get("is_padding", False) for tag in batch.tags],
                dtype=bool,
            )
            metrics_aggregator.add_step_metrics(
                {
                    key: value
                    for key, value in iter_metrics.items()
                    if not any(
                        key.startswith(f"{policy_name}/")
                        for policy_name in self.policy_trainers
                    )
                },
                sample_count=int(non_padding_mask.sum()),
            )
            for policy_name, aggregator in policy_metrics_aggregators.items():
                prefix = f"{policy_name}/"
                policy_metrics = {
                    key.removeprefix(prefix): value
                    for key, value in iter_metrics.items()
                    if key.startswith(prefix)
                }
                policy_sample_count = sum(
                    not tag.get("is_padding", False)
                    and tag.get("policy_name") == policy_name
                    for tag in batch.tags
                )
                aggregator.add_step_metrics(
                    policy_metrics, sample_count=policy_sample_count
                )

            updated_policy_names = {
                tag.get("policy_name")
                for tag in batch.tags
                if tag.get("policy_name") in policy_local_trigger_steps
            }
            for policy_name in updated_policy_names:
                policy_local_trigger_steps[policy_name] += 1

        metrics.update(metrics_aggregator.get_aggregated_metrics())
        for policy_name, aggregator in policy_metrics_aggregators.items():
            for key, value in aggregator.get_aggregated_metrics().items():
                metrics[f"{policy_name}/{key}"] = value

        return self._make_batch_like(
            step_batches[0],
            keys=[key for batch in step_batches for key in batch.keys],
            tags=[tag for batch in step_batches for tag in batch.tags],
        )

    def _step_once(
        self,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        sample_batch_size: int,
    ) -> KVBatchMeta:
        self.timing_raw = timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            multi_agent_batch = self.sample_multi_agent_batch(
                sample_batch_size=sample_batch_size,
                metrics=metrics,
            )
        per_policy_batches = self.build_per_policy_batches(multi_agent_batch)
        per_policy_batches = self.prepare_policy_batches_for_ppo_update(
            per_policy_batches, metrics
        )
        with marked_timer("adv", timing_raw, color="brown"):
            multi_agent_batch = self.compute_multi_agent_advantage_from_policy_batches(
                per_policy_batches,
                metrics,
            )
        # Chain the advantage-computed batch into the update, mirroring verl's
        # standard flow (`batch = _compute_advantage(batch); _update_actor(batch)`).
        # Without this, the per-policy update batches would not contain the
        # advantages produced by _compute_advantage.
        per_policy_batches = self.build_per_policy_batches(multi_agent_batch)
        self.update_policy_trainers(per_policy_batches, metrics=metrics)
        return multi_agent_batch

    def sample_multi_agent_batch(
        self,
        sample_batch_size: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> KVBatchMeta:
        self.on_sample_begin()
        result = self.replay_buffer.sample(
            global_steps=self.global_steps,
            partition_id="train",
            batch_size=(
                sample_batch_size
                if sample_batch_size is not None
                else self.train_batch_size
            ),
        )
        self.on_sample_end()

        off_policy_metrics = {}
        if isinstance(result, tuple):
            batch, off_policy_metrics = result
        else:
            batch = result
        if metrics is not None and off_policy_metrics:
            metrics.update(off_policy_metrics)

        return batch

    def build_per_policy_batches(
        self,
        multi_agent_batch: KVBatchMeta,
    ) -> dict[str, KVBatchMeta]:
        if not hasattr(multi_agent_batch, "keys") or not hasattr(
            multi_agent_batch, "tags"
        ):
            raise TypeError(
                "multi_agent_batch must be a KVBatchMeta-like object with keys and tags"
            )
        if len(multi_agent_batch.keys) != len(multi_agent_batch.tags):
            raise ValueError(
                "multi_agent_batch keys and tags must have the same length"
            )

        grouped: dict[str, dict[str, list[Any]]] = {}
        for key, tag in zip(
            multi_agent_batch.keys, multi_agent_batch.tags, strict=True
        ):
            policy_name = tag.get("policy_name")
            if not policy_name:
                raise ValueError(
                    f"Multi-agent trajectory tag for key {key!r} is missing policy_name"
                )
            if policy_name not in self.policy_trainers:
                raise ValueError(
                    f"Unknown policy_name in multi-agent trajectory tag: {policy_name}"
                )
            group = grouped.setdefault(str(policy_name), {"keys": [], "tags": []})
            group["keys"].append(key)
            group["tags"].append(tag)

        return {
            policy_name: self._make_batch_like(
                multi_agent_batch,
                keys=grouped[policy_name]["keys"],
                tags=grouped[policy_name]["tags"],
            )
            for policy_name in self.policy_trainers
            if policy_name in grouped
        }

    def prepare_policy_batches_for_ppo_update(
        self,
        per_policy_batches: Mapping[str, KVBatchMeta],
        metrics: dict[str, Any],
    ) -> dict[str, KVBatchMeta]:
        """Run per-policy balance/log-prob stages concurrently on disjoint GPUs."""
        futures = [
            self._policy_pool.submit(
                self._prepare_policy_batch_for_update, policy_name, batch
            )
            for policy_name, batch in per_policy_batches.items()
        ]
        prepared = {}
        timing_raw: dict[str, float] = {}
        for future in futures:
            policy_name, batch, policy_metrics, per_policy_timing = future.result()
            self._prefix_metrics(metrics, policy_name, policy_metrics)
            timing_raw.update(per_policy_timing)
            prepared[policy_name] = batch
        self.timing_raw.update(timing_raw)
        return prepared

    def _prepare_policy_batch_for_update(
        self,
        policy_name: str,
        batch: KVBatchMeta,
    ) -> tuple[str, KVBatchMeta, dict[str, Any], dict[str, float]]:
        """Run one policy's pre-update stages (balance + old/ref log-probs + values)."""
        trainer = self.policy_trainers[policy_name]
        policy_metrics: dict[str, Any] = {}
        timing_raw: dict[str, float] = {}

        batch = trainer._balance_batch(
            batch,
            metrics=policy_metrics,
            logging_prefix="global_seqlen",
        )

        with marked_timer(f"{policy_name}/old_log_prob", timing_raw, color="blue"):
            batch = trainer._compute_old_log_prob(batch, metrics=policy_metrics)

        if trainer.use_reference_policy:
            with marked_timer(f"{policy_name}/ref_log_prob", timing_raw, color="olive"):
                batch = trainer._compute_ref_log_prob(batch, metrics=policy_metrics)

        if trainer.use_critic:
            with marked_timer(f"{policy_name}/values", timing_raw, color="cyan"):
                batch = trainer._compute_values(batch, metrics=policy_metrics)

        return policy_name, batch, policy_metrics, timing_raw

    def compute_multi_agent_advantage_from_policy_batches(
        self,
        per_policy_batches: Mapping[str, KVBatchMeta],
        metrics: dict[str, Any],
    ) -> KVBatchMeta:
        # Advantage needs the merged rollout/group view. The per-policy batches
        # keep the same TQ keys, so updates can reuse them after advantage is
        # written back to the shared trajectory records.
        multi_agent_batch = self._merge_policy_batches(per_policy_batches)
        return self.compute_multi_agent_advantage(multi_agent_batch, metrics)

    def compute_multi_agent_advantage(
        self,
        multi_agent_batch: KVBatchMeta,
        metrics: dict[str, Any],
    ) -> KVBatchMeta:
        advantage_trainer = next(iter(self.policy_trainers.values()))
        multi_agent_batch = advantage_trainer._compute_advantage(
            multi_agent_batch, metrics=metrics
        )
        self._add_advantage_metrics(multi_agent_batch, metrics)
        return multi_agent_batch

    def _add_advantage_metrics(
        self, batch: KVBatchMeta, metrics: dict[str, Any]
    ) -> None:
        """Log per-policy advantage distribution stats from the TransferQueue.

        verl's per-policy ``_compute_advantage`` writes token-level
        ``advantages``/``returns`` back to the shared trajectory records but
        does not emit any advantage metrics itself. This reads them back (one
        extra TQ fetch per step) and aggregates masked distribution stats per
        policy, so GRPO group-normalized advantages become observable.
        """
        if (
            batch is None
            or not getattr(batch, "keys", None)
            or not getattr(batch, "tags", None)
        ):
            return
        try:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
                select_fields=["advantages", "response_mask"],
            )
            padded = data.to_padded_tensor()
            advantages = padded["advantages"].detach().cpu().float()
            response_mask = padded["response_mask"].to(bool).cpu()
        except Exception as exc:
            logger.warning("failed to compute advantage metrics: %s", exc)
            return

        per_policy: dict[str, list[torch.Tensor]] = {
            policy: [] for policy in self.policy_trainers
        }
        for i, tag in enumerate(batch.tags):
            policy_name = tag.get("policy_name")
            if policy_name not in per_policy:
                continue
            values = advantages[i][response_mask[i]]
            if values.numel() == 0:
                continue
            per_policy[policy_name].append(values)

        for policy_name, tensors in per_policy.items():
            if not tensors:
                continue
            values = torch.cat(tensors)
            stats = {
                "advantages_mean": values.mean().item(),
                # Population std (correction=0): a single value or all-equal
                # advantages yields 0.0 instead of NaN (torch's default
                # sample std with correction=1 returns NaN for n=1).
                "advantages_std": values.std(correction=0).item(),
                "advantages_min": values.min().item(),
                "advantages_max": values.max().item(),
                "advantages_abs_mean": values.abs().mean().item(),
                "advantages_positive_ratio": (values > 0).float().mean().item(),
            }
            for key, value in stats.items():
                metrics[f"{policy_name}/actor/{key}"] = value

    def _add_data_metrics(
        self,
        batch: KVBatchMeta,
        metrics: dict[str, Any],
        timing_raw: dict[str, float] | None = None,
    ) -> None:
        """Mirror verl v1 ``_compute_metrics``' data-metrics portion, per policy.

        Fetches score/reward/advantage/return/length/num-turns fields from the
        TransferQueue and aggregates them through verl's
        ``compute_data_metrics`` (token-level masked, identical semantics to
        native verl), split per policy. Keys are prefixed ``policy_X/``, e.g.
        ``policy_1/critic/score/mean`` / ``policy_1/response_length/mean``.
        Trajectory version-span and staleness metrics use the corresponding
        records' tags, matching verl v1's separate_async definitions.
        """
        if (
            batch is None
            or not getattr(batch, "keys", None)
            or not getattr(batch, "tags", None)
        ):
            return

        non_padding_mask = np.array(
            [not tag.get("is_padding", False) for tag in batch.tags],
            dtype=bool,
        )
        policy_masks = {}
        for policy_name in self.policy_trainers:
            mask = non_padding_mask & np.array(
                [tag.get("policy_name") == policy_name for tag in batch.tags],
                dtype=bool,
            )
            policy_masks[policy_name] = mask

        if all(
            tag.get("is_padding", False)
            or ("min_global_steps" in tag and "max_global_steps" in tag)
            for tag in batch.tags
        ):
            min_global_steps = np.asarray(
                [tag.get("min_global_steps", 0) for tag in batch.tags],
                dtype=np.int64,
            )
            max_global_steps = np.asarray(
                [tag.get("max_global_steps", 0) for tag in batch.tags],
                dtype=np.int64,
            )
            staleness_metric_values = (
                ("trajectory_spans", max_global_steps - min_global_steps + 1),
                ("trajectory_staleness", (self.global_steps - 1) - max_global_steps),
                (
                    "trajectory_staleness_worst",
                    (self.global_steps - 1) - min_global_steps,
                ),
            )
            for metric_prefix, mask in [
                ("training", non_padding_mask),
                *policy_masks.items(),
            ]:
                if not mask.any():
                    continue
                for metric_name, values in staleness_metric_values:
                    selected = values[mask]
                    metrics[f"{metric_prefix}/off_policy/{metric_name}/mean"] = (
                        selected.mean().item()
                    )
                    metrics[f"{metric_prefix}/off_policy/{metric_name}/max"] = (
                        selected.max().item()
                    )
                    metrics[f"{metric_prefix}/off_policy/{metric_name}/min"] = (
                        selected.min().item()
                    )

        try:
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=getattr(batch, "partition_id", "train"),
                select_fields=[
                    "prompts",
                    "responses",
                    "response_mask",
                    "advantages",
                    "returns",
                    "rm_scores",
                    "num_turns",
                ],
            )
            # num_turns is a per-row scalar field; capture it before padding.
            num_turns = np.array(data.pop("num_turns").tolist())
            prompt_length = data["prompts"].offsets().diff()
            response_length = data["responses"].offsets().diff()

            data = data.to_padded_tensor()
            if "token_level_scores" not in data:
                data["token_level_scores"] = data["rm_scores"]
            if "token_level_rewards" not in data:
                data["token_level_rewards"] = data["rm_scores"]
            data["prompt_length"] = prompt_length.float()
            data["response_length"] = response_length.float()
            global_token_num = (prompt_length + response_length).tolist()
            dp = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        except Exception as exc:
            logger.warning("failed to compute data metrics: %s", exc)
            return

        if timing_raw and "step" in timing_raw:
            try:
                metrics.update(compute_timing_metrics(batch=dp, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=dp,
                        timing_raw=timing_raw,
                        n_gpus=self._get_n_gpus_for_throughput(),
                    )
                )
            except Exception as exc:
                logger.warning("failed to compute timing metrics: %s", exc)

        for policy_name in self.policy_trainers:
            mask = policy_masks[policy_name]
            if not mask.any():
                continue
            try:
                sub = dp.select_idxs(mask)
                for key, value in compute_data_metrics(sub, use_critic=False).items():
                    metrics[f"{policy_name}/{key}"] = value
                policy_num_turns = num_turns[mask]
                if policy_num_turns.size > 0:
                    metrics[f"{policy_name}/num_turns/mean"] = (
                        policy_num_turns.mean().item()
                    )
                    metrics[f"{policy_name}/num_turns/max"] = (
                        policy_num_turns.max().item()
                    )
                    metrics[f"{policy_name}/num_turns/min"] = (
                        policy_num_turns.min().item()
                    )
            except Exception as exc:
                logger.warning(
                    "failed to compute data metrics for %s: %s", policy_name, exc
                )

    def update_policy_trainers(
        self,
        per_policy_batches: Mapping[str, KVBatchMeta],
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, KVBatchMeta]:
        """Run per-policy critic/actor updates concurrently on disjoint GPUs."""
        metrics = metrics if metrics is not None else {}
        critic_warmup = self.config.trainer.get("critic_warmup", 0)
        futures = [
            self._policy_pool.submit(
                self._update_one_policy, policy_name, batch, critic_warmup
            )
            for policy_name, batch in per_policy_batches.items()
        ]
        updated = {}
        timing_raw: dict[str, float] = {}
        for future in futures:
            policy_name, batch, policy_metrics, per_policy_timing = future.result()
            self._prefix_metrics(metrics, policy_name, policy_metrics)
            timing_raw.update(per_policy_timing)
            updated[policy_name] = batch
        self.timing_raw.update(timing_raw)
        return updated

    def _get_n_gpus_for_throughput(self) -> int:
        """Return the total GPU count represented by all policy trainers."""
        return sum(
            trainer._get_n_gpus_for_throughput()
            for trainer in self.policy_trainers.values()
        )

    def _update_one_policy(
        self,
        policy_name: str,
        batch: KVBatchMeta,
        critic_warmup: int,
    ) -> tuple[str, KVBatchMeta, dict[str, Any], dict[str, float]]:
        """Run one policy's critic/actor update."""
        trainer = self.policy_trainers[policy_name]
        policy_metrics: dict[str, Any] = {}
        timing_raw: dict[str, float] = {}

        if trainer.use_critic:
            with marked_timer(f"{policy_name}/update_critic", timing_raw, color="pink"):
                batch = trainer._update_critic(batch, metrics=policy_metrics)

        if critic_warmup <= self.global_steps:
            with marked_timer(f"{policy_name}/update_actor", timing_raw, color="red"):
                batch = trainer._update_actor(batch, metrics=policy_metrics)

        return policy_name, batch, policy_metrics, timing_raw

    def _build_dataloader(self) -> None:
        """Build the single shared prompt stream used by multi-agent sampling.

        Mirrors verl PPOTrainer._init_dataloader but runs once at the outer
        level: SinglePPOTrainer skips per-policy dataloader creation, so the
        full dataset is loaded only once instead of once per policy.
        """
        source_trainer = next(iter(self.policy_trainers.values()))
        tokenizer = source_trainer.tokenizer
        processor = source_trainer.processor

        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )
        self.steps_per_epoch = (
            len(self.train_dataset) // self.config.data.train_batch_size
        )

        # Mirror verl: exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups")
        dapo_enabled = bool(
            filter_groups is not None and filter_groups.get("enable", False)
        )
        sync_refill_failed_groups = bool(self.config.sampler.sync_refill_failed_groups)
        trainer_mode = self.trainer_mode
        requires_exact_refill = (
            dapo_enabled or sync_refill_failed_groups or trainer_mode != "sync"
        )
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size")
            if user_gen_batch_size not in (None, 1):
                logger.warning(
                    f"data.gen_batch_size={user_gen_batch_size} is overridden to 1."
                )
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config.data):
                self.config.data.gen_batch_size = 1

        gen_batch_size = self.config.data.get("gen_batch_size") or self.train_batch_size
        dataloader_num_workers = self.config.data.dataloader_num_workers
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=dataloader_num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=dataloader_num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            "DataLoader: steps_per_epoch=%d, total_training_steps=%d",
            self.steps_per_epoch,
            self.total_training_steps,
        )

    def _build_replay_buffer(self) -> ReplayBuffer | ReplayBufferAsync:
        """Build the outer shared ReplayBuffer used by multi-agent sampling.

        Mirrors verl PPOTrainer._build_replay_buffer but sources every knob
        from the outer config (the outer sampler/algorithm sections are the
        single source of truth; per-policy sampler configs are not used).
        refill_fn is bound directly to the outer trainer, so DAPO group
        filtering / sync_refill_failed_groups refills never route through a
        per-policy trainer that lacks an agent_loop_manager.
        """
        trainer_mode = self.trainer_mode
        buffer_cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync

        max_off_policy_threshold = self.config.sampler.max_off_policy_threshold
        max_off_policy_strategy = self.config.sampler.max_off_policy_strategy
        sampler_kwargs = self.config.sampler.sampler_kwargs
        sync_refill_failed_groups = bool(self.config.sampler.sync_refill_failed_groups)
        filter_groups_metric = self._resolve_filter_groups_metric()
        train_batch_size = self.train_batch_size
        gen_batch_size = (
            1
            if filter_groups_metric is not None
            or sync_refill_failed_groups
            or trainer_mode != "sync"
            else self.config.data.get("gen_batch_size") or train_batch_size
        )
        max_inflight_gen_batches = 1
        if filter_groups_metric is not None:
            filter_groups = self.config.algorithm.get("filter_groups")
            max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)

        self.replay_buffer = buffer_cls(
            trainer_mode=trainer_mode,
            trainer_config=OmegaConf.create({}),
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
            filter_groups_metric=filter_groups_metric,
            sync_refill_failed_groups=sync_refill_failed_groups,
            train_batch_size=train_batch_size,
            gen_batch_size=int(gen_batch_size),
            max_inflight_gen_batches=int(max_inflight_gen_batches),
        )
        return self.replay_buffer

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling.

        Mirrors verl PPOTrainer._resolve_filter_groups_metric, reading the
        outer config.algorithm.filter_groups (inherited by every policy).
        """
        filter_groups = self.config.algorithm.get("filter_groups")
        filter_enabled = bool(
            filter_groups is not None and filter_groups.get("enable", False)
        )
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric")
        if not filter_metric:
            raise ValueError(
                "algorithm.filter_groups.metric must be set when group filtering is enabled"
            )

        reward_model = self.config.reward.get("reward_model")
        streaming_reward_path = reward_model is None or (
            not reward_model.get("enable", False)
            or reward_model.get("enable_resource_pool", False)
        )
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def _load_checkpoint(self) -> None:
        global_step = self.checkpoint_manager.load(
            policy_trainers=self.policy_trainers,
            train_dataloader=self.train_dataloader,
        )
        if global_step is None:
            self.global_steps = 0
            return
        self.global_steps = global_step
        self._sync_policy_runtime_context()

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Re-dispatch restored pending/running prompt groups after resume."""
        if self.trainer_mode == "sync" or not supports_transfer_queue_checkpoint():
            return 0

        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False)
            and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0
        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        trajectory_prefixes = tuple(f"{uid}_" for uid in inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.startswith(trajectory_prefixes)
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [
            {"is_prompt": True, "status": "pending", "global_steps": self.global_steps}
            for _ in inflight_uids
        ]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        logger.info(
            "Re-issued %d in-flight prompts at step %d and cleared %d partial trajectories",
            len(inflight_uids),
            self.global_steps,
            len(old_trajectory_keys),
        )
        return len(inflight_uids)

    def _save_checkpoint(self) -> None:
        self.checkpoint_manager.save(
            global_step=self.global_steps,
            policy_trainers=self.policy_trainers,
            train_dataloader=self.train_dataloader,
            policy_pool=self._policy_pool,
        )

    def _should_save_checkpoint(self) -> bool:
        save_freq = self.save_freq
        if save_freq <= 0:
            return False
        return (
            self.global_steps >= self.total_training_steps
            or self.global_steps % save_freq == 0
        )

    def _fetch_one_gen_batch(self) -> TensorDict:
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array(
            [str(uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
        )
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> TensorDict:
        """Fetch and coalesce the requested number of prompts.

        Mirrors verl's v1 trainer semantics: ``num_prompts`` must be a positive
        multiple of ``data.gen_batch_size`` (defaults to ``data.train_batch_size``),
        and is submitted in whole gen-batch dataloader fetches.
        """
        train_batch_size = self.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        # Read the dataloader's actual fetch granularity instead of the outer
        # config default: verl forces per-policy data.gen_batch_size=1 when
        # DAPO group filtering / sync_refill_failed_groups is enabled, so the
        # outer fetch loop must use the same granularity as the shared
        # dataloader (1 sample per fetch) rather than train_batch_size.
        gen_batch_size = int(
            getattr(self.train_dataloader, "batch_size", None) or train_batch_size
        )
        if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
                f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
            )

        chunks = [
            self._fetch_one_gen_batch() for _ in range(num_prompts // gen_batch_size)
        ]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the agent loop manager."""
        batch = self._next_train_batch(num_prompts)
        if batch is None:
            return 0
        return self._submit_batch_to_rollout(batch)

    def _add_batch_to_generate(self) -> None:
        batch = self._next_train_batch()
        if batch is None:
            return
        if len(batch) == 0:
            return
        self._submit_batch_to_rollout(batch)

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Register prompts in TransferQueue and dispatch them for generation.

        This is the training-only submission path, matching native v1's
        ``PPOTrainer._submit_batch_to_rollout`` contract. Validation uses its
        own explicit ``val`` registration in ``_validate`` below.

        Returns the number of submitted prompts (refill_fn contract).
        """
        if batch is None:
            return 0

        uid_values = tu.get(batch, "uid")
        if uid_values is None:
            raise ValueError(
                "MultiPolicyTrainer requires batch['uid'] before rollout submission"
            )
        uid_values = (
            uid_values.tolist() if hasattr(uid_values, "tolist") else list(uid_values)
        )
        if not uid_values:
            return 0

        tags = [
            {
                "is_prompt": True,
                "status": "pending",
                "global_steps": self.global_steps,
            }
            for _ in uid_values
        ]
        put_kwargs = {
            "keys": [str(uid) for uid in uid_values],
            "partition_id": "train",
            "tags": tags,
        }
        # Mirror verl: async trainers persist prompt fields in TQ so in-flight
        # prompts can be re-issued after a checkpoint resume.
        trainer_mode = self.trainer_mode
        if trainer_mode != "sync":
            fields = batch.select(
                *[
                    key
                    for key in batch.keys()
                    if not isinstance(batch.get(key), NonTensorData)
                ]
            )
            put_kwargs["fields"] = fields
        tq.kv_batch_put(**put_kwargs)
        self.agent_loop_manager.generate_sequences(batch)
        return len(uid_values)

    @staticmethod
    def _validation_final_record_keys(batch: KVBatchMeta) -> list[str]:
        """Return the final record for each complete MAS rollout.

        A single ``{uid, sample_idx}`` is one MAS rollout and may contain one
        trajectory per agent. The final record is the only record counted as a
        validation sample; retaining the other keys lets callers clean up all
        records from TransferQueue.
        """
        final: dict[tuple[str, int], tuple[int, int, str]] = {}
        for position, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
            key_parts = str(key).rsplit("_", 2)
            uid = str(tag.get("uid") or key_parts[0])
            try:
                sample_value = tag.get("sample_idx")
                sample_idx = int(
                    sample_value if sample_value is not None else key_parts[1]
                )
            except (IndexError, TypeError, ValueError):
                sample_idx = 0
            try:
                record_value = tag.get("record_idx")
                record_idx = int(
                    record_value if record_value is not None else key_parts[2]
                )
            except (IndexError, TypeError, ValueError):
                record_idx = position
            group_key = (uid, sample_idx)
            previous = final.get(group_key)
            if previous is None or record_idx > previous[0]:
                final[group_key] = (record_idx, position, str(key))
        return [item[2] for item in sorted(final.values(), key=lambda item: item[1])]

    @staticmethod
    def _validation_values(data, key: str) -> list[Any]:
        value = data.get(key) if hasattr(data, "get") else None
        if value is None:
            return []
        if isinstance(value, NonTensorData):
            return [value.data]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [
                item.data if isinstance(item, NonTensorData) else item for item in value
            ]
        return [value]

    def _validate(self) -> dict[str, Any]:
        """Run validation entirely at the multi-agent orchestration layer."""
        if getattr(self, "val_dataloader", None) is None:
            return {}

        data_sources: list[str] = []
        sample_uids: list[str] = []
        sample_turns: list[float] = []
        reward_values: list[float] = []
        reward_extra_infos: dict[str, list[Any]] = defaultdict(list)
        timing_raw: dict[str, Any] = {}
        self.timing_raw = timing_raw

        for batch_dict in self.val_dataloader:
            batch_dict = dict(batch_dict)
            raw_prompts = batch_dict.get("raw_prompt")
            if raw_prompts is None:
                raise ValueError("Validation batch must contain raw_prompt")
            batch_dict["uid"] = np.array(
                [str(uuid4()) for _ in range(len(raw_prompts))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Match native v1's separate validation path: register validation
            # prompt markers directly in the ``val`` TQ partition instead of
            # widening the train-only submission helper with a partition arg.
            tags = [
                {
                    "is_prompt": True,
                    "status": "pending",
                    "global_steps": self.global_steps,
                }
                for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)
            sampled = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="val",
                batch_size=len(batch),
            )

            val_batch, _ = sampled if isinstance(sampled, tuple) else (sampled, {})
            if not val_batch.keys:
                raise RuntimeError("Validation replay buffer returned no trajectories")
            try:
                final_keys = self._validation_final_record_keys(val_batch)
                fields = tq.kv_batch_get(
                    keys=final_keys,
                    partition_id="val",
                    select_fields=[
                        "uid",
                        "rm_scores",
                        "num_turns",
                        "data_source",
                        "extra_fields",
                    ],
                )
                uid_values = self._validation_values(fields, "uid")
                score_values = fields["rm_scores"].sum(dim=1).tolist()
                turn_values = self._validation_values(fields, "num_turns")
                source_values = self._validation_values(fields, "data_source")
                extra_values = self._validation_values(fields, "extra_fields")
                if len(score_values) != len(final_keys):
                    raise RuntimeError("Validation trajectories are missing rm_scores")
                for index, score in enumerate(score_values):
                    reward = float(score)
                    reward_values.append(reward)
                    sample_uids.append(
                        str(uid_values[index])
                        if index < len(uid_values)
                        else final_keys[index]
                    )
                    sample_turns.append(
                        float(turn_values[index]) if index < len(turn_values) else 0.0
                    )
                    data_sources.append(
                        str(source_values[index])
                        if index < len(source_values)
                        else "unknown"
                    )
                    extra = extra_values[index] if index < len(extra_values) else {}
                    extra = getattr(extra, "data", extra)
                    current_extra = (
                        extra.get("reward_extra_info", {})
                        if isinstance(extra, dict)
                        else {}
                    )
                    sample_position = len(reward_extra_infos["reward"])
                    for key in (
                        reward_extra_infos.keys() - {"reward"} - current_extra.keys()
                    ):
                        reward_extra_infos[key].append(None)
                    for key, value in current_extra.items():
                        if key not in reward_extra_infos:
                            reward_extra_infos[key] = [None] * sample_position
                        reward_extra_infos[key].append(value)
                    reward_extra_infos["reward"].append(reward)
            finally:
                tq.kv_clear(keys=val_batch.keys, partition_id="val")

        if not reward_values:
            return {}
        metrics = {}
        try:
            metrics.update(
                self._format_validation_metrics(
                    data_sources, sample_uids, reward_extra_infos, sample_turns
                )
            )
        except (ImportError, AttributeError):
            metrics = {}
        if not metrics:
            metrics["validation/reward/mean"] = float(np.mean(reward_values))
            metrics["validation/reward/min"] = float(np.min(reward_values))
            metrics["validation/reward/max"] = float(np.max(reward_values))
        metrics["val-aux/num_turns/mean"] = float(np.mean(sample_turns))
        metrics["val-aux/num_turns/min"] = float(np.min(sample_turns))
        metrics["val-aux/num_turns/max"] = float(np.max(sample_turns))
        return metrics

    @staticmethod
    def _format_validation_metrics(
        data_sources, sample_uids, reward_extra_infos, sample_turns
    ):
        structured = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos
        )
        metrics = {}
        for data_source, variables in structured.items():
            for variable, values in variables.items():
                core_variable = "acc" if "acc" in variables else "reward"
                n_max = max(int(name.split("@")[-1].split("/")[0]) for name in values)
                for name, value in values.items():
                    is_core = (
                        variable == core_variable
                        and name.startswith(("mean", "maj", "best"))
                        and f"@{n_max}" in name
                    )
                    section = "val-core" if is_core else "val-aux"
                    metrics[f"{section}/{data_source}/{variable}/{name}"] = value
        return metrics

    def _sync_policy_runtime_context(self) -> None:
        for trainer in self.policy_trainers.values():
            trainer.global_steps = self.global_steps
            # Sync hooks share the outer timing context. Separate-async step
            # hooks receive private contexts in on_step_end before dispatch.
            if self.trainer_mode == "sync":
                trainer.timing_raw = self.timing_raw

    def on_train_begin(self) -> None:
        if self.config.get("skip") is not None:
            SkipManager.init(self.config)
        if self.trainer_mode == "sync":
            return
        # PPOTrainer.fit() seeds the per-policy rollout profiling state before
        # entering the training loop. The outer trainer replaces fit(), so
        # mirror that initialization here: the delegated on_step_end() reads
        # prev_step_profile on every policy runtime.
        for policy_trainer in self.policy_trainers.values():
            policy_trainer.prev_step_profile = False
            policy_trainer.curr_step_profile = False
            policy_trainer.next_step_profile = False
        num_warmup_batches = self.config.trainer.v1.separate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()

    def on_train_end(self) -> None:
        # Shut down dataloader workers while the interpreter is still alive.
        # torch's DataLoader registers worker pids with a SIGCHLD handler; if the
        # iterator is never garbage-collected (StatefulDataLoader keeps its own
        # ``_iterator`` reference), ``_shutdown_workers`` never runs and workers
        # remain registered. When the interpreter exits, their teardown SIGKILL
        # surfaces as ``RuntimeError: DataLoader worker ... killed by signal:
        # Killed`` from the atexit path, making a successful run exit non-zero.
        self._close_dataloader()

    def on_validate_begin(self) -> None:
        """Prepare every policy runtime for outer multi-agent validation."""
        self._sync_policy_runtime_context()
        futures = [
            self._policy_pool.submit(policy_trainer.on_validate_begin)
            for policy_trainer in self.policy_trainers.values()
        ]
        for future in futures:
            future.result()

    def on_validate_end(self) -> None:
        """Finish every policy's native validation lifecycle."""
        futures = [
            self._policy_pool.submit(policy_trainer.on_validate_end)
            for policy_trainer in self.policy_trainers.values()
        ]
        for future in futures:
            future.result()

    def _close_dataloader(self) -> None:
        """Terminate dataloader workers cleanly and drop iterator references.

        Safe to call multiple times (idempotent): after the first call the
        attributes are None and torch's ``_shutdown_workers`` is itself guarded
        by an internal ``_shutdown`` flag.
        """
        for attr in ("train_dataloader_it", "val_dataloader_it"):
            it = getattr(self, attr, None)
            if it is None:
                continue
            try:
                shutdown = getattr(it, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
            except Exception:
                logger.warning("failed to shut down %s", attr, exc_info=True)
            finally:
                setattr(self, attr, None)

        for attr in ("train_dataloader", "val_dataloader"):
            dl = getattr(self, attr, None)
            if dl is None:
                continue
            try:
                # StatefulDataLoader keeps ``self._iterator`` alive, so the
                # iterator's ``__del__`` (which calls ``_shutdown_workers``)
                # would otherwise never run. Shut it down and drop the ref.
                internal_it = getattr(dl, "_iterator", None)
                if internal_it is not None:
                    shutdown = getattr(internal_it, "_shutdown_workers", None)
                    if callable(shutdown):
                        shutdown()
                    dl._iterator = None
            except Exception:
                logger.warning("failed to shut down %s", attr, exc_info=True)
            finally:
                setattr(self, attr, None)

    def _dispatch_policy_hook(
        self, hook_name: str, *, collect_timing: bool = False
    ) -> None:
        """Run one lifecycle hook on every policy trainer concurrently.

        Args:
            hook_name: Name of the lifecycle hook to invoke on each policy
                trainer (for example ``on_step_begin``).
            collect_timing: Replace each trainer's ``timing_raw`` with a fresh
                dict for the call, then merge the collected entries into the
                outer ``timing_raw`` under a ``{policy_name}/`` prefix.
        """
        futures: list[tuple[str, Future[Any]]] = []
        policy_timings: dict[str, dict[str, Any]] = {}
        for policy_name, policy_trainer in self.policy_trainers.items():
            if collect_timing:
                policy_timing: dict[str, Any] = {}
                policy_trainer.timing_raw = policy_timing
                policy_timings[policy_name] = policy_timing
            futures.append(
                (
                    policy_name,
                    self._policy_pool.submit(getattr(policy_trainer, hook_name)),
                )
            )
        for policy_name, future in futures:
            future.result()
            if collect_timing:
                for key, value in policy_timings[policy_name].items():
                    self.timing_raw[f"{policy_name}/{key}"] = value

    def on_step_begin(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            return
        # PPOTrainerSeparateAsync.on_step_begin() resets the sampling wait
        # bookkeeping and, with enable_switch=False, switches hybrid replicas
        # back to trainer mode (freeing their GPUs and stopping generation
        # from stale weights). Skipping it leaves the hybrid pool routable.
        self._dispatch_policy_hook("on_step_begin", collect_timing=True)

    def on_step_end(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            futures = [
                self._policy_pool.submit(self._update_weights_one_policy, policy_name)
                for policy_name in self.policy_trainers
            ]
            timing_raw: dict[str, Any] = {}
            for future in futures:
                _, per_policy_timing = future.result()
                timing_raw.update(per_policy_timing)
            self.timing_raw.update(timing_raw)
            return
        # separate_async: delegate to verl's per-policy on_step_end (standalone
        # checkpoint manager; update_weights contains abort/resume internally).
        self._dispatch_policy_hook("on_step_end", collect_timing=True)

    def _update_weights_one_policy(
        self, policy_name: str
    ) -> tuple[None, dict[str, Any]]:
        """Wake one policy's vLLM replicas with the current weights."""
        timing_raw: dict[str, Any] = {}
        trainer = self.policy_trainers[policy_name]
        with marked_timer(f"{policy_name}/update_weights", timing_raw, color="red"):
            trainer.checkpoint_manager.update_weights(self.global_steps)
        return None, timing_raw

    def on_sample_begin(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            return
        # Pairs with the delegated on_sample_end(): records _sample_start.
        self._dispatch_policy_hook("on_sample_begin")

    def on_sample_end(self) -> None:
        self._sync_policy_runtime_context()
        if self.trainer_mode == "sync":
            for policy_name, trainer in self.policy_trainers.items():
                with marked_timer(
                    f"{policy_name}/sleep_replicas", self.timing_raw, color="red"
                ):
                    trainer.checkpoint_manager.sleep_replicas()
            return
        # separate_async: delegate to verl's per-policy on_sample_end. The
        # matching on_sample_begin is dispatched by the outer trainer, which is
        # what initializes _sample_start on each policy runtime.
        self._dispatch_policy_hook("on_sample_end")

    @staticmethod
    def _prefix_metrics(
        metrics: dict[str, Any], policy_name: str, policy_metrics: dict[str, Any]
    ) -> None:
        for key, value in policy_metrics.items():
            metrics[f"{policy_name}/{key}"] = value

    @staticmethod
    def _make_batch_like(
        template: KVBatchMeta,
        *,
        keys: list[str],
        tags: list[dict[str, Any]],
    ) -> KVBatchMeta:
        kwargs = {
            "partition_id": template.partition_id,
            "keys": list(keys),
            "tags": [dict(tag) for tag in tags],
            "fields": template.fields,
            "extra_info": dict(template.extra_info or {}),
        }
        return template.__class__(**kwargs)

    def _merge_policy_batches(
        self,
        per_policy_batches: Mapping[str, KVBatchMeta],
    ) -> KVBatchMeta:
        merged_keys: list[str] = []
        merged_tags: list[dict[str, Any]] = []
        template = None
        for policy_name in self.policy_trainers:
            batch = per_policy_batches.get(policy_name)
            if batch is None:
                continue
            if template is None:
                template = batch
            merged_keys.extend(list(batch.keys))
            merged_tags.extend([dict(tag) for tag in batch.tags])
        if template is None:
            raise ValueError("Cannot merge an empty per-policy batch mapping")
        return self._make_batch_like(template, keys=merged_keys, tags=merged_tags)

__all__ = ["MultiPolicyTrainer"]
