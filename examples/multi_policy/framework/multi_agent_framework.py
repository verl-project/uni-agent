"""Multi-agent framework extension for session group rollouts.

This framework subclasses the reference ``GatewayAgentFramework`` and changes
the session scope: instead of one Gateway session per runner execution, each
execution owns a session group with several named role sessions. Each role is
routed to its own policy and generates trajectories independently.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from uni_agent.framework.framework import GatewayAgentFramework, TrajectoryPostprocessor
from uni_agent.gateway.session import Trajectory
from uni_agent.tasks import TaskResult

from .multi_agent_types import SessionGroupHandle

logger = logging.getLogger(__name__)


class MultiAgentGatewayFramework(GatewayAgentFramework):
    """Framework that runs multi-role rollouts over a session group manager.

    Each runner execution creates a session group where every role gets its own
    Gateway session and policy binding. The framework delegates lifecycle calls
    to the manager's group methods and adds role/policy routing metadata to
    TransferQueue records.
    """

    def __init__(
        self,
        *,
        gateway_manager,  # MultiAgentGatewayManager: framework calls create_session/finalize_session/abort_session
        runner_registry,
        role_policy_mapping: Mapping[str, str],
        policy_processors: Mapping[str, Any],
        policy_rollout_configs: Mapping[str, DictConfig] | None = None,
        reward_loop_worker_handles=None,
        custom_reward_function_configured: bool = False,
        rollout_config=None,
        log_dir: str | None = None,
        mask_unfinished_episode: bool = False,
        trajectory_postprocessor: TrajectoryPostprocessor | None = None,
        trajectory_postprocessor_kwargs: dict[str, object] | None = None,
    ):
        """
        Args:
            role_policy_mapping: Maps each role name to its policy name.
            policy_processors: Mapping from policy name to processor; values may be None.
            runner_registry: Named MAS runners selected by the parent framework.
                Each execution owns a whole session group.
        """
        super().__init__(
            gateway_manager=gateway_manager,
            runner_registry=runner_registry,
            reward_loop_worker_handles=reward_loop_worker_handles,
            custom_reward_function_configured=custom_reward_function_configured,
            rollout_config=rollout_config,
            log_dir=log_dir,
            mask_unfinished_episode=mask_unfinished_episode,
            trajectory_postprocessor=trajectory_postprocessor,
            trajectory_postprocessor_kwargs=trajectory_postprocessor_kwargs,
        )
        self._role_policy_mapping = dict(role_policy_mapping)
        self._policy_processors = dict(policy_processors)
        self._policy_rollout_configs = policy_rollout_configs or {}

    @classmethod
    def from_config(
        cls,
        *,
        config,
        gateway_manager,
        processor=None,
        policy_rollout_configs: Mapping[str, DictConfig] | None = None,
        policy_processors: Mapping[str, Any] | None = None,
        reward_loop_worker_handles=None,
    ) -> MultiAgentGatewayFramework:
        """Build a multi-agent framework from config and per-policy mappings.

        ``processor`` is accepted only to preserve the
        :meth:`GatewayAgentFramework.from_config` signature used by the generic
        ``framework_class_fqn`` entry point. It is ignored because this
        framework resolves a processor per policy from ``policy_processors``.

        The per-policy mappings cannot be derived by the generic entry point,
        so callers must build this framework through
        ``examples.multi_policy.framework.multi_agent_entry.build_agent_framework``.
        """
        if policy_rollout_configs is None or policy_processors is None:
            raise ValueError(
                "MultiAgentGatewayFramework requires per-policy configuration; "
                "build it via examples.multi_policy.framework.multi_agent_entry."
                "build_agent_framework instead of the generic framework_class_fqn "
                "entry point"
            )
        if processor is not None:
            logger.debug(
                "Ignoring processor=%r; per-policy processors are used instead",
                processor,
            )
        framework_kwargs = cls._parse_framework_config(config)
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework")
        return cls(
            gateway_manager=gateway_manager,
            role_policy_mapping=dict(af_cfg.role_policy_mapping),
            policy_rollout_configs=policy_rollout_configs,
            reward_loop_worker_handles=reward_loop_worker_handles,
            policy_processors=policy_processors,
            **framework_kwargs,
        )

    def _build_session_sampling_params(
        self,
        *,
        partition_id: str,
        sample_fields: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        """Build session defaults from each policy's composed rollout config."""
        policy_sampling_params: dict[str, dict[str, object]] = {}
        for policy_name, rollout_config in self._policy_rollout_configs.items():
            policy_sampling_params[policy_name] = self._build_sampling_params_from_config(
                config=rollout_config,
                partition_id=partition_id,
                sample_fields=sample_fields,
            )
        return policy_sampling_params

    async def _select_reward_source(
        self,
        session_trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        task_result: TaskResult,
    ) -> tuple[list[tuple[float, dict[str, object]]] | None, str | None]:
        """Prefer the runner's team reward, including zero, over worker scoring."""
        runner_reward_info = {
            "reward": task_result.reward,
            "metrics": ({} if task_result.accuracy is None else {"acc": task_result.accuracy}),
            "reward_context": copy.deepcopy(task_result.extra_info),
        }
        for index, trajectory in enumerate(session_trajectories):
            session_trajectories[index] = replace(
                trajectory,
                extra_fields={
                    **trajectory.extra_fields,
                    "runner_reward_info": copy.deepcopy(runner_reward_info),
                },
            )

        return await super()._select_reward_source(
            session_trajectories,
            sample_fields,
            task_result,
        )

    async def _create_session(
        self,
        session_id: str,
        *,
        metadata: dict[str, object] | None = None,
        sampling_params: object = None,
    ) -> SessionGroupHandle:
        """Create a session group with one session per role."""
        return await self.gateway_manager.create_session(
            session_id,
            role_policy_mapping=self._role_policy_mapping,
            sampling_params=sampling_params,
            metadata=metadata,
        )

    def _resolve_processor(self, trajectory: Trajectory):
        """Resolve processor from trajectory's policy_name."""
        policy_name = trajectory.extra_fields.get("policy_name")
        if not policy_name:
            raise ValueError("Trajectory is missing policy_name in extra_fields")
        try:
            return self._policy_processors[policy_name]
        except KeyError as exc:
            raise ValueError(f"No processor configured for policy {policy_name!r}") from exc

    def _extra_tags(self, trajectory: Trajectory) -> dict[str, object]:
        """Add role, policy, and session_group_id to TransferQueue tags."""
        return {
            "role": trajectory.extra_fields.get("role"),
            "policy_name": trajectory.extra_fields.get("policy_name"),
            "session_group_id": trajectory.extra_fields.get("session_group_id"),
        }
