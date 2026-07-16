from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol
from uuid import uuid4

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from uni_agent.gateway.session import SessionHandle, Trajectory
from verl.tools.tool_registry import initialize_tools_from_config
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.transferqueue_utils import tq

from .base import AgentFramework
from .multi_modal_postprocess import compute_multi_modal_inputs, compute_position_ids

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    """Callable contract for OpenAI-compatible agent runners."""

    async def __call__(
        self,
        *,
        session: SessionHandle,
        raw_prompt: object,
        sample_index: int,
        **sample_runner_kwargs: object,
    ) -> None: ...


@dataclass
class _RunnerConfig:
    runner_fqn: str
    runner_kwargs: dict[str, object]
    dispatch_mode: str
    max_concurrent_sessions: int

    def __post_init__(self) -> None:
        if not self.runner_fqn:
            raise ValueError("runner_fqn is required")
        if self.dispatch_mode not in {"inline_async", "ray_task"}:
            raise ValueError(f"Unknown dispatch mode: {self.dispatch_mode}")
        if self.max_concurrent_sessions < 0:
            raise ValueError(f"max_concurrent_sessions must be non-negative, got {self.max_concurrent_sessions}")

    @classmethod
    def from_config(cls, runner_name: object, runner_cfg) -> _RunnerConfig:
        runner_fqn = runner_cfg.get("runner_fqn")
        runner_kwargs = dict(
            OmegaConf.to_container(OmegaConf.create(runner_cfg.get("runner_kwargs", {})), resolve=True) or {}
        )
        tool_config_path = runner_cfg.get("tool_config_path")
        if tool_config_path:
            tool_config = initialize_tools_from_config(str(tool_config_path))
            if not tool_config:
                raise ValueError(
                    f"agent_runners.{runner_name}.tool_config_path did not initialize any tools: {tool_config_path}"
                )
            runner_kwargs["tool_config"] = tool_config
        dispatch_mode = str(runner_cfg.get("dispatch_mode", "inline_async"))
        max_concurrent_sessions = int(runner_cfg.get("max_concurrent_sessions", 0) or 0)
        try:
            return cls(
                runner_fqn="" if runner_fqn is None else str(runner_fqn),
                runner_kwargs=runner_kwargs,
                dispatch_mode=dispatch_mode,
                max_concurrent_sessions=max_concurrent_sessions,
            )
        except ValueError as exc:
            raise ValueError(f"agent_runners.{runner_name}: {exc}") from exc


def _materialize_runner(runner_fqn: str, runner_kwargs: dict[str, object]):
    runner = load_class_from_fqn(runner_fqn, description="agent runner")
    if isinstance(runner, type):
        return runner(**runner_kwargs)
    if runner_kwargs:
        return partial(runner, **runner_kwargs)
    return runner


@ray.remote
def _run_agent_runner_ray_task(
    *,
    runner_fqn: str,
    runner_kwargs: dict[str, object],
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: object | None,
) -> None:
    """Run only the user runner in Ray; parent owns session lifecycle outputs."""
    runner = _materialize_runner(runner_fqn, runner_kwargs)
    asyncio.run(
        runner(
            raw_prompt=raw_prompt,
            session=session,
            sample_index=sample_index,
            **({"tools_kwargs": tools_kwargs} if tools_kwargs is not None else {}),
        )
    )


def _short_failure_reason(error: BaseException) -> str:
    message = str(error)
    if not message:
        message = error.__class__.__name__
    return message[:512]


_TQ_NESTED_SEQUENCE_FIELDS = {
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
    "teacher_logprobs",
    "teacher_ids",
}


def _list_of_tq_fields_to_tensordict(fields: list[dict[str, object]]) -> TensorDict:
    td = tu.list_of_dict_to_tensordict(fields)
    for key in _TQ_NESTED_SEQUENCE_FIELDS:
        if key not in fields[0]:
            continue
        values = [field[key] for field in fields]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        ragged_idx = 2 if key == "position_ids" and values[0].dim() == 2 else None
        td[key] = tu.nested_tensor_from_tensor_list(values, ragged_idx=ragged_idx)
    return td


def _trajectory_to_reward_dataproto(trajectory, sample_fields):
    """Build a single-sample DataProto for RewardLoopWorker.compute_score.

    Field shape matches AgentLoopWorker._compute_score
    (verl/experimental/agent_loop/agent_loop.py:753-772). Only fields actually
    consumed by NaiveRewardManager.run_single / RewardLoopWorker dispatch are
    populated; tool_extra_fields / num_turns are passed via non_tensor_batch
    for parity.
    """
    import numpy as np

    from verl.protocol import DataProto

    prompt_ids = torch.tensor(trajectory.prompt_ids, dtype=torch.long).unsqueeze(0)
    response_ids = torch.tensor(trajectory.response_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )

    non_tensor_batch: dict[str, object] = {}
    for key in ("raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
        if key in sample_fields:
            non_tensor_batch[key] = np.array([sample_fields[key]], dtype=object)
    non_tensor_batch["__num_turns__"] = np.array([trajectory.num_turns])

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class OpenAICompatibleAgentFramework(AgentFramework):
    """Reference AgentFramework implementation for OpenAI-compatible agent loops.

    Each sample in the batch is run as an independent session: the agent
    communicates with the Gateway via standard ``/v1/chat/completions``
    requests, and the Gateway collects token-level trajectories.  After
    finalization, ``_score_trajectories`` dispatches the session's final
    trajectory to a RewardLoopWorker and broadcasts the score back to all
    trajectories in the session (matching
    ``AgentLoopWorkerTQ._agent_loop_postprocess``); the framework then writes
    them to the TransferQueue schema consumed by sync training.
    """

    def __init__(
        self,
        gateway_manager,  # GatewayManager: framework calls create_session/finalize_session/abort_session
        *,
        runner_registry: dict[str, _RunnerConfig],
        reward_loop_worker_handles=None,
        processor=None,
        rollout_config=None,
    ):
        self.gateway_manager = gateway_manager
        self.runner_registry = runner_registry
        # Materialize inline runners at construction since they run in-process and may maintain state;
        # ray_task runners are materialized per-run since they run remotely.
        self._inline_runners = {
            runner_name: _materialize_runner(runner_config.runner_fqn, runner_config.runner_kwargs)
            for runner_name, runner_config in runner_registry.items()
            if runner_config.dispatch_mode == "inline_async"
        }
        self.reward_loop_worker_handles = list(reward_loop_worker_handles) if reward_loop_worker_handles else None
        self._processor = processor
        self._rollout_config = rollout_config
        self._runner_semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_config(
        cls,
        *,
        config,
        gateway_manager,
        processor=None,
        reward_loop_worker_handles=None,
    ) -> OpenAICompatibleAgentFramework:
        # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
        af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        runner_registry: dict[str, _RunnerConfig] = {}
        agent_runners_cfg = af_cfg.get("agent_runners")
        if not agent_runners_cfg:
            raise ValueError("actor_rollout_ref.rollout.custom.agent_framework.agent_runners is required")

        for runner_name, runner_cfg in agent_runners_cfg.items():
            runner_registry[str(runner_name)] = _RunnerConfig.from_config(runner_name, runner_cfg)

        return cls(
            gateway_manager=gateway_manager,
            runner_registry=runner_registry,
            reward_loop_worker_handles=reward_loop_worker_handles,
            processor=processor,
            rollout_config=config.actor_rollout_ref.rollout,
        )

    def _build_session_sampling_params(
        self,
        *,
        partition_id: str,
        sample_fields: dict[str, object],
    ) -> dict[str, object]:
        """Build trusted per-session sampling defaults using VERL rollout semantics."""
        config = self._rollout_config
        sampling_params: dict[str, object] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if partition_id == "val":
            sampling_params.update(
                temperature=config.val_kwargs.temperature,
                top_p=config.val_kwargs.top_p,
                top_k=config.val_kwargs.top_k,
            )
        elif "__do_sample__" in sample_fields and not bool(sample_fields["__do_sample__"]):
            sampling_params.update(temperature=0, top_p=1.0, top_k=-1)
        return sampling_params

    async def generate_sequences(self, prompts: TensorDict) -> None:
        """Run rollout-manager generation and write outputs into TransferQueue."""
        if self._rollout_config is None:
            raise RuntimeError("OpenAICompatibleAgentFramework requires rollout_config for generate_sequences")

        global_steps = tu.get(prompts, "global_steps")
        if global_steps is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['global_steps']")

        partition_id = "val" if "validate" in prompts.keys() else "train"
        if partition_id == "val":
            val_kwargs = self._rollout_config.get("val_kwargs", {})
            num_sessions = int(val_kwargs.get("n"))
        else:
            num_sessions = int(self._rollout_config.get("n"))

        uids = tu.get(prompts, "uid")
        if uids is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['uid'] for TransferQueue output")

        stats = await self._run_batch_to_tq(
            prompts,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        logger.info(
            "generate_sequences summary: num_input_prompts=%s num_success_sessions=%s "
            "num_failed_sessions=%s num_success_outputs=%s num_failed_uids=%s failure_reasons=%s",
            stats["num_input_prompts"],
            stats["num_success_sessions"],
            stats["num_failed_sessions"],
            stats["num_success_outputs"],
            stats["num_failed_uids"],
            stats["failure_reasons"][:3],
        )
        if stats["num_success_outputs"] == 0:
            raise RuntimeError(
                f"All rollouts failed at global_steps={global_steps}. "
                f"failures={stats['num_failed_uids']}/{stats['num_input_prompts']}"
            )
        return None

    async def _run_batch_to_tq(
        self,
        prompts: TensorDict,
        *,
        global_steps: int,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict:
        """Run all prompts in a batch and aggregate prompt/session stats."""
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")

        # Batch layer: each sample/prompt owns its own group of rollout.n sessions.
        # Prompt tasks are isolated so one prompt failure does not drop the whole batch.
        tasks = []
        for sample_index in range(len(prompts)):
            tasks.append(
                self._run_prompt_sessions_to_tq(
                    sample_fields=self._extract_sample_fields(prompts=prompts, sample_index=sample_index),
                    sample_index=sample_index,
                    global_steps=global_steps,
                    partition_id=partition_id,
                    num_sessions=num_sessions,
                )
            )
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        failure_reasons: list[str] = []
        stats = {
            "num_input_prompts": len(prompts),
            "num_success_sessions": 0,
            "num_failed_sessions": 0,
            "num_success_outputs": 0,
            "num_failed_uids": 0,
            "failure_reasons": failure_reasons,
        }
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                stats["num_failed_sessions"] += num_sessions
                stats["num_failed_uids"] += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            # Propagate control-flow exceptions such as CancelledError/SystemExit;
            # only ordinary Exceptions are treated as isolated rollout failures.
            if isinstance(outcome, BaseException):
                raise outcome
            stats["num_success_sessions"] += outcome["num_success_sessions"]
            stats["num_failed_sessions"] += outcome["num_failed_sessions"]
            stats["num_success_outputs"] += outcome["num_success_outputs"]
            stats["num_failed_uids"] += outcome["num_failed_uids"]
            failure_reasons.extend(outcome["failure_reasons"])
        return stats

    async def _run_prompt_sessions_to_tq(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        global_steps: int,
        partition_id: str,
        num_sessions: int,
    ) -> dict:
        uid = sample_fields.get("uid")
        if uid is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['uid'] for TransferQueue output")
        uid = str(uid)
        sampling_params = self._build_session_sampling_params(
            partition_id=partition_id,
            sample_fields=sample_fields,
        )

        # Prompt layer: rollout.n sessions race independently for the same uid.
        # Successful sessions are written to TQ; failed sessions only affect this uid's stats.
        tasks = [
            self._run_session_with_concurrency_limit(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                sampling_params=sampling_params,
            )
            for session_index in range(num_sessions)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        success_sessions = 0
        failed_sessions = 0
        success_outputs = 0
        failure_reasons: list[str] = []
        for session_index, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed_sessions += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            # Propagate control-flow exceptions such as CancelledError/SystemExit;
            # only ordinary Exceptions are treated as isolated rollout failures.
            if isinstance(outcome, BaseException):
                raise outcome

            trajectories, session_sample_fields = outcome
            if not trajectories:
                failed_sessions += 1
                failure_reasons.append(f"empty trajectories for uid={uid} session_index={session_index}")
                continue

            success_sessions += 1
            await self._write_session_trajectories_to_tq(
                uid=uid,
                session_index=session_index,
                trajectories=trajectories,
                sample_fields=session_sample_fields,
                global_steps=global_steps,
                partition_id=partition_id,
            )
            success_outputs += len(trajectories)

        if success_sessions > 0:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
            failed_uids = 0
        else:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
            failed_uids = 1

        return {
            "num_success_sessions": success_sessions,
            "num_failed_sessions": failed_sessions,
            "num_success_outputs": success_outputs,
            "num_failed_uids": failed_uids,
            "failure_reasons": failure_reasons,
        }

    async def _run_session_with_concurrency_limit(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        session_index: int,
        sampling_params: dict[str, object],
    ) -> tuple[list[Trajectory], dict[str, object]]:
        # Lazy-init semaphores on first use and rebind if the running loop
        # changed: asyncio.Semaphore binds to the loop at construction, but
        # Ray actors may run sessions on a different loop than __init__.
        loop = asyncio.get_running_loop()
        if self._semaphore_loop is not loop:
            self._runner_semaphores = {}
            self._semaphore_loop = loop

        if len(self.runner_registry) == 1:
            runner_name, runner_config = next(iter(self.runner_registry.items()))
        else:
            agent_name = sample_fields.get("agent_name")
            if agent_name is None:
                raise ValueError("agent_name is required when multiple agent_runners are configured")
            if not isinstance(agent_name, str):
                raise ValueError(f"agent_name must be a string, got {type(agent_name).__name__}")
            try:
                runner_name = agent_name
                runner_config = self.runner_registry[runner_name]
            except KeyError as exc:
                raise ValueError(f"Unknown agent runner: {agent_name}") from exc

        runner_cap = runner_config.max_concurrent_sessions
        if runner_cap <= 0:
            return await self._run_session(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                runner_name=runner_name,
                runner_config=runner_config,
                sampling_params=sampling_params,
            )

        runner_semaphore = self._runner_semaphores.get(runner_name)
        if runner_semaphore is None:
            runner_semaphore = asyncio.Semaphore(runner_cap)
            self._runner_semaphores[runner_name] = runner_semaphore

        async with runner_semaphore:
            return await self._run_session(
                sample_fields=sample_fields,
                sample_index=sample_index,
                session_index=session_index,
                runner_name=runner_name,
                runner_config=runner_config,
                sampling_params=sampling_params,
            )

    async def _run_session(
        self,
        *,
        sample_fields: dict[str, object],
        sample_index: int,
        session_index: int,
        runner_name: str,
        runner_config: _RunnerConfig,
        sampling_params: dict[str, object],
    ) -> tuple[list[Trajectory], dict[str, object]]:
        """Run one gateway session lifecycle and return finalized trajectories."""
        session_id = f"session-{sample_index}-{session_index}-{uuid4().hex}"
        raw_prompt = sample_fields["raw_prompt"]
        tools_kwargs = sample_fields.get("tools_kwargs")
        session = await self.gateway_manager.create_session(
            session_id,
            sampling_params=dict(sampling_params),
        )
        try:
            if runner_config.dispatch_mode == "ray_task":
                # Ray workers run only the runner. Gateway token truth,
                # finalization, reward scoring, and TQ writes stay in parent.
                object_ref = _run_agent_runner_ray_task.remote(
                    runner_fqn=runner_config.runner_fqn,
                    runner_kwargs=runner_config.runner_kwargs,
                    raw_prompt=raw_prompt,
                    session=session,
                    sample_index=sample_index,
                    tools_kwargs=tools_kwargs,
                )
                await object_ref
            else:
                runner = self._inline_runners[runner_name]
                await runner(
                    raw_prompt=raw_prompt,
                    session=session,
                    sample_index=sample_index,
                    **({"tools_kwargs": tools_kwargs} if tools_kwargs is not None else {}),
                )
            session_trajectories = await self.gateway_manager.finalize_session(session_id)
        except Exception:
            await self.gateway_manager.abort_session(session_id)
            raise

        # Score the session's trajectories immediately after finalization,
        # consistent with VERL's per-sample reward path.
        if not self.reward_loop_worker_handles or not session_trajectories:
            return session_trajectories, sample_fields

        annotations = await self._score_trajectories(session_trajectories, sample_fields)
        scored_trajectories = []
        for traj, (score, extra) in zip(session_trajectories, annotations, strict=True):
            scored_trajectories.append(
                replace(
                    traj,
                    reward_score=score,
                    extra_fields={**traj.extra_fields, "reward_extra_info": extra},
                )
            )
        return scored_trajectories, sample_fields

    async def _score_trajectories(
        self,
        session_trajectories: list[Trajectory],
        sample_fields: dict[str, object],
    ) -> list[tuple[float, dict[str, object]]]:
        """Score the session's final trajectory and broadcast (score, extra_info) to all.

        Mirrors AgentLoopWorkerTQ._agent_loop_postprocess
        (verl/trainer/main_ppo_sync.py:353-396): only the final trajectory (the
        session's last interaction segment) is dispatched to RewardLoopWorker;
        its score + reward_extra_info are then broadcast to every trajectory in
        the session. Subclasses can override this method to implement custom
        session-to-trajectory scoring policies.
        """
        assert self.reward_loop_worker_handles is not None
        assert session_trajectories, "expected non-empty session_trajectories"

        final_trajectory = session_trajectories[-1]
        scoring_sample_fields = dict(sample_fields)
        if final_trajectory.reward_info:
            scoring_sample_fields["extra_info"] = {
                **dict(sample_fields.get("extra_info") or {}),
                **final_trajectory.reward_info,
            }
        data = _trajectory_to_reward_dataproto(final_trajectory, scoring_sample_fields)
        worker = random.choice(self.reward_loop_worker_handles)
        result = await worker.compute_score.remote(data)

        if not isinstance(result, dict) or "reward_score" not in result:
            raise ValueError(
                f"RewardLoopWorker result missing 'reward_score' key or invalid for uid={sample_fields.get('uid')}"
            )
        score = float(result["reward_score"])
        extra = dict(result.get("reward_extra_info") or {})
        return [(score, extra)] * len(session_trajectories)

    def _extract_sample_fields(self, *, prompts: TensorDict, sample_index: int) -> dict[str, object]:
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value if value.ndim == 0 else value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data
        return sample_fields

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_index: int,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int,
        partition_id: str,
    ) -> None:
        keys = []
        fields = []
        tags = []
        for index, trajectory in enumerate(trajectories):
            field, tag = self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=sample_fields,
                session_index=session_index,
                global_steps=global_steps,
                uid=uid,
            )
            keys.append(f"{uid}_{session_index}_{index}")
            fields.append(field)
            tags.append(tag)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=_list_of_tq_fields_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
        session_index: int,
        global_steps: int,
        uid: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        prompts = torch.tensor(trajectory.prompt_ids, dtype=torch.long)
        responses = torch.tensor(trajectory.response_ids, dtype=torch.long)
        response_mask = torch.tensor(trajectory.response_mask, dtype=torch.long)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        multi_modal_inputs = compute_multi_modal_inputs(
            self._processor,
            input_ids.unsqueeze(0),
            trajectory.multi_modal_data,
        )
        if self._processor is None:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        else:
            position_ids = compute_position_ids(
                self._processor,
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)

        field: dict[str, object] = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "loss_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }
        if trajectory.response_logprobs is not None:
            field["rollout_log_probs"] = torch.tensor(trajectory.response_logprobs, dtype=torch.float32)
        else:
            field["rollout_log_probs"] = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.routed_experts is not None:
            field["routed_experts"] = (
                torch.from_numpy(trajectory.routed_experts.copy())
                if hasattr(trajectory.routed_experts, "copy")
                and not isinstance(trajectory.routed_experts, torch.Tensor)
                else trajectory.routed_experts
            )
        rm_scores = torch.zeros_like(responses, dtype=torch.float32)
        if trajectory.reward_score is not None and responses.numel() > 0:
            rm_scores[-1] = float(trajectory.reward_score)
        field["rm_scores"] = rm_scores

        extra_fields = dict(trajectory.extra_fields)
        extra_fields.pop("materialization_reason", None)
        field.update(extra_fields)
        field.pop("multi_modal_data", None)
        for key in ("uid", "raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
            if key in sample_fields:
                field[key] = sample_fields[key]
        field["session_id"] = session_index
        field["global_steps"] = global_steps
        field["num_turns"] = torch.tensor(int(trajectory.num_turns), dtype=torch.long)

        prompt_len = prompts.size(0)
        response_len = responses.size(0)
        tag = {
            "global_steps": global_steps,
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
            "uid": uid,
        }
        materialization_reason = trajectory.extra_fields.get("materialization_reason")
        if materialization_reason is not None:
            tag["materialization_reason"] = materialization_reason
        return field, tag
