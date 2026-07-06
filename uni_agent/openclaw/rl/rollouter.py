"""OpenClaw RL rollouter for verl ``fully_async_policy``."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional
from uuid import uuid4

import ray
from omegaconf import OmegaConf

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.manager import GatewayManager
from uni_agent.openclaw import scorers
from uni_agent.openclaw.common.http import post_json as _post_json
from uni_agent.openclaw.rl.sample_builder import build_rollout_sample
from uni_agent.openclaw.rl.server import OpenClawRLServer
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter as _FullyAsyncRollouterActor,
)
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__name__)

_FullyAsyncRollouterBase = _FullyAsyncRollouterActor.__ray_actor_class__


class _OpenClawRLRollouterImpl(_FullyAsyncRollouterBase):
    """Implementation class (decorated with ``@ray.remote`` below).

    Subclasses (OPD / Combine) customise only the per-turn data path via the
    template-method hooks :meth:`_make_scorer`, :meth:`_make_server` and
    :meth:`_build_sample`; the shared proxy launch (:meth:`_start_proxy`) and
    MessageQueue submission (:meth:`_submit_sample`) live here once.
    """

    # Prefix for generated sample ids (overridden by OPD / Combine subclasses).
    _sample_prefix = "openclaw"

    def __init__(self, config, tokenizer, processor=None, device_name=None):
        # Replicate FullyAsyncRollouter.__init__ WITHOUT dataset/dataloader
        # creation -- online data is client-driven, there is no train file.
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        from verl.trainer.ppo.utils import need_reward_model

        self.use_reference_policy = False
        self.use_rm = need_reward_model(self.config)
        if self.use_rm:
            assert self.config.reward.reward_model.enable_resource_pool, (
                "GenRM in fully async mode requires standalone mode (enable_resource_pool=True)."
            )
        self.use_critic = False
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self._init_dump_executor()

        # ---- online: total rollout steps come from config, not a dataloader ----
        total_rollout_steps = self.config.rollout.get("total_rollout_steps", None)
        assert total_rollout_steps is not None and total_rollout_steps > 0, (
            "online training requires rollout.total_rollout_steps > 0"
        )
        self.total_rollout_steps = int(total_rollout_steps)
        print(f"[OpenClawRLRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None
        self.train_dataloader = None  # no dataloader in online mode

        self.message_queue_client = None
        self.async_rollout_manager = None
        self._hybrid_worker_group = None

        self.staleness_threshold = config.async_training.get("staleness_threshold", 1)
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        self.max_queue_size = None

        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        self.paused = False
        self.running = True

        self.dataloader_lock = asyncio.Lock()
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()

        # online-specific
        self._current_param_version = 0
        self._server: OpenClawRLServer | None = None
        self._gateway_manager: GatewayManager | None = None

    # ---- dataset/dataloader-dependent methods become no-ops in online mode ----
    def load_checkpoint(self):
        """Online mode keeps no dataloader state; resume of in-flight client
        traffic is not supported, so this is a no-op (start from scratch)."""
        print("[OpenClawRLRollouter] online mode: skipping dataloader checkpoint load")
        return 0

    async def save_checkpoint(self, local_global_step_folder: str):
        print("[OpenClawRLRollouter] online mode: skipping dataloader checkpoint save")

    def do_validate(self):
        """No validation dataset in online mode."""
        return {}

    async def reset_staleness(self):
        # Track the rollout server's weight version so produced samples carry the
        # correct param_version for the trainer's staleness accounting.
        self._current_param_version += 1
        return await super().reset_staleness()

    def _gateway_count(self) -> int:
        af_cfg = OmegaConf.select(self.config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        openclaw_cfg = af_cfg.get("openclaw", {}) or {}
        rl_cfg = openclaw_cfg.get("rl", {}) or {}
        return int(rl_cfg.get("gateway_count", os.environ.get("OPENCLAW_GATEWAY_COUNT", "1")))

    async def _ensure_gateway_manager(self) -> None:
        if self._gateway_manager is not None:
            return
        gateway_actor_config = GatewayActorConfig(
            tokenizer=self.tokenizer,
            processor=self.processor,
            tool_parser_name=self.config.actor_rollout_ref.rollout.get("multi_turn", {}).get("format"),
            prompt_length=self.config.actor_rollout_ref.rollout.prompt_length,
            response_length=self.config.actor_rollout_ref.rollout.response_length,
        )
        self._gateway_manager = GatewayManager(
            llm_client=self.llm_server_manager.get_client(),
            gateway_count=self._gateway_count(),
            gateway_actor_config=gateway_actor_config,
        )
        logger.info("[%s] gateway manager enabled with count=%d", type(self).__name__, self._gateway_count())

    async def _shutdown_gateway_manager(self) -> None:
        if self._gateway_manager is None:
            return
        await self._gateway_manager.shutdown()
        self._gateway_manager = None

    # ----------------------------------------------------------- PRM (GenRM)
    def _build_genrm_generate_fn(self):
        addr = getattr(self.reward_loop_manager, "reward_router_address", None)
        if not addr:
            return None
        rm_cfg = self.config.reward.reward_model
        model_name = rm_cfg.model_path
        prm_cfg = OmegaConf.select(self.config, "reward.reward_model.openclaw", default={}) or {}
        temperature = float(prm_cfg.get("temperature", 0.6))
        max_tokens = int(prm_cfg.get("max_tokens", 2048))

        async def _generate(messages: list[dict]) -> str:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            out = await _post_json(f"http://{addr}/v1/chat/completions", payload)
            return out["choices"][0]["message"]["content"] or ""

        return _generate

    def _make_scorer(self) -> Optional[scorers.GenericPRMScorer]:
        """Hook: build the per-turn scorer (online: PRM majority-vote judge)."""
        generate_fn = self._build_genrm_generate_fn()
        if generate_fn is None:
            return None
        prm_cfg = OmegaConf.select(self.config, "reward.reward_model.openclaw", default={}) or {}
        prm_m = int(prm_cfg.get("prm_m", 3))
        return scorers.GenericPRMScorer(generate_fn, prm_m=prm_m)

    # ----------------------------------------------------------- submission
    def _build_sample(self, *, session_id, turn, sample_id, uid, rollout_status, **kwargs):
        """Hook: build the per-turn ``RolloutSample`` (online: reward-only)."""
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return build_rollout_sample(
            prompt_ids=kwargs["prompt_ids"],
            response_ids=kwargs["response_ids"],
            response_logprobs=kwargs["response_logprobs"],
            score=kwargs["score"],
            sample_id=sample_id,
            uid=uid,
            pad_token_id=self.tokenizer.pad_token_id,
            prompt_length=rollout_cfg.prompt_length,
            response_length=rollout_cfg.response_length,
            param_version=self._current_param_version,
            rollout_status=rollout_status,
            turn=turn,
            has_next_state=kwargs["has_next_state"],
        )

    async def _submit_sample(self, *, session_id, turn, **build_kwargs) -> None:
        """Build a per-turn sample (via :meth:`_build_sample`) and push it to the MQ.

        The mode-specific keyword arguments passed by each proxy's
        ``submit_fn`` flow straight into :meth:`_build_sample`; the id/uid
        minting, ``get_statistics`` snapshot and staleness accounting are shared.
        """
        sample_id = f"{self._sample_prefix}_{session_id}_t{turn}_{uuid4().hex[:8]}"
        uid = f"{session_id}-t{turn}-{uuid4().hex[:8]}"
        rollout_status = await self.get_statistics()
        rs = self._build_sample(
            session_id=session_id,
            turn=turn,
            sample_id=sample_id,
            uid=uid,
            rollout_status=rollout_status,
            **build_kwargs,
        )
        success = await self.message_queue_client.put_sample(sample=ray.cloudpickle.dumps(rs))
        self.staleness_samples += 1
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    # ----------------------------------------------------------- proxy
    def _proxy_params(self) -> tuple[str, int, str, int]:
        """Resolve ``(host, port, api_key, max_tokens)`` for the embedded proxy."""
        af_cfg = OmegaConf.select(self.config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}
        openclaw_cfg = af_cfg.get("openclaw", {}) or {}
        rl_cfg = openclaw_cfg.get("rl", {}) or {}
        host = rl_cfg.get("host") or os.environ.get("OPENCLAW_RL_HOST", "0.0.0.0")
        port = int(rl_cfg.get("port") or os.environ.get("OPENCLAW_RL_PORT", "30000"))
        api_key = rl_cfg.get("api_key") or os.environ.get("OPENCLAW_RL_API_KEY", "")
        max_tokens = int(self.config.actor_rollout_ref.rollout.response_length)
        return host, port, api_key, max_tokens

    def _make_server(self, scorer, *, max_tokens: int, host: str, port: int, api_key: str, gateway_manager=None):
        """Hook: construct the proxy server (online: PRM-scored proxy)."""
        return OpenClawRLServer(
            submit_fn=self._submit_sample,
            is_paused=lambda: self.paused,
            scorer=scorer,
            model_name=self.config.actor_rollout_ref.model.path,
            default_max_tokens=max_tokens,
            host=host,
            port=port,
            api_key=api_key,
            gateway_manager=gateway_manager,
        )

    def _start_proxy(self) -> None:
        host, port, api_key, max_tokens = self._proxy_params()
        scorer = self._make_scorer()
        self._server = self._make_server(
            scorer,
            max_tokens=max_tokens,
            host=host,
            port=port,
            api_key=api_key,
            gateway_manager=self._gateway_manager,
        )
        self._server.start()
        logger.info("[%s] proxy started on %s:%d", type(self).__name__, host, port)

    # ----------------------------------------------------------- main loop
    async def _streaming_generation_main(self):
        """Run the embedded proxy and manage pause/back-pressure until done."""
        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        await self._ensure_gateway_manager()
        self._start_proxy()
        print("[OpenClawRLRollouter] streaming via OpenClaw proxy; waiting for client traffic")

        try:
            while True:
                async with self.lock:
                    if not self.running:
                        break
                should_pause = await self._should_pause_generation()
                async with self.lock:
                    self.paused = should_pause
                    if self.total_generated_samples >= self.total_rollout_steps:
                        print(
                            f"[OpenClawRLRollouter] reached total_rollout_steps "
                            f"({self.total_generated_samples} >= {self.total_rollout_steps}), stopping"
                        )
                        self.running = False
                        break
                await asyncio.sleep(0.5)
        finally:
            if self._server is not None:
                self._server.stop()
            await self._shutdown_gateway_manager()
            await self.message_queue_client.put_sample(sample=None)
            async with self.lock:
                self.running = False
        print("[OpenClawRLRollouter] streaming generation finished")


OpenClawRLRollouter = ray.remote(num_cpus=10, max_concurrency=100)(_OpenClawRLRollouterImpl)
