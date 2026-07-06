"""OpenClaw OPD online rollouter for verl ``fully_async_policy``."""

from __future__ import annotations

import logging
import os
from typing import Optional

import ray
from omegaconf import OmegaConf

from uni_agent.openclaw import scorers
from uni_agent.openclaw.opd import teacher_channel
from uni_agent.openclaw.opd.sample_builder import build_opd_rollout_sample
from uni_agent.openclaw.opd.server import OpenClawOPDServer
from uni_agent.openclaw.rl.rollouter import _OpenClawRLRollouterImpl

logger = logging.getLogger(__name__)


class _OpenClawOPDRollouterImpl(_OpenClawRLRollouterImpl):
    """Implementation class (decorated with ``@ray.remote`` below)."""

    _sample_prefix = "openclawopd"

    # ------------------------------------------- disable native teacher infra
    async def init_workers(self):
        """Disable distillation on the rollouter's config copy before init.

        OPD computes teacher log-probs externally against the GenRM (with the
        hindsight hint appended). verl's native distillation infra is therefore
        unused on the rollout side, but ``distillation.enabled=true`` would make
        both :meth:`_create_teacher_model_manager` (a teacher GPU pool) and the
        per-worker ``AsyncTeacherLLMServerManager`` (agent loop) try to wire a
        native teacher -- the latter crashes on a ``None`` teacher client. Since
        the rollouter never runs the distillation loss, we flip ``enabled`` off
        on its **own** config copy; the trainer's actor workers run in a separate
        Ray actor with their own config copy and keep distillation enabled for
        the OPD loss.
        """
        try:
            if self.config.get("distillation") is not None and self.config.distillation.get("enabled"):
                self.config.distillation.enabled = False
                logger.info(
                    "[OpenClawOPDRollouter] native distillation teacher infra disabled on "
                    "rollouter (teacher log-probs computed externally via GenRM)"
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[OpenClawOPDRollouter] could not disable rollouter distillation: %s", e)
        return await super().init_workers()

    # ----------------------------------------------------------- distill topk
    def _distill_topk(self) -> int:
        loss_cfg = OmegaConf.select(self.config, "distillation.distillation_loss", default={}) or {}
        loss_mode = loss_cfg.get("loss_mode", "")
        if loss_mode == "forward_kl_topk":
            return int(loss_cfg.get("topk", 0) or 0)
        return 0

    # ----------------------------------------------------------- OPD scorer
    def _make_scorer(self) -> Optional[scorers.GenericOPDScorer]:  # type: ignore[override]
        addr = getattr(self.reward_loop_manager, "reward_router_address", None)
        if not addr:
            logger.warning(
                "[OpenClawOPDRollouter] no GenRM address; OPD scorer disabled "
                "(all turns dropped). Enable reward.reward_model."
            )
            return None
        rm_cfg = self.config.reward.reward_model
        model_name = rm_cfg.model_path
        prm_cfg = OmegaConf.select(self.config, "reward.reward_model.openclaw", default={}) or {}
        prm_m = int(prm_cfg.get("prm_m", 3))
        temperature = float(prm_cfg.get("temperature", 0.6))
        max_tokens = int(prm_cfg.get("max_tokens", 2048))
        eval_mode = bool(prm_cfg.get("eval_mode", False))
        force_hint = str(prm_cfg.get("force_hint", "") or os.environ.get("OPENCLAW_OPD_FORCE_HINT", ""))

        generate_fn = teacher_channel.make_judge_generate_fn(
            addr, model_name, temperature=temperature, max_tokens=max_tokens
        )
        teacher_fn = teacher_channel.make_teacher_logprobs_fn(
            addr, model_name, self.tokenizer, distill_topk=self._distill_topk()
        )
        return scorers.GenericOPDScorer(
            generate_fn, teacher_fn, prm_m=prm_m, eval_mode=eval_mode, force_hint=force_hint
        )

    # ----------------------------------------------------------- submission
    def _build_sample(self, *, session_id, turn, sample_id, uid, rollout_status, **kwargs):  # type: ignore[override]
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return build_opd_rollout_sample(
            prompt_ids=kwargs["prompt_ids"],
            response_ids=kwargs["response_ids"],
            response_logprobs=kwargs["response_logprobs"],
            teacher_log_probs=kwargs["teacher_log_probs"],
            teacher_topk_log_probs=kwargs.get("teacher_topk_log_probs"),
            teacher_topk_indices=kwargs.get("teacher_topk_indices"),
            distill_topk=self._distill_topk(),
            sample_id=sample_id,
            uid=uid,
            pad_token_id=self.tokenizer.pad_token_id,
            prompt_length=rollout_cfg.prompt_length,
            response_length=rollout_cfg.response_length,
            param_version=self._current_param_version,
            rollout_status=rollout_status,
            turn=turn,
            eval_score=kwargs.get("eval_score"),
        )

    # ----------------------------------------------------------- proxy
    def _make_server(self, scorer, *, max_tokens: int, host: str, port: int, api_key: str, gateway_manager=None):  # type: ignore[override]
        return OpenClawOPDServer(
            opd_scorer=scorer,
            distill_topk=self._distill_topk(),
            submit_fn=self._submit_sample,
            is_paused=lambda: self.paused,
            model_name=self.config.actor_rollout_ref.model.path,
            default_max_tokens=max_tokens,
            host=host,
            port=port,
            api_key=api_key,
            gateway_manager=gateway_manager,
        )


OpenClawOPDRollouter = ray.remote(num_cpus=10, max_concurrency=100)(_OpenClawOPDRollouterImpl)
