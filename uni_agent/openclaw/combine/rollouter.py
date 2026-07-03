"""OpenClaw Combine rollouter (client-driven RL + OPD) for fully_async_policy."""

from __future__ import annotations

import logging
import os
from typing import Optional

import ray
from omegaconf import OmegaConf

from uni_agent.openclaw import scorers
from uni_agent.openclaw.combine.sample_builder import build_combine_rollout_sample
from uni_agent.openclaw.combine.server import OpenClawCombineServer
from uni_agent.openclaw.opd import teacher_channel
from uni_agent.openclaw.opd.rollouter import _OpenClawOPDRollouterImpl

logger = logging.getLogger(__name__)


class _OpenClawCombineRollouterImpl(_OpenClawOPDRollouterImpl):
    """Implementation class (decorated with @ray.remote below).

    Reuses all online/OPD machinery; only the scorer (hint + eval judge), the
    proxy server (four-way dispatch) and the per-turn sample (reward + teacher)
    differ, supplied via the :meth:`_make_scorer` / :meth:`_make_server` /
    :meth:`_build_sample` hooks.
    """

    _sample_prefix = "openclawcombine"

    # ----------------------------------------------------------- scorer
    def _make_scorer(self) -> Optional[scorers.GenericCombinedScorer]:  # type: ignore[override]
        addr = getattr(self.reward_loop_manager, "reward_router_address", None)
        if not addr:
            logger.warning(
                "[OpenClawCombineRollouter] no GenRM address; combine scorer disabled "
                "(all turns dropped). Enable reward.reward_model."
            )
            return None
        rm_cfg = self.config.reward.reward_model
        model_name = rm_cfg.model_path
        prm_cfg = OmegaConf.select(self.config, "reward.reward_model.openclaw", default={}) or {}
        prm_m = int(prm_cfg.get("prm_m", 3))
        temperature = float(prm_cfg.get("temperature", 0.6))
        max_tokens = int(prm_cfg.get("max_tokens", 2048))
        force_hint = str(prm_cfg.get("force_hint", "") or os.environ.get("OPENCLAW_COMBINE_FORCE_HINT", ""))

        generate_fn = teacher_channel.make_judge_generate_fn(
            addr,
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        teacher_fn = teacher_channel.make_teacher_logprobs_fn(
            addr,
            model_name,
            self.tokenizer,
            distill_topk=0,
        )
        return scorers.GenericCombinedScorer(
            generate_fn,
            teacher_fn,
            prm_m=prm_m,
            force_hint=force_hint,
        )

    # ----------------------------------------------------------- submission
    def _build_sample(self, *, session_id, turn, sample_id, uid, rollout_status, **kwargs):  # type: ignore[override]
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return build_combine_rollout_sample(
            prompt_ids=kwargs["prompt_ids"],
            response_ids=kwargs["response_ids"],
            response_logprobs=kwargs["response_logprobs"],
            teacher_log_probs=kwargs["teacher_log_probs"],
            score=kwargs["score"],
            sample_id=sample_id,
            uid=uid,
            pad_token_id=self.tokenizer.pad_token_id,
            prompt_length=rollout_cfg.prompt_length,
            response_length=rollout_cfg.response_length,
            param_version=self._current_param_version,
            rollout_status=rollout_status,
            turn=turn,
            sample_kind=kwargs.get("sample_kind", "opd+rl"),
        )

    # ----------------------------------------------------------- proxy
    def _make_server(self, scorer, *, max_tokens: int, host: str, port: int, api_key: str, gateway_manager=None):  # type: ignore[override]
        return OpenClawCombineServer(
            combine_scorer=scorer,
            submit_fn=self._submit_sample,
            is_paused=lambda: self.paused,
            model_name=self.config.actor_rollout_ref.model.path,
            default_max_tokens=max_tokens,
            host=host,
            port=port,
            api_key=api_key,
            gateway_manager=gateway_manager,
        )


OpenClawCombineRollouter = ray.remote(num_cpus=10, max_concurrency=100)(_OpenClawCombineRollouterImpl)
