import asyncio
import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import yaml

from uni_agent.async_logging import add_file_handler, get_logger
from uni_agent.interaction import (
    AgentChatModel,
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.reward import load_reward_spec
from uni_agent.skills import SkillsManager, SkillsManagerConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.utils import resolve_config_path


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` on top of ``base``, returning a new dict.

    - Nested dicts are merged key-wise (``overrides`` wins on conflicts).
    - Lists and all non-dict types are replaced wholesale (no element-wise merge).
    - An empty dict in ``overrides`` is a no-op for that key (use ``None`` to clear).
    - ``base`` is never mutated.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class UniAgentLoop(AgentLoopBase):
    _semaphore: asyncio.Semaphore | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        config_dict = self._init_config(sampling_params, **kwargs)
        self.mask_abnormal_exit_traj = config_dict.get("mask_abnormal_exit_traj", False)
        global_concurrent = config_dict.get("concurrency", 512)
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        worker_concurrent = max(global_concurrent // num_workers, 1)
        if UniAgentLoop._semaphore is None:
            UniAgentLoop._semaphore = asyncio.Semaphore(worker_concurrent)

        self.run_id = str(uuid.uuid4())
        self.logger = get_logger("agent-loop", run_id=self.run_id)
        # init chat model, tools manager and environment
        self.chat_model = self._init_chat_model(config_dict["model"])
        self.tools_manager = self._init_tools_manager(
            tools_config_list=config_dict["tools"],
            parser=config_dict.get("tool_parser", "qwen3_coder"),
        )
        self.skills_manager = self._init_skills_manager(config_dict.get("skills"))
        self.env = self._init_env(config_dict["env"])
        self.output_dir = Path(config_dict["log_dir"]) / self.run_id
        self.interaction = AgentInteraction(
            run_id=self.run_id,
            env=self.env,
            model=self.chat_model,
            tools_manager=self.tools_manager,
            messages=list(kwargs["raw_prompt"]),
            skills_manager=self.skills_manager,
            **config_dict["interaction"],
        )
        if config_dict["reward"] is not None:
            reward_config = {
                **config_dict["reward"],
                "run_id": self.run_id,
                "env": self.env,
            }
            self.reward_spec = load_reward_spec(reward_config)
        else:
            self.reward_spec = None

        add_file_handler(self.output_dir / "run.log", self.run_id)

        self.logger.info(f"model name: {self.config.actor_rollout_ref.model.path}")
        self.logger.info(f"sampling_params: {sampling_params}")
        self.logger.info(f"environment config: {config_dict['env']}")
        self.logger.info(f"tools config: {config_dict['tools']}")
        self.logger.info(f"interaction config: {config_dict['interaction']}")
        self.logger.info(f"mask_abnormal_exit_traj: {self.mask_abnormal_exit_traj}")
        self.logger.info(f"output_dir: {self.output_dir}")

        async with self._semaphore:
            try:
                await self.env.start()

                # tools schemas should be visible to the model
                # to generate correct tool call format in response
                self.chat_model.set_tools_schemas(self.tools_manager.tools_schemas)
                await self.env.install_tools(self.tools_manager.tools)
                if self.skills_manager is not None:
                    await self.env.install_skills(self.skills_manager)
                    self.interaction.inject_skills_manifest()

                interaction_result = await self.interaction.run()
                interaction_result["metrics"] = dict(interaction_result.get("rollout_cache", {}).get("metrics", {}))

                # interaction environment should be visible to the reward spec
                if self.reward_spec is not None:
                    reward_score, _ = await self.reward_spec.compute_reward(
                        interaction_result=interaction_result,
                    )
                    interaction_result["reward_score"] = reward_score
                else:
                    self.logger.warning("No reward spec is provided, reward score will be set to -100")
                    interaction_result["reward_score"] = -100

                self._save_interaction_result(interaction_result)
                output = await self.convert_to_agent_output(interaction_result)
            except Exception as e:
                self.logger.critical(f"Agent loop failed before producing interaction result: {e}")
                output = await self._build_empty_agent_output(exit_reason="agent_loop_failed")
            finally:
                await self.env.close()
            return output

    async def _build_empty_agent_output(self, exit_reason: str) -> AgentLoopOutput:
        self.chat_model.set_tools_schemas(self.tools_manager.tools_schemas)
        rollout_cache = await self.chat_model.prepare_rollout_cache(self.interaction.messages)
        prompt_ids = rollout_cache["prompt_ids"]
        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]

        dummy_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if dummy_token_id is None:
            dummy_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(dummy_token_id, list):
            dummy_token_id = dummy_token_id[0] if dummy_token_id else 0
        if dummy_token_id is None:
            dummy_token_id = 0

        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        dummy_response_length = min(512, max_response_length)

        extra_fields = dict(rollout_cache.get("extra_fields") or {})
        # TODO: implement traj_mask in verl
        extra_fields["traj_masked"] = 1
        extra_fields["traj_exit_reason"] = exit_reason
        extra_fields["global_steps"] = 0
        extra_fields["min_global_steps"] = 0
        extra_fields["max_global_steps"] = 0

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[dummy_token_id] * dummy_response_length,
            response_mask=[0] * dummy_response_length,
            response_logprobs=[0.0] * dummy_response_length,
            routed_experts=None,
            multi_modal_data={},
            reward_score=0,
            num_turns=0,
            metrics={},
            extra_fields=extra_fields,
        )

    def _save_interaction_result(self, interaction_result: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # rollout_cache: binary pickle for fast I/O (no readability needed)
        with (self.output_dir / "rollout_cache.pkl").open("wb") as f:
            pickle.dump(interaction_result["rollout_cache"], f, protocol=pickle.HIGHEST_PROTOCOL)
        # rest: readable JSON
        save_content = {
            "trajectory": [s.model_dump() for s in interaction_result["trajectory"]],
            "execution_time": interaction_result["execution_time"],
            "messages": interaction_result["messages"],
            "metrics": interaction_result.get("metrics", {}),
        }
        (self.output_dir / "interaction_result.json").write_text(
            json.dumps(save_content, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _init_config(self, sampling_params: dict[str, Any], **kwargs):
        """Assemble the effective per-run config.

        The config is built from two sources, in priority order (later wins):

        1. **YAML defaults** at ``rollout.agent.agent_loop_config_path``.
           Provides global fields like ``_target_``, ``name``, ``log_dir``,
           ``concurrency``, and any default values for ``env`` / ``tools`` /
           ``interaction`` / ``reward``.
        2. **Per-sample overrides** in ``kwargs["tools_kwargs"]`` (carried in
           the dataset's ``extra_info.tools_kwargs``). Any top-level field is
           overridable here *except* ``model`` (which is always synthesized
           from rollout/server state below). Dicts are deep-merged, so a
           partial override like ``{"env": {"deployment": {"image": "..."}}}``
           leaves everything else in the YAML's ``env`` untouched.

        This supports two usage styles cleanly:

        - **Partial init config** (existing behavior): YAML carries most of
          the config; per-sample ``tools_kwargs`` patches in a few task-
          specific bits (image, reward metadata, etc.).
        - **Full config in the dataset**: the preprocessing script emits a
          fully-specified ``tools_kwargs`` (env + reward + interaction +
          tools + ...); the YAML is then only a thin shell carrying
          ``_target_`` / ``name`` / global knobs.
        """
        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        assert agent_loop_config_path is not None, "agent_loop_config_path is None"
        resolved_path = resolve_config_path(agent_loop_config_path)
        base_config = yaml.safe_load(Path(resolved_path).read_text())[0]

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        if "model" in tools_kwargs:
            raise ValueError(
                "tools_kwargs.model is reserved; the model config is always "
                "derived from the rollout config and cannot be overridden "
                "per-sample. Remove `model` from your dataset's tools_kwargs."
            )
        config_dict = _deep_merge(base_config, tools_kwargs)

        rollout_config = self.config.actor_rollout_ref.rollout
        max_model_len = (
            rollout_config.max_model_len
            if rollout_config.max_model_len is not None
            else rollout_config.prompt_length + rollout_config.response_length
        )
        config_dict["model"] = {
            "client": self.server_manager,
            "tokenizer": self.tokenizer,
            "max_model_len": max_model_len,
            "sampling_params": sampling_params,
        }

        if not config_dict.get("reward"):
            config_dict["reward"] = None

        return config_dict

    def _init_chat_model(self, config_dict: dict) -> AgentChatModel:
        chat_model = AgentChatModel(**config_dict)
        return chat_model

    def _init_tools_manager(self, tools_config_list: list[dict], parser: str = "qwen3_coder") -> ToolsManager:
        tools_manager_config = ToolsManagerConfig(tools=tools_config_list, parser=parser)
        return ToolsManager(tools_manager_config=tools_manager_config)

    def _init_skills_manager(self, skills_config: dict | None) -> SkillsManager | None:
        """Build a SkillsManager from per-run config.

        - ``skills_config is None`` or missing: skills system disabled (no
          manifest injection, no container push). Backward-compatible default.
        - ``skills_config`` provided: build a ``SkillsManagerConfig`` and
          scan its ``skills_dir`` for ``<name>/SKILL.md`` subdirectories.
        """
        if not skills_config:
            return None
        cfg = SkillsManagerConfig(**skills_config)
        return SkillsManager.from_config(cfg)

    def _init_env(self, config_dict: dict) -> AgentEnv:
        env_config = AgentEnvConfig(**config_dict)
        return AgentEnv(run_id=self.run_id, env_config=env_config)

    async def convert_to_agent_output(self, interaction_result: dict) -> AgentLoopOutput:
        rollout_cache = interaction_result["rollout_cache"]
        reward_score = interaction_result.get("reward_score", None)

        if len(rollout_cache["response_mask"]) == 0:
            return await self._build_empty_agent_output(
                exit_reason="no_response",
            )

        num_turns = len(interaction_result["trajectory"])
        self.logger.info(f"num_turns: {num_turns}")

        prompt_ids = rollout_cache["prompt_ids"]
        traj_exit_reason = interaction_result["trajectory"][-1].exit_reason if num_turns > 0 else "unknown"
        should_mask_traj = self.mask_abnormal_exit_traj and traj_exit_reason != "finished"
        traj_masked = int(should_mask_traj)

        if should_mask_traj:
            response_mask = [0] * len(rollout_cache["response_mask"])
        else:
            response_mask = rollout_cache["response_mask"]
        response_logprobs = rollout_cache.get("response_logprobs") or []
        routed_experts = rollout_cache.get("routed_experts")
        metrics = interaction_result.get("metrics", rollout_cache.get("metrics", {}))
        extra_fields = dict(rollout_cache.get("extra_fields") or {})
        extra_fields["traj_masked"] = traj_masked
        extra_fields["traj_exit_reason"] = traj_exit_reason
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
            self.logger.warning(
                f"prompt_ids length {len(prompt_ids)} exceeds max_prompt_length {max_prompt_length} "
                "truncate prompt_ids length"
            )
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]
            response_mask = response_mask[:max_response_length]
            response_logprobs = response_logprobs[:max_response_length]
            self.logger.warning(
                f"response_ids length {len(response_ids)} exceeds max_response_length {max_response_length} "
                "truncate response_ids length"
            )

        self.logger.info(f"prompt_ids length: {len(prompt_ids)}")
        self.logger.info(f"response_ids length: {len(response_ids)}")
        self.logger.info(f"reward_score: {reward_score}")
        response_logprobs = response_logprobs if response_logprobs else None
        if routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + len(response_ids)]

        multi_modal_data = {}
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )
