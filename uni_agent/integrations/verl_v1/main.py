"""Launch uni-agent training through verl's public TaskRunner extension."""

from __future__ import annotations

from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer import main_ppo as native_entry
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class UniAgentTaskRunner:
    """Own integration construction while native trainers own optimization."""

    def run(self, config: DictConfig):
        configure_verl_logging()
        import transfer_queue as tq

        from .trainer import get_trainer_class

        if not config.trainer.use_v1:
            raise ValueError("The uni-agent verl integration requires trainer.use_v1=True")
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = get_trainer_class(config.trainer.v1.trainer_mode)(config=config)
            trainer.init()
            manager_fqn = (
                OmegaConf.select(
                    config,
                    "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
                    default="uni_agent.framework.entry.AgentFrameworkRolloutAdapter",
                )
                or "uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
            )
            manager_class = load_class_from_fqn(manager_fqn)
            manager = manager_class.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(
    config_path=str(Path(native_entry.__file__).parent / "config"), config_name="ppo_trainer", version_base=None
)
def main(config: DictConfig):
    """Use the native launcher and config with an explicit integration runner."""
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    native_entry.run_ppo(config, task_runner_class=UniAgentTaskRunner)


if __name__ == "__main__":
    main()
