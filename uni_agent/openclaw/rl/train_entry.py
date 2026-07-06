"""Entry point for OpenClaw RL (client-driven, no train data) training."""

from __future__ import annotations

import hydra

from uni_agent.openclaw.common.entry import patch_fully_async, run
from uni_agent.openclaw.rl.rollouter import OpenClawRLRollouter
from uni_agent.openclaw.rl.trainer import OpenClawFullyAsyncTrainer

patch_fully_async(OpenClawRLRollouter, OpenClawFullyAsyncTrainer)


@hydra.main(config_path="config", config_name="openclaw_rl", version_base=None)
def main(config):
    run(config)


if __name__ == "__main__":
    main()
