"""Entry point for OpenClaw Combine (RL + OPD) on fully_async_policy."""

from __future__ import annotations

import hydra

from uni_agent.openclaw.losses import ensure_registered

ensure_registered()

from uni_agent.openclaw.combine.rollouter import OpenClawCombineRollouter  # noqa: E402
from uni_agent.openclaw.common.entry import patch_fully_async, run  # noqa: E402
from uni_agent.openclaw.rl.trainer import OpenClawFullyAsyncTrainer as OpenClawCombineTrainer  # noqa: E402

patch_fully_async(OpenClawCombineRollouter, OpenClawCombineTrainer)


@hydra.main(config_path="config", config_name="openclaw_combine", version_base=None)
def main(config):
    run(config)


if __name__ == "__main__":
    main()
