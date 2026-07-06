"""Entry point for OpenClaw OPD (on-policy distillation, client-driven) training."""

from __future__ import annotations

import hydra

from uni_agent.openclaw.losses import ensure_registered

# Register openclaw_opd / openclaw_combine / forward_kl_topk-compatible losses
# before Hydra instantiates DistillationLossConfig (driver-side validation).
ensure_registered()

from uni_agent.openclaw.common.entry import patch_fully_async, run  # noqa: E402
from uni_agent.openclaw.opd.rollouter import OpenClawOPDRollouter  # noqa: E402
from uni_agent.openclaw.opd.trainer import OpenClawOPDTrainer  # noqa: E402

# Swap the rollouter (client-driven OPD generation + external teacher) and the
# trainer (task advantage zeroed; OPD signal in the distillation loss).
patch_fully_async(OpenClawOPDRollouter, OpenClawOPDTrainer)


@hydra.main(config_path="config", config_name="openclaw_opd", version_base=None)
def main(config):
    run(config)


if __name__ == "__main__":
    main()
