"""OpenClaw personal-agent optimization, re-platformed onto uni-agent.

This package ports the "personal OpenClaw optimization" track of OpenClaw-RL
(https://github.com/Gen-Verse/OpenClaw-RL) -- Binary RL, On-Policy Distillation
(OPD) and the Combine/Top-K-Select hybrid.
"""

from uni_agent.openclaw import protocol, scorers

__all__ = ["protocol", "scorers"]
