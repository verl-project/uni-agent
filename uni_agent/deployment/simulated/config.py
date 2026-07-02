"""Config for the simulated sandbox deployment.

``SimulatedDeploymentConfig`` is the ``type: simulated`` member of the ``DeployConfig``
discriminated union -- selectable from ``agent_config`` exactly like
host/local/modal. It carries the tunables that control SimulatedRuntime's
observation sampling (seed/scale), plus the failure-simulation knobs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TimeoutSimConfig(BaseModel):
    """Command-timeout simulation. When enabled, a run_in_session call may
    sleep ``delay_seconds`` then raise swerex's ``CommandTimeoutError`` so
    ``AgentEnv`` exercises its real timeout/interrupt/timeout_budget path."""

    enabled: bool = False
    probability: float = Field(default=0.05, ge=0.0, le=1.0)
    delay_seconds: float = Field(default=0.5, ge=0.0)


class TerminalDeadConfig(BaseModel):
    """Terminal-death simulation. When enabled, a run_in_session call may
    raise swerex's ``TerminalNotAliveError`` so the episode exits via the
    real ``terminal_dead`` path."""

    enabled: bool = False
    probability: float = Field(default=0.01, ge=0.0, le=1.0)


class SimulatedDeploymentConfig(BaseModel):
    """Configuration for the simulated (CPU-only, canned-observation) deployment.

    Used for performance testing: the LLM runs for real, the sandbox is
    stubbed. Observation sizes/structure follow hand-tuned representative
    templates (see ``simulated/deployment.py``).
    """

    type: Literal["simulated"] = "simulated"
    """Discriminator for (de)serialization. Do not change."""

    seed: int | None = None
    """Sampling seed. ``None`` (default) = random sampling each run; an int =
    reproducible sampling (same command sequence -> identical outputs)."""

    observation_scale: float = Field(default=1.0, ge=0.0)
    """Multiplier applied to every template's length. 1.0 = as-authored;
    use >1 to stress KV/prefill load."""

    timeout: TimeoutSimConfig = Field(default_factory=TimeoutSimConfig)
    terminal_dead: TerminalDeadConfig = Field(default_factory=TerminalDeadConfig)

    # ``extra="ignore"`` (NOT "forbid"): swebench datasets carry per-sample
    # ``extra_info.tools_kwargs`` overrides for the docker path (image, command,
    # container_runtime, ...). The agent loop deep-merges these onto the YAML
    # deployment block, so SimulatedDeploymentConfig receives docker-only keys it
    # doesn't model. Ignoring them lets the same dataset drive both local and
    # simulated deployments without a per-sample schema fork.
    model_config = ConfigDict(extra="ignore")

    def get_deployment(self, run_id: str):
        from .deployment import SimulatedDeployment

        return SimulatedDeployment.from_config(self, run_id)
