from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.config import LocalDeploymentConfig as SwerexLocalDeploymentConfig


class HostDeploymentConfig(BaseModel):
    """Configuration for host-local execution (no container)."""

    type: Literal["host"] = "host"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake.

    During parameter-sync weight reloads, fork()/exec() and even the asyncio event loop can be
    starved for tens of seconds.
    """

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .host.deployment import HostDeployment

        return HostDeployment.from_config(self, run_id)


class LocalDeploymentConfig(SwerexLocalDeploymentConfig):
    """SWE-ReX local config with Uni-Agent's deployment factory signature."""

    def get_deployment(self, run_id: str):
        from .local.deployment import LocalDeployment

        return LocalDeployment.from_config(self, run_id)


class ModalDeploymentConfig(BaseModel):
    """Configuration for Modal deployment."""

    image: str | PurePath = "python:3.11"
    """Image to use for the deployment."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    runtime_timeout: float = 60.0
    """Timeout for runtime operations."""
    deployment_timeout: float = 3600.0
    """Timeout for the Modal sandbox."""
    modal_sandbox_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to `modal.Sandbox.create`."""
    proxy: str | None = None
    """Proxy to use for runtime HTTP requests."""
    type: Literal["modal"] = "modal"
    """Discriminator for (de)serialization/CLI. Do not change."""
    install_pipx: bool = True
    """Whether to install pipx in the Modal image."""

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .modal.deployment import ModalDeployment

        return ModalDeployment.from_config(self, run_id)


class VefaasDeploymentConfig(BaseModel):
    """Configuration for VEFAAS deployment."""

    image: str | None = None
    """Docker image to use for the sandbox."""
    command: str = "python3 -m swerex.server --auth-token {token}"
    """Command to run in the sandbox with authentication token."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout waiting for runtime to start."""
    function_id: str | None = None
    """VEFAAS function ID."""
    function_route: str | None = None
    """VEFAAS function Route."""
    proxy: str | None = None
    """Proxy to use for the connection."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .vefaas.deployment import VefaasDeployment

        return VefaasDeployment.from_config(self, run_id)


DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig | LocalDeploymentConfig | HostDeploymentConfig | ModalDeploymentConfig,
    Field(discriminator="type"),
]
