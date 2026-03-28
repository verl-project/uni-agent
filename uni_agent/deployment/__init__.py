from typing import Annotated, TypeAlias

from pydantic import Field

from .local.deployment import LocalDeploymentConfig
from .vefaas.deployment import VefaasDeploymentConfig

DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig | LocalDeploymentConfig,
    Field(discriminator="type"),
]

__all__ = [
    "DeployConfig",
    "LocalDeploymentConfig",
    "VefaasDeploymentConfig",
]
