from importlib import import_module

from .config import (
    DeployConfig,
    HostDeploymentConfig,
    LocalAttachDeploymentConfig,
    LocalDeploymentConfig,
    LocalNativeDeploymentConfig,
    SimulatedDeploymentConfig,
    ModalDeploymentConfig,
    VefaasDeploymentConfig,
)

_LAZY_EXPORTS = {
    "HostDeployment": ".host.deployment",
    "LocalAttachDeployment": ".local_attach.deployment",
    "LocalDeployment": ".local.deployment",
    "LocalNativeDeployment": ".local_native.deployment",
    "SimulatedDeployment": ".simulated.deployment",
    "ModalDeployment": ".modal.deployment",
    "VefaasDeployment": ".vefaas.deployment",
}

__all__ = [
    "DeployConfig",
    "HostDeploymentConfig",
    "LocalAttachDeploymentConfig",
    "LocalDeploymentConfig",
    "LocalNativeDeploymentConfig",
    "SimulatedDeploymentConfig",
    "ModalDeploymentConfig",
    "VefaasDeploymentConfig",
    "HostDeployment",
    "LocalAttachDeployment",
    "LocalDeployment",
    "LocalNativeDeployment",
    "SimulatedDeployment",
    "ModalDeployment",
    "VefaasDeployment",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
