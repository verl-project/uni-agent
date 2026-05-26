"""One-call registration with vLLM's KVConnectorFactory."""
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

CONNECTOR_NAME = "MooncakeKVConnector"
_MODULE_PATH = "llm_router.connector.connector"
_CLASS_NAME = "MooncakeKVConnector"


def register_with_vllm() -> None:
    """Register MooncakeKVConnector with vLLM. Safe to call multiple times."""
    if CONNECTOR_NAME in KVConnectorFactory._registry:
        return
    KVConnectorFactory.register_connector(
        name=CONNECTOR_NAME,
        module_path=_MODULE_PATH,
        class_name=_CLASS_NAME,
    )
