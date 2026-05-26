"""Plan B: vLLM v1 KV connector backed by Mooncake.

Importing this package does not register with vLLM automatically — call
`register_with_vllm()` once during trainer startup. This keeps the side
effect explicit (vLLM does not otherwise import this module).
"""
from llm_router.connector.connector import MooncakeKVConnector
from llm_router.connector.prefix_hash import (
    VersionedKey,
    make_versioned_key,
)
from llm_router.connector.registry import (
    CONNECTOR_NAME,
    register_with_vllm,
)

__all__ = [
    "CONNECTOR_NAME",
    "MooncakeKVConnector",
    "VersionedKey",
    "make_versioned_key",
    "register_with_vllm",
]
