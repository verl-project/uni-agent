"""KV-cache-aware LLM Router configuration and routing primitives."""

from uni_agent.llm_router.collectors import MetricKey, RouteDataProvider
from uni_agent.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from uni_agent.llm_router.strategies import (
    KVCacheAwareStrategy,
    ReplicaInfo,
    RoutingStrategy,
    StrategyError,
    StrategyRegistry,
    route,
)

__all__ = [
    "CacheStoreConfig",
    "CollectorConfig",
    "ConfigError",
    "KVCAwareConfig",
    "KVCAwareStrategyConfig",
    "StrategyConfig",
    "KVCacheAwareStrategy",
    "RoutingStrategy",
    "StrategyError",
    "StrategyRegistry",
    "route",
    "MetricKey",
    "ReplicaInfo",
    "RouteDataProvider",
]
