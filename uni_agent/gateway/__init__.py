__all__ = [
    "GatewayActor",
    "GatewayManager",
]


def __getattr__(name: str):
    """Keep lightweight KV OOT modules importable without Ray/FastAPI."""
    if name == "GatewayActor":
        from uni_agent.gateway.gateway import GatewayActor

        return GatewayActor
    if name == "GatewayManager":
        from uni_agent.gateway.manager import GatewayManager

        return GatewayManager
    raise AttributeError(name)
