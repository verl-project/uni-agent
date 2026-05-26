"""Abstract base for request → replica routing policies."""
from abc import ABC, abstractmethod
from typing import Any


class RouterPolicy(ABC):
    """Policy that picks a replica server for an incoming request.

    Sub-classes implement three methods:
    - `acquire_server`: choose a server, increment its in-flight counter.
    - `release_server`: decrement a server's in-flight counter on completion.
    - `report_prefixes`: record that a server has been observed handling
      these prefix signatures (used by context-aware policies; legacy
      policies can implement this as a no-op).
    """

    def __init__(self, server_ids: list[str]):
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        self.server_ids = list(server_ids)

    @abstractmethod
    def acquire_server(self, request_id: str, **kwargs: Any) -> str:
        """Return the server id chosen for this request. Implementations
        SHOULD increment any internal in-flight counter for that server.
        """

    @abstractmethod
    def release_server(self, server_id: str) -> None:
        """Decrement the in-flight counter for the given server."""

    @abstractmethod
    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
        *,
        tier: str = "gpu",
    ) -> None:
        """Tell the policy that `server_id` has just processed a request
        whose prompt yields these prefix signatures. Policies that don't
        do prefix-aware routing implement this as a no-op.

        Each tuple is `(weight_version, prefix_hash, prefix_len)`.
        `tier` is `"gpu"` for local HBM prefix cache and `"cpu"` for
        Mooncake-backed local CPU placement.
        """
