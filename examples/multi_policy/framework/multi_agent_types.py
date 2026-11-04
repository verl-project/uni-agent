"""Types exposed to runners that execute a group of role sessions."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class SessionGroupHandle:
    """Identify a session group and expose the API root for each role."""

    session_group_id: str
    role_base_urls: Mapping[str, str]
