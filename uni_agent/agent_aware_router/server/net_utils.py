# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consecutive-port allocation for the kv-events ZMQ endpoints."""

import ipaddress
import socket


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _bind_tcp(address: str, port: int, reuse: bool) -> socket.socket:
    """Bind a TCP socket to ``address:port`` and return it.

    ``reuse`` toggles ``SO_REUSEADDR``. The socket is closed and the error
    re-raised on bind failure, so a failed attempt leaks no descriptor.
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    if reuse:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((address, port))
    except OSError:
        sock.close()
        raise
    return sock


def get_free_port_range(
    address: str, count: int, with_alive_sock: bool = False
) -> tuple[int, list[socket.socket] | None]:
    """Find ``count`` consecutive free ports, optionally holding them open.

    Binds a random anchor (port 0) plus its ``count-1`` neighbors, retrying up
    to 64 times on conflict. Sockets intentionally skip ``SO_REUSEADDR``: Linux
    lets bind-only sockets share a port when both sides set it (ZMQ does), so a
    reservation made with it would not block a later ZMQ bind. With
    ``with_alive_sock=True`` the caller holds the sockets and must close them
    before the target service binds, or that bind fails with ``EADDRINUSE``.
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    for _ in range(64):
        socks: list[socket.socket] = []
        try:
            socks.append(_bind_tcp(address, 0, reuse=False))
            base_port = socks[0].getsockname()[1]
            for offset in range(1, count):
                socks.append(_bind_tcp(address, base_port + offset, reuse=False))
        except OSError:
            for s in socks:
                s.close()
            continue
        if with_alive_sock:
            return base_port, socks
        for s in socks:
            s.close()
        return base_port, None
    raise RuntimeError(f"failed to find {count} consecutive free ports on {address}")
