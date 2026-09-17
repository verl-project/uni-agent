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

"""Consecutive-port-block allocation for kv-events endpoints (CPU-only)."""

import socket

import pytest

from uni_agent.agent_aware_router.server.net_utils import get_free_port_range

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def _bind(addr: str, port: int, reuse: bool = False) -> socket.socket:
    family = socket.AF_INET
    s = socket.socket(family=family, type=socket.SOCK_STREAM)
    if reuse:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((addr, port))
    return s


@pytest.mark.parametrize("count", [1, 4])
def test_get_free_port_range_returns_consecutive_ports(count):
    base, socks = get_free_port_range("0.0.0.0", count, with_alive_sock=True)
    try:
        assert len(socks) == count
        assert all(s.getsockname()[1] == base + i for i, s in enumerate(socks))
    finally:
        for s in socks:
            s.close()


def test_get_free_port_range_held_blocks_others():
    base, socks = get_free_port_range("0.0.0.0", 2, with_alive_sock=True)
    try:
        # While held, a second bind must fail — including one with SO_REUSEADDR
        # set (Linux lets bind-only TCP sockets share a port when BOTH sides set
        # it, and ZMQ binds with it set, so the reservation must not use it).
        with pytest.raises(OSError):
            _bind("0.0.0.0", base, reuse=True)
        with pytest.raises(OSError):
            _bind("0.0.0.0", base + 1)
    finally:
        for s in socks:
            s.close()


@pytest.mark.parametrize("with_alive_sock", [True, False])
def test_get_free_port_range_release_unblocks(with_alive_sock):
    base, socks = get_free_port_range("0.0.0.0", 2, with_alive_sock=with_alive_sock)
    if socks is not None:
        for s in socks:
            s.close()
    # After release the port is bindable again, with and without SO_REUSEADDR
    # (vLLM's ZMQ bind is the real consumer and binds with it set).
    _bind("0.0.0.0", base).close()
    _bind("0.0.0.0", base, reuse=True).close()


def test_get_free_port_range_skips_occupied_neighbor():
    # Occupy an ephemeral port, then ask for a range that must avoid it.
    blocker = _bind("0.0.0.0", 0)
    occupied = blocker.getsockname()[1]
    try:
        # Requesting 2 consecutive ports; the allocator must not hand back a
        # base whose block overlaps `occupied`. Since the anchor itself is
        # OS-assigned (bind 0), the only way to overlap is base==occupied or
        # base+1==occupied; just assert neither holds.
        base, socks = get_free_port_range("0.0.0.0", 2, with_alive_sock=True)
        try:
            assert occupied not in range(base, base + 2)
        finally:
            for s in socks:
                s.close()
    finally:
        blocker.close()


def test_get_free_port_range_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        get_free_port_range("0.0.0.0", 0)
    with pytest.raises(ValueError):
        get_free_port_range("0.0.0.0", -1)
