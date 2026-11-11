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

"""Bounded prefix identities reconstructed through an injected source codec.

The KV store serializes access. Identities describe token positions; each
ReplicaCacheIndex separately records which published blocks remain resident.
"""

import struct
from collections import OrderedDict
from dataclasses import dataclass

from ..types.kv_cache import KVCacheEvent
from ..types.native_hash import NativeHashCodec
from ..utils.hash import compute_hash


@dataclass(frozen=True)
class PrefixIdentity:
    digest: bytes
    local_hash: str
    tokens: int


class PrefixIdentityIndex:
    def __init__(self, codec: NativeHashCodec, *, history_limit=262144):
        self.codec = codec
        self.unit = codec.unit
        self.root = codec.root
        self.history_limit = history_limit
        self.identities: OrderedDict[str, PrefixIdentity] = OrderedDict()
        self.identity_revision = 0
        self.enabled = True

    def resolve_event(self, event: KVCacheEvent, *, allow_reconstruct: bool = True):
        if not self.enabled:
            return None
        known = [self.identities.get(remote) for remote in event.block_hashes]
        if all(known):
            return known
        if not allow_reconstruct:
            return None
        parent = self.identities.get(event.parent_block_hash) if event.parent_block_hash else None
        if event.parent_block_hash is not None and parent is None:
            return None
        tokens = event.raw_token_ids
        if not tokens or len(tokens) % self.unit:
            return None
        position = parent.tokens if parent else 0
        digest = parent.digest if parent else self.root
        local = int(parent.local_hash) if parent else 0
        chain = {}
        units = []
        published = set(event.block_hashes)
        for offset in range(0, len(tokens), self.unit):
            part = tokens[offset : offset + self.unit]
            digest = self.codec.hash_block(digest, part)
            local = compute_hash(local, struct.pack(f">{self.unit}I", *part))
            position += self.unit
            identity = PrefixIdentity(digest, str(local), position)
            units.append(identity)
            chain[self.codec.external_key(digest)] = identity
        if event.block_size > self.unit and event.block_size % self.unit == 0:
            # Only accept coarse encodings verified by the source codec. An
            # alias points to its final unit digest for parent continuation.
            aliases = self.codec.coarse_keys([identity.digest for identity in units], event.block_size)
            for remote, index in aliases.items():
                if remote in published:
                    chain[remote] = units[index]
        resolved = [chain.get(remote) for remote in event.block_hashes]
        if not all(resolved):
            self.enabled = False
            return None
        for remote, identity in chain.items():
            if remote not in self.identities:
                self.identity_revision += 1
            self.identities[remote] = identity
            self.identities.move_to_end(remote)
        while len(self.identities) > self.history_limit:
            self.identities.popitem(last=False)
        return resolved
