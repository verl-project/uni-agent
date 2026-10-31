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

"""Native SHA256 prefix identities reconstructed from Full cache events.

The KV store serializes access. Identities describe token positions; each
GroupCache separately records which published blocks remain resident.
"""

import hashlib
import pickle
import struct
from collections import OrderedDict
from dataclasses import dataclass

from .hash import compute_hash


@dataclass(frozen=True)
class NativeIdentity:
    digest: bytes
    local_hash: str
    tokens: int


class NativeHashIndex:
    def __init__(self, config, *, history_limit=262144):
        self.unit = config["unit"]
        self.protocol = config["pickle_protocol"]
        self.int_hashes = config["int_hashes"]
        self.root = self._hash(config["seed"])
        self.history_limit = history_limit
        self.identities: OrderedDict[str, NativeIdentity] = OrderedDict()
        self.identity_revision = 0
        self.enabled = True

    @classmethod
    def from_metadata(cls, metadata, local_unit):
        config = metadata.get("native_hash_config")
        if not config or config.get("algorithm") != "sha256" or config.get("unit") != local_unit:
            return None
        seed = config.get("seed")
        if (
            not isinstance(seed, str)
            or not seed.isdecimal()
            or not 0 <= int(seed) < 2**32
            or config.get("pickle_protocol") not in (4, 5)
            or type(config.get("int_hashes")) is not bool
        ):
            return None
        return cls(config)

    def _hash(self, value):
        return hashlib.sha256(pickle.dumps(value, protocol=self.protocol)).digest()

    def _remote(self, digest):
        return str(int.from_bytes(digest, "big") & ((1 << 64) - 1)) if self.int_hashes else str(digest)

    def resolve_event(self, event):
        if not self.enabled:
            return None
        known = [self.identities.get(remote) for remote in event.block_hashes]
        if all(known):
            return known
        if event.kind not in {"full_attention", "mla_attention", "sink_full_attention"}:
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
        for offset in range(0, len(tokens), self.unit):
            part = tokens[offset : offset + self.unit]
            digest = self._hash((digest, tuple(part), None))
            local = compute_hash(local, struct.pack(f">{self.unit}I", *part))
            position += self.unit
            chain[self._remote(digest)] = NativeIdentity(digest, str(local), position)
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
