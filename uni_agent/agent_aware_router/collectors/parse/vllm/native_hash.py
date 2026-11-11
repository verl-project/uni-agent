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

"""Use installed vLLM hash and wire encodings without keeping identity state."""

from __future__ import annotations

import logging
import pickle
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMHashCodec:
    """Native operations bound to an engine-published, compatible hash profile."""

    unit: int
    root: bytes
    profile_key: Hashable
    _hash_block_tokens: Callable
    _sha256: Callable
    _maybe_convert_block_hash: Callable
    _block_hash_list: Callable

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any], local_unit: int) -> VLLMHashCodec | None:
        """Disable optional reconstruction when the source/native APIs disagree."""
        if not isinstance(metadata, Mapping):
            return None
        config = metadata.get("native_hash_config")
        if not isinstance(config, Mapping):
            return None
        version = config.get("vllm_version")
        match = re.match(r"^(\d+)\.(\d+)\.", version) if isinstance(version, str) else None
        if (
            type(config.get("schema_version")) is not int
            or config["schema_version"] != 1
            or config.get("algorithm") != "sha256"
            or type(config.get("unit")) is not int
            or config["unit"] <= 0
            or config["unit"] != local_unit
            or type(config.get("pickle_protocol")) is not int
            or config["pickle_protocol"] != pickle.HIGHEST_PROTOCOL
            or type(config.get("int_hashes")) is not bool
            or type(config.get("lookahead_tokens")) is not int
            or config["lookahead_tokens"] != 0
            or match is None
            or tuple(map(int, match.groups())) < (0, 20)
        ):
            return None
        root_hex = config.get("root_hash_hex")
        if not isinstance(root_hex, str):
            return None
        try:
            root = bytes.fromhex(root_hex)
        except ValueError:
            return None
        if len(root) != 32:
            return None
        try:
            import vllm
            import vllm.envs as envs
            from vllm.utils.hashing import sha256
            from vllm.v1.core.kv_cache_utils import (
                BlockHashListWithBlockSize,
                hash_block_tokens,
                maybe_convert_block_hash,
            )

            local_version = vllm.__version__
            local_int_hashes = envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES
        except (ImportError, AttributeError) as exc:
            logger.warning("Native vLLM hash reconstruction unavailable: %s", exc)
            return None
        if version != local_version or config["int_hashes"] != local_int_hashes:
            return None
        operations = (hash_block_tokens, sha256, maybe_convert_block_hash, BlockHashListWithBlockSize)
        if not all(callable(operation) for operation in operations):
            return None
        profile_key = (
            config["schema_version"],
            version,
            config["algorithm"],
            root,
            config["unit"],
            config["pickle_protocol"],
            config["int_hashes"],
            config["lookahead_tokens"],
        )
        return cls(
            config["unit"],
            root,
            profile_key,
            hash_block_tokens,
            sha256,
            maybe_convert_block_hash,
            BlockHashListWithBlockSize,
        )

    def hash_block(self, parent: bytes, tokens: Sequence[int]) -> bytes:
        return bytes(self._hash_block_tokens(self._sha256, parent, tokens, None))

    def external_key(self, digest: bytes) -> str:
        # The wire decoder normalizes both bytes and integer identities with str.
        return str(self._maybe_convert_block_hash(digest))

    def coarse_keys(self, digests: Sequence[bytes], block_size: int) -> Mapping[str, int]:
        if block_size < self.unit or block_size % self.unit:
            raise ValueError("Coarse block size must be a positive multiple of the hash unit")
        scale = block_size // self.unit
        native = self._block_hash_list(list(digests), self.unit, block_size)
        return {self.external_key(digest): (index + 1) * scale - 1 for index, digest in enumerate(native)}
