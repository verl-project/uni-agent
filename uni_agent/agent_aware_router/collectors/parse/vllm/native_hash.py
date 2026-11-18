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

import pickle
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ....types.kv_cache import NativeHashCodecResult


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
    def from_metadata(cls, metadata: Mapping[str, Any], local_unit: int) -> NativeHashCodecResult:
        """Validate native APIs and report why optional reconstruction is unavailable."""
        if not isinstance(metadata, Mapping):
            return NativeHashCodecResult(reason="invalid_metadata")
        config = metadata.get("native_hash_config")
        if not isinstance(config, Mapping):
            return NativeHashCodecResult(reason="missing_native_hash_config")
        if config.get("codec_unavailable_reason"):
            return NativeHashCodecResult(reason=str(config["codec_unavailable_reason"]))
        for field, expected in (
            ("schema_version", 1),
            ("algorithm", "sha256"),
            ("unit", local_unit),
            ("pickle_protocol", pickle.HIGHEST_PROTOCOL),
            ("lookahead_tokens", 0),
        ):
            actual = config.get(field)
            if type(actual) is not type(expected) or actual != expected:
                return NativeHashCodecResult(reason=f"{field}_mismatch", details={"remote": actual, "local": expected})
        if type(config.get("unit")) is not int or config["unit"] <= 0:
            return NativeHashCodecResult(reason="invalid_native_hash_unit", details={"remote": config.get("unit")})
        if type(config.get("int_hashes")) is not bool:
            return NativeHashCodecResult(reason="invalid_int_hashes", details={"remote": config.get("int_hashes")})
        version = config.get("vllm_version")
        match = re.match(r"^(\d+)\.(\d+)\.", version) if isinstance(version, str) else None
        if match is None or tuple(map(int, match.groups())) < (0, 20):
            return NativeHashCodecResult(reason="unsupported_vllm_version", details={"remote": version})
        root_hex = config.get("root_hash_hex")
        try:
            root = bytes.fromhex(root_hex) if isinstance(root_hex, str) else b""
        except ValueError:
            root = b""
        if len(root) != 32:
            return NativeHashCodecResult(reason="invalid_root_hash")
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
            return NativeHashCodecResult(reason="native_api_unavailable", details={"error": str(exc)})
        for field, local in (("vllm_version", local_version), ("int_hashes", local_int_hashes)):
            if config[field] != local:
                return NativeHashCodecResult(
                    reason=f"{field}_mismatch", details={"remote": config[field], "local": local}
                )
        operations = (hash_block_tokens, sha256, maybe_convert_block_hash, BlockHashListWithBlockSize)
        if not all(callable(operation) for operation in operations):
            return NativeHashCodecResult(reason="native_api_unavailable")
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
        return NativeHashCodecResult(codec=cls(config["unit"], root, profile_key, *operations))

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
