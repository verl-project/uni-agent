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

"""Backend-neutral operations for reconstructing published prefix identities."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol


class NativeHashCodec(Protocol):
    """A stateless, validated source hash profile and its encoding operations."""

    unit: int
    root: bytes
    profile_key: Hashable

    def hash_block(self, parent: bytes, tokens: Sequence[int]) -> bytes:
        """Hash one unit with an explicit initialized parent digest."""
        ...

    def external_key(self, digest: bytes) -> str:
        """Encode a digest as the normalized identity published by this source."""
        ...

    def coarse_keys(self, digests: Sequence[bytes], block_size: int) -> Mapping[str, int]:
        """Map published coarse identities to their final zero-based unit index."""
        ...


NativeHashCodecFactory = Callable[[Mapping[str, Any], int], NativeHashCodec | None]
