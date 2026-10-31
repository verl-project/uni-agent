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

"""Cache lifecycle state and event transport frames."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CacheLifecycleState:
    instance_id: str
    epoch: int = 0
    revision: int = 0
    serving: bool = True
    index_state: str = "ready"
    reason: str | None = None
    expected_clears: int = 0
    observed_clears: int = 0
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "CacheLifecycleState":
        state = cls(**value)
        if state.schema_version != 1 or not isinstance(state.instance_id, str) or not state.instance_id:
            raise ValueError("unsupported cache lifecycle identity")
        for count in (state.epoch, state.revision, state.expected_clears, state.observed_clears):
            if type(count) is not int or count < 0:
                raise ValueError("invalid cache lifecycle counter")
        if type(state.serving) is not bool or state.index_state not in {"ready", "awaiting_clear", "unknown"}:
            raise ValueError("invalid cache lifecycle state")
        return state


@dataclass(frozen=True)
class KVEventFrame:
    payload: bytes
    sequence: int | None = None
    error: str | None = None
    reset_stream: bool = False


BRIDGE_SCHEMA_KEY = "uni_agent_kv_bridge"
