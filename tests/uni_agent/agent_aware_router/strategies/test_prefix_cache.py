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

"""Prefix checkpoints work with providers that return detached copies."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental
from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("rid", ["r", ""])
def test_incremental_checkpoint_is_written_back_to_copying_provider(rid):
    class CopyingRequests:
        def __init__(self):
            self.rows = {}
            self.writes = 0

        def get_prefix_cache(self, request_id):
            return deepcopy(self.rows.get(request_id))

        def set_prefix_cache(self, request_id, value):
            self.rows[request_id] = deepcopy(value)
            self.writes += 1

    requests = CopyingRequests()
    store = SimpleNamespace(kv=SimpleNamespace(get_block_size=lambda: 2), requests=requests)
    for size in (2, 4, 6, 2):
        prompt = list(range(size))
        expected, parent = get_prefix_hashes_incremental(prompt, 2, 0, 0)
        assert resolve_prefix_hashes(prompt, rid, store) == [str(h) for h in expected]
        assert requests.rows[rid] == {"hash_strs": [str(h) for h in expected], "parent_hash": parent}
    assert requests.writes == 4
