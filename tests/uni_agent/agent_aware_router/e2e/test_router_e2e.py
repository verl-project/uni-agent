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

"""GPU end-to-end coverage for agent-aware routing."""

from __future__ import annotations

import os

import pytest

from .utils.router_driver import assert_router_signals, run_infer

# CUDA_VISIBLE_DEVICES controls which GPUs Ray/vLLM see; --n-gpus-per-node
# must match the count.
_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")


@pytest.mark.gpu
@pytest.mark.level1
def test_kvc_aware_router_full_e2e():
    log = run_infer(
        device="gpu",
        visible_devices=_VISIBLE,
        tensor_parallel_size=2,
        log_name="router_e2e.log",
    )
    assert_router_signals(log)
