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

"""Ascend (NPU) end-to-end coverage for agent-aware routing.

Reuses the simulated sandbox (no OpenYuanrong container) and asserts the same
router telemetry as the GPU e2e. Defaults to eight NPUs with ``TP=1`` so
dispatch actually ranks across eight replicas (one 8-card Atlas A2 host);
override the device set with ``ROUTER_E2E_ASCEND_VISIBLE`` (e.g. ``0,1``).
"""

from __future__ import annotations

import os

import pytest

from .utils.router_driver import assert_router_signals, run_infer

_VISIBLE = os.environ.get("ROUTER_E2E_ASCEND_VISIBLE", "0,1,2,3,4,5,6,7")
_TENSOR_PARALLEL_SIZE = 1


def _num_replicas(visible: str, tensor_parallel_size: int) -> int:
    return len([v for v in visible.split(",") if v.strip()]) // tensor_parallel_size


@pytest.mark.npu
@pytest.mark.level0
@pytest.mark.card_eight
def test_kvc_aware_router_full_e2e_on_npu():
    log = run_infer(
        device="ascend",
        visible_devices=_VISIBLE,
        tensor_parallel_size=_TENSOR_PARALLEL_SIZE,
        log_name="router_e2e_npu.log",
    )
    # TP=1 across the visible NPUs gives one replica per device.
    assert_router_signals(log, replicas=_num_replicas(_VISIBLE, _TENSOR_PARALLEL_SIZE))
