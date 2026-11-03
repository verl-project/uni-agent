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

"""Manager cleanup attempts every collector and retains failed cleanup for retry."""

import logging

import pytest

from uni_agent.agent_aware_router.collectors import provider as provider_module
from uni_agent.agent_aware_router.collectors.provider import CollectorManager
from uni_agent.agent_aware_router.config.collector import CollectorConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("failed_indices", [(0,), (0, 2)], ids=["one-failure", "multiple-failures"])
def test_stop_attempts_every_collector_and_can_retry_after_failures(monkeypatch, caplog, failed_indices):
    calls = []
    failures = {index: ValueError(f"cleanup {index} failed") for index in failed_indices}

    class StopOnceCollector:
        def __init__(self, index):
            self.index = index
            self.attempts = 0

        def stop(self):
            calls.append(self.index)
            self.attempts += 1
            if self.attempts == 1 and self.index in failures:
                raise failures[self.index]

    collectors = {str(index): StopOnceCollector(index) for index in range(3)}
    monkeypatch.setattr(provider_module, "get_collector", lambda name, *args, **kwargs: collectors[name])
    manager = CollectorManager(CollectorConfig(), list(collectors))

    with caplog.at_level(logging.ERROR, logger=provider_module.__name__):
        with pytest.raises(RuntimeError, match=rf"Failed to stop {len(failed_indices)} collector") as raised:
            manager.stop()

    assert calls == [0, 1, 2]
    assert raised.value.__cause__ is failures[failed_indices[0]]
    error_records = [record for record in caplog.records if record.name == provider_module.__name__]
    assert len(error_records) == len(failures)
    assert [record.exc_info[1] for record in error_records] == list(failures.values())

    manager.stop()
    assert calls == [0, 1, 2, 0, 1, 2]
    assert all(collector.attempts == 2 for collector in collectors.values())
