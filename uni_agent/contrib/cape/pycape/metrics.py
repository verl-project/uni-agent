# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from .journal import EventRecord


class NoopMetricsSink:
    def record_event(self, event: EventRecord) -> None:
        return None


class MemoryMetricsSink:
    def __init__(self) -> None:
        self.events: List[EventRecord] = []

    def record_event(self, event: EventRecord) -> None:
        self.events.append(event)


class FailingMetricsSink:
    def record_event(self, event: EventRecord) -> None:
        raise RuntimeError("metrics exporter failed")
