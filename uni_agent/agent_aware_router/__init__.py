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

"""KV-cache-aware LLM Router."""

import logging

from uni_agent.logging.handlers import _mount
from uni_agent.logging.session import _setup_console_logging

from .balancer import KVCAwareBalancer

# Bootstrap the uni_agent namespace only when nothing has configured it yet
if _mount().level == logging.NOTSET and not _mount().handlers:
    _setup_console_logging()

__all__ = ["KVCAwareBalancer"]
