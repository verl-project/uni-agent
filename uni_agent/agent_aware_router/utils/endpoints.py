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

"""Format replica addresses consistently for HTTP and ZMQ."""


def is_ipv6_host(host: str) -> bool:
    """Recognize an IPv6 host, including an already bracketed literal."""
    return ":" in host


def format_host_port(host: str, port: int | str) -> str:
    """Bracket IPv6 literals once, leaving IPv4 and DNS names unchanged."""
    if is_ipv6_host(host) and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"
