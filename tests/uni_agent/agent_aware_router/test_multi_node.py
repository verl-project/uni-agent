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

"""Test worker environment forwarding without executing the cluster launcher."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def worker_exec_function() -> str:
    script = Path(__file__).resolve().parents[3] / "examples/agent_aware_router/multi-node.sh"
    # Extract only this definition, never source the launcher's cluster cleanup/main.
    match = re.search(r"^worker_exec_with_env\(\) \{\n.*?^\}", script.read_text(), re.MULTILINE | re.DOTALL)
    assert match is not None, "worker_exec_with_env definition not found"
    return match.group()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("device", ["gpu", "ascend"])
@pytest.mark.parametrize(
    ("level", "expected", "warns"),
    [
        pytest.param(None, "basic", False, id="unset"),
        pytest.param("basic", "basic", False, id="basic"),
        pytest.param("debug", "debug", False, id="debug"),
        pytest.param("verbose", "basic", True, id="invalid"),
        pytest.param("", "basic", False, id="empty"),
        pytest.param("basic; printf injected #", "basic", True, id="semicolon"),
        pytest.param("$(printf injected)", "basic", True, id="command-substitution"),
        pytest.param("`printf injected`", "basic", True, id="backticks"),
    ],
)
def test_worker_exec_with_env_forwards_validated_level(worker_exec_function, device, level, expected, warns):
    # Do not inherit credentials, launcher settings, or BASH_ENV from the caller.
    env = {
        "PATH": os.defpath,
        "DEVICE": device,
        "WORKER_CONTAINER": "test-worker-container",
        "RL_INSIGHT_PORT": "8123",
        "OPENYUANRONG_SERVER_ADDRESS": "http://sandbox.invalid",
        "OPENYUANRONG_TOKEN": "test-token",
    }
    if level is not None:
        env["VERL_RL_INSIGHT_LEVEL"] = level
    bash = shutil.which("bash")
    assert bash is not None, "bash is required for the launcher regression test"
    result = subprocess.run(
        [
            bash,
            "--noprofile",
            "--norc",
            "-c",
            worker_exec_function
            + r"""
set -euo pipefail
ssh() { printf '%s\0' "$@"; }
log() { printf '%s\n' "$*" >&2; }
worker_exec_with_env test-worker 192.0.2.10 ray status
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    # ssh is a recording stub: the remote command is inspected, never executed.
    ssh_args = result.stdout.split("\0")
    assert ssh_args[-1] == ""
    assert ssh_args[:3] == ["-o", "StrictHostKeyChecking=no", "test-worker"]
    assert len(ssh_args) == 5
    remote_command = ssh_args[3]
    docker_args = shlex.split(remote_command)
    assert docker_args[:2] == ["docker", "exec"]
    assert docker_args[-3:] == ["test-worker-container", "ray", "status"]
    env_args = docker_args[2:-3]
    assert len(env_args) % 2 == 0
    assert env_args[::2] == ["-e"] * (len(env_args) // 2)
    forwarded = dict(arg.split("=", 1) for arg in env_args[1::2])
    assert len(forwarded) == len(env_args) // 2
    expected_env = {
        "VERL_LOGGING_LEVEL": "INFO",
        "VERL_RL_INSIGHT_ENABLE": "1",
        "VERL_RL_INSIGHT_LEVEL": expected,
        "RL_INSIGHT_SERVER_URL": "http://192.0.2.10:8123",
        "OPENYUANRONG_SERVER_ADDRESS": "http://sandbox.invalid",
        "OPENYUANRONG_TOKEN": "test-token",
        "OPENYUANRONG_TUNNEL_SSL_VERIFY": "0",
        "SANDBOX_NAME_PREFIX": "mini-swe-",
        "PYTHONHASHSEED": "0",
    }
    if device == "ascend":
        expected_env.update(
            HCCL_IF_BASE_PORT="26100",
            RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="1",
            ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        )
    assert forwarded == expected_env
    warnings = ["WARNING: invalid VERL_RL_INSIGHT_LEVEL; using basic"] if warns else []
    assert result.stderr.splitlines() == warnings
    if warns:
        assert level not in remote_command
