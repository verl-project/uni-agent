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

"""CPU regressions for the simulated runner and GPU test completion checks."""

import asyncio
import json

import httpx
import pytest

from tests.uni_agent.agent_aware_router.e2e import test_router_e2e
from tests.uni_agent.agent_aware_router.e2e.utils import simulated_sandbox
from uni_agent.gateway.session import SessionHandle

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize(
    "arguments,expected_observation",
    [
        ('"not-an-object"', None),
        ("[]", None),
        ("null", None),
        ("7", None),
        ("invalid-json", None),
        (json.dumps({"command": "echo '<<<Finished>>>'"}), "<<<Finished>>>"),
    ],
    ids=["string", "list", "null", "number", "malformed", "object"],
)
def test_simulated_session_handles_tool_arguments(monkeypatch, arguments, expected_observation):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "first" if first else "final",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_bash" if first else "submit",
                                        "arguments": arguments if first else "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    session = simulated_sandbox._SimulatedSession(
        SessionHandle("session", "http://gateway.test/session/v1"),
        simulated_sandbox._ObservationSampler(seed=0, scale=1),
        "Resolve the task.",
        max_turns=3,
    )

    turns = asyncio.run(session.run())

    assert turns == len(requests) == 2
    assert session._messages[-1]["content"] == "Submitted."
    observation = requests[1]["messages"][-1]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "first"
    if expected_observation is not None:
        assert observation["content"] == expected_observation


def _inference_log(scored, failed):
    return "\n".join(
        [
            "route(): replicas=2",
            "CAPACITY_TOKEN_AWARE load_threshold=0.8",
            f"generate_sequences summary: num_success_sessions={scored} num_failed_sessions={failed}",
            "--------------- inference summary ----------------",
            f"  mean rm_score        1.0000   (over {scored} sessions)",
            f"  scored sessions       {scored} / 8    (4 prompts x n=2)",
        ]
    )


@pytest.mark.parametrize("scored,failed", [(7, 0), (8, 1)])
def test_gpu_test_rejects_incomplete_or_failed_sessions(monkeypatch, scored, failed):
    monkeypatch.setattr(test_router_e2e, "_run_infer", lambda: _inference_log(scored, failed))

    with pytest.raises(AssertionError):
        test_router_e2e.TestKVCAwareRouterE2E().test_kvc_aware_router_full_e2e()


def test_gpu_test_accepts_all_eight_sessions_without_failures(monkeypatch):
    monkeypatch.setattr(test_router_e2e, "_run_infer", lambda: _inference_log(8, 0))

    test_router_e2e.TestKVCAwareRouterE2E().test_kvc_aware_router_full_e2e()
