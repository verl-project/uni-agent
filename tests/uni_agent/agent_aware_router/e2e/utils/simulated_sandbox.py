"""Simulated sandbox for agent_aware_router e2e tests — no real container involved.

Rewrites the uni-agent-sglang ``SimulatedRuntime`` asset onto the current
``AgentRunner`` architecture: instead of stubbing swerex's AbstractRuntime,
this module provides :func:`simulated_runner`, an AgentRunner-protocol
callable that drives the gateway session directly and answers tool calls
with canned observations sampled from ``observations.yaml`` (ported from the
sglang repo, hand-tuned weights preserved).

The LLM (vLLM replicas + KVCAware router) runs for real; only the sandbox
side of the agent loop is simulated. That keeps the e2e assertions —
``routed to server`` / ``COMBINED`` / ``mean rm_score`` / trajectory
output — meaningful while dropping the container dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx

from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks.base import TaskResult

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_TEMPLATES_PATH = Path(__file__).with_name("observations.yaml")

# Command -> route key. Order matters: install-phase commands must be
# classified before generic python; editor subcommands are split out so each
# gets its own template pool. Ported verbatim from the sglang SimulatedRuntime.
_ROUTE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("finish", re.compile(r"""echo\s+['"]<<<Finished>>>['"]""")),
    ("install", re.compile(r"^(which|export|chmod|mkdir|pip|pip3)\b")),
    ("install", re.compile(r"\bpip3?\s+install\b")),
    ("install", re.compile(r"\bpython\d?\s+-m\s+pip\b")),
    ("editor:view", re.compile(r"^str_replace_editor\b.*--command\s+view\b")),
    ("editor:create", re.compile(r"^str_replace_editor\b.*--command\s+create\b")),
    ("editor:str_replace", re.compile(r"^str_replace_editor\b.*--command\s+str_replace\b")),
    ("editor:insert", re.compile(r"^str_replace_editor\b.*--command\s+insert\b")),
    ("editor:undo_edit", re.compile(r"^str_replace_editor\b.*--command\s+undo_edit\b")),
    ("test_output", re.compile(r"^(\S*python\S*\s+-m\s+pytest\b|^pytest\b)")),
    ("python_script", re.compile(r"^python\d?\s")),
    ("listing", re.compile(r"^(find|ls)\b")),
    ("search", re.compile(r"^grep\b")),
    ("file_view", re.compile(r"^(cat|head|tail)\b")),
]

_FINISH_OUTPUT = "<<<Finished>>>"


def _load_templates() -> dict[str, list[tuple[int, str]]]:
    """Load route_key -> [(weight, text)] from the yaml next to this module."""
    import yaml

    raw = yaml.safe_load(_TEMPLATES_PATH.read_text())
    return {key: [(int(e["weight"]), e["text"]) for e in entries] for key, entries in raw.items()}


class _ObservationSampler:
    """Weighted canned-observation sampler with optional per-run seed."""

    def __init__(self, seed: int | None, scale: float):
        self._seed = seed
        self._scale = scale
        self._templates = _load_templates()
        self._rngs: dict[str, random.Random] = {}
        for key, pool in self._templates.items():
            rng_seed = self._derive_seed(key)
            self._rngs[key] = random.Random(rng_seed)
            # Burn one draw per key so distinct keys with equal seeds diverge.
            self._rngs[key].random()

    def _derive_seed(self, key: str) -> int | None:
        if self._seed is None:
            return None
        digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def route(self, command: str) -> str:
        stripped = command.strip()
        for key, pattern in _ROUTE_RULES:
            if pattern.search(stripped):
                return key
        return "default"

    def render(self, route_key: str) -> str:
        if route_key == "finish":
            return _FINISH_OUTPUT
        if route_key == "install":
            return ""
        pool = self._templates.get(route_key) or self._templates.get("default") or [(1, "")]
        weights = [w for w, _ in pool]
        texts = [t for _, t in pool]
        rng = self._rngs.get(route_key) or random.Random(self._derive_seed(route_key))
        text = rng.choices(texts, weights=weights, k=1)[0]
        if self._scale != 1.0 and text:
            text = text * max(1, int(self._scale))
        return text


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a bash command in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _extract_task(raw_prompt: object) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        return next(
            (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
    return str(raw_prompt)


class _SimulatedSession:
    """Agent loop against one gateway session with a canned sandbox."""

    def __init__(
        self,
        session: SessionHandle,
        sampler: _ObservationSampler,
        task: str,
        *,
        max_turns: int,
    ):
        self._session = session
        self._sampler = sampler
        self._max_turns = max_turns
        self._messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a helpful coding agent. Use the tools to fix the issue, then submit.",
            },
            {"role": "user", "content": task or "Resolve the task."},
        ]

    async def run(self) -> int:
        """Drive the loop; return the number of turns executed."""
        turns = 0
        async with httpx.AsyncClient(timeout=None) as client:
            for _ in range(self._max_turns):
                turns += 1
                response = await client.post(
                    f"{self._session.base_url}/chat/completions",
                    json={"messages": self._messages, "tools": _TOOLS},
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                self._messages.append(message)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    self._messages.append(
                        {"role": "tool", "tool_call_id": "none", "content": "No tool call; continue with tools."}
                    )
                    continue
                finished = False
                for call in tool_calls:
                    name = call["function"]["name"]
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    if name == "submit":
                        observation = "Submitted."
                        finished = True
                    else:
                        route = self._sampler.route(str(arguments.get("command", "")))
                        observation = self._sampler.render(route)
                    self._messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                if finished:
                    break
        return turns


async def simulated_runner(
    *,
    session: SessionHandle,
    raw_prompt: object,
    sample_index: int,
    seed: int | None = None,
    observation_scale: float = 1.0,
    max_turns: int = 8,
    reward_score: float = 1.0,
    **_: Any,
) -> TaskResult:
    """AgentRunner-protocol callable backed by the simulated sandbox.

    Wire from yaml::

        agent_runners:
          swe_agent:
            runner_fqn: tests.uni_agent.agent_aware_router.e2e.utils.simulated_sandbox.simulated_runner
            runner_kwargs: {max_turns: 8, reward_score: 1.0}
    """
    sampler = _ObservationSampler(seed, observation_scale)
    loop = _SimulatedSession(
        session,
        sampler,
        _extract_task(raw_prompt),
        max_turns=max_turns,
    )
    turns = await loop.run()

    # The framework now owns reward annotation and TransferQueue writes.
    return TaskResult(reward=reward_score, extra_info={"turns": turns})
