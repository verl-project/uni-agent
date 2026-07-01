"""SWE-bench task: one problem family, solved by whichever agent you configure.

A task *selects* an agent and subclasses :class:`TaskConfig` to narrow ``agent``
and add task knobs (the default here is the white-box
:class:`~uni_agent.agents.code_act.CodeActConfig`, but a black-box agent such as
``claude_code`` works too -- both talk to the model at the gateway session's
``base_url``, which the task passes in, so it runs them the same way). ``run.py``
owns the runtime lifecycles -- mirroring the driver in
:mod:`uni_agent.framework.framework`:

* it starts the sandbox (cleaning up even if ``start`` fails) and always stops it;
* it creates a gateway session, hands its ``base_url`` / ``api_key`` plus the task
  ``messages`` to the agent, and finalizes the session for trajectories (aborting
  on error);
* it then scores what the agent produced.

Values below are illustrative defaults -- the shape is the point.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import Field

from ...agents.code_act import CodeActConfig
from ...reward import load_reward_spec
from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import reward_config


class SWEBenchTaskConfig(TaskConfig):

    run_gold_patch: bool = Field(
        default=False,
        description="Oracle mode: skip the agent and score the dataset's gold patch directly.",
    )


@register_task("swe_bench")
class SWEBenchTask(Task):
    name = "swe_bench"

    async def run(self) -> TaskResult:
        """Run one episode -- white-box or black-box -- then score it.

        ``run`` takes no arguments: the sample is :attr:`SWEBenchTaskConfig.metadata`
        and, when a model is needed, the gateway is the process-global
        :func:`~uni_agent.gateway.get_gateway_manager` the runner installed.

        If :attr:`SWEBenchTaskConfig.run_gold_patch` is set, short-circuit to an
        oracle run: score the dataset's gold patch directly, with no agent (and so
        no gateway or sandbox). It should pass all tests -- a sanity baseline.

        Otherwise the task owns every runtime lifecycle (the agent only solves the
        problem), mirroring the driver in :mod:`uni_agent.framework.framework`:

        1. **Sandbox** -- entered with ``async with``: ``start`` on enter (cleaned
           up if start fails), ``stop`` always on exit.
        2. **Gateway session** -- the task creates a session, then hands its
           ``base_url`` (+ ``api_key``) and the task ``messages`` to the agent:
           ``create_session`` -> run agent -> ``finalize_session`` for trajectories
           (``abort_session`` on error). The agent never sees the session: a
           white-box agent calls the URL from our framework loop; a black-box agent
           is launched in the sandbox pointed at the same URL.
        3. **Reward** -- score the patch the agent left in the sandbox.
        """
        cfg: SWEBenchTaskConfig = self.config  # type: ignore[assignment]
        sample = cfg.metadata  # the dataset sample now lives on the config

        # Oracle baseline: score the dataset's gold patch directly -- no agent, and
        # so no gateway or sandbox. Useful as a sanity check (it should pass).
        if cfg.run_gold_patch:
            gold_patch = sample.get("patch")
            if not gold_patch:
                raise ValueError("swe_bench: run_gold_patch=True but sample has no 'patch' (the gold patch)")
            reward_spec = load_reward_spec(reward_config())
            reward, info = await reward_spec.compute_reward(
                {"sample": sample, "patch": gold_patch, "transcript": [], "trajectories": []}
            )
            return TaskResult(reward=reward, info={"gold_patch": True, "patch": gold_patch, "eval": info})

        # Every agent drives the model through the gateway's session URL, so fetch the
        # process-global manager the runner installed (raises if none). Imported lazily
        # like build_sandbox / build_agent so a gold-patch run pulls in no gateway deps.
        from ...gateway import get_gateway_manager

        gateway = get_gateway_manager()
        agent = self.build_agent()
        session_id = f"swe-bench-{uuid4().hex}"

        # Sandbox lifecycle via async-with: __aenter__ starts it (cleaning up if
        # start fails), __aexit__ always stops it -- same guarantees as try/finally.
        async with self.build_sandbox() as sandbox:
            # Gateway session lifecycle: create -> run agent -> finalize (abort on error).
            session = await gateway.create_session(session_id)
            try:
                if session.base_url is None:
                    raise RuntimeError(f"gateway session {session_id!r} has no base_url")
                # The agent talks to the model at the session's OpenAI-compatible URL;
                # it never sees the session. The gateway accepts any non-empty api_key.
                messages = [{"role": "user", "content": sample.get("problem_statement", "")}]
                result = await agent.run(
                    sandbox=sandbox,
                    base_url=session.base_url,
                    api_key="EMPTY",
                    messages=messages,
                )
                trajectories = await gateway.finalize_session(session_id)
            except Exception:
                await gateway.abort_session(session_id)
                raise

            # Patch = reward input. Black-box agents return it in ``output``; for a
            # white-box agent the task reads it off the sandbox before stopping it.
            patch = result.output.get("patch")
            if patch is None:
                patch = (await sandbox.exec_shell("git diff")).stdout

        # Score what the agent left behind.
        reward_spec = load_reward_spec(reward_config())
        reward, info = await reward_spec.compute_reward(
            {
                "sample": sample,
                "patch": patch,
                "transcript": result.transcript,
                "trajectories": trajectories,
            }
        )
        return TaskResult(
            reward=reward,
            info={
                "agent": agent.name,
                "session_id": session_id,
                "patch": patch,
                "num_trajectories": len(trajectories),
                "transcript": result.transcript,
                "eval": info,
            },
        )
