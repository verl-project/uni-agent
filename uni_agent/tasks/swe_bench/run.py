"""SWE-bench task: one problem family, solved by whichever agent you configure.

A task *selects* an agent and subclasses :class:`TaskConfig` to narrow ``agent``
and add task knobs (the default here is the white-box
:class:`~uni_agent.agents.code_act.CodeActConfig`, but a black-box agent such as
``claude_code`` works too -- both drive the model through the gateway session, so
the task runs them the same way). ``run.py`` owns the runtime lifecycles --
mirroring the driver in :mod:`uni_agent.framework.framework`:

* it starts the sandbox (cleaning up even if ``start`` fails) and always stops it;
* it creates a gateway session, runs the agent against it, and finalizes it for
  trajectories (aborting on error);
* it then scores what the agent produced.

Values below are illustrative defaults -- the shape is the point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import Field

from ...agents.code_act import CodeActConfig
from ...reward import load_reward_spec
from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import reward_config

if TYPE_CHECKING:
    from ...gateway.manager import GatewayManager


class SWEBenchTaskConfig(TaskConfig):
    """SWE-bench config: the white-box code_act agent + dataset knobs."""

    agent: CodeActConfig = Field(default_factory=CodeActConfig)
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"


@register_task("swe_bench")
class SWEBenchTask(Task):
    name = "swe_bench"

    async def run(self, sample: dict[str, Any], *, gateway: GatewayManager | None = None) -> TaskResult:
        """Run one episode -- white-box or black-box -- then score it.

        The task owns every runtime lifecycle (the agent only solves the problem),
        mirroring the driver in :mod:`uni_agent.framework.framework`:

        1. **Sandbox** -- entered with ``async with``: ``start`` on enter (cleaned
           up if start fails), ``stop`` always on exit.
        2. **Gateway session** -- both agent kinds drive the model through the
           per-session ``base_url``, so the task always creates a session:
           ``create_session`` -> run agent -> ``finalize_session`` for trajectories
           (``abort_session`` on error). A white-box agent calls the URL from our
           framework loop; a black-box agent is launched in the sandbox pointed at
           that same URL.
        3. **Reward** -- score the patch the agent left in the sandbox.
        """
        if gateway is None:
            raise ValueError(
                "swe_bench: run(...) requires a gateway -- every agent drives the model through the session URL"
            )
        agent = self.build_agent()
        session_id = f"swe-bench-{uuid4().hex}"

        # Sandbox lifecycle via async-with: __aenter__ starts it (cleaning up if
        # start fails), __aexit__ always stops it -- same guarantees as try/finally.
        async with self.build_sandbox() as sandbox:

            # Gateway session lifecycle: create -> run agent -> finalize (abort on error).
            session = await gateway.create_session(session_id)
            try:
                result = await agent.run(sandbox=sandbox, sample=sample, session=session)
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
