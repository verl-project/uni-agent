"""Claude Code runner for the blackbox SWE-agent recipe.

Claude Code runs inside a remote sandbox via a sidecar tool image mounted at
``/opt/claude-code``. The runner creates the sandbox, invokes the ``claude``
binary pointed at the gateway, evaluates the reward in the same sandbox, and
posts ``reward_info`` to the per-session endpoint (same contract as the
mini-swe-agent runner).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
from pathlib import Path

import httpx

from examples.blackbox_recipes.claude_code.dataset import extract_image
from examples.blackbox_recipes.claude_code.reward import build_reward_context, evaluate_in_env
from examples.blackbox_recipes.sandbox_client import (
    SandboxClient,
    extract_upstream,
    rewrite_gateway_url,
)
from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)

DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest"
TOOL_TARGET = "/opt/claude-code"


class SandboxEnvForReward:
    """Adapts :class:`SandboxClient` to the async env interface used by reward
    specs (``communicate``, ``write_file``, ``read_file``).
    """

    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=600, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.run(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(f"{error_msg}: {result.stdout[:200]}")
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")


def extract_task(raw_prompt) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _extract_issue_text(task: str) -> str:
    start = task.find("<issue_description>")
    end = task.find("</issue_description>")
    if start >= 0 and end > start:
        return task[start + len("<issue_description>") : end].strip()
    marker = "\nFollow these steps to resolve the issue:"
    if marker in task:
        return task.split(marker, 1)[0].strip()
    return task.strip()


def _decode_metadata_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def build_claude_task(raw_prompt, tools_kwargs: dict | None = None) -> str:
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt)
    metadata = (tools_kwargs.get("reward") or {}).get("metadata") or {}
    issue = metadata.get("problem_statement") or _extract_issue_text(task)
    tests = _decode_metadata_list(metadata.get("FAIL_TO_PASS"))
    if not tests:
        tests = _decode_metadata_list(metadata.get("PASS_TO_PASS"))[:3]
    tests_block = (
        "\n".join(f"- {test}" for test in tests) if tests else "- Run the closest relevant tests you identify."
    )

    return (
        "You are fixing a SWE-bench task in /testbed.\n\n"
        "Issue:\n"
        f"{issue}\n\n"
        "Rules:\n"
        "- Edit source files only. Do not modify tests.\n"
        "- The development environment is already installed; do not install packages unless a test command proves it "
        "is necessary.\n"
        "- There is no submit tool in this environment. Do not try to submit.\n"
        "- Do not create extra edge-case test files after the relevant tests pass.\n"
        "- Do not run `pytest --collect-only`, `git log`, or any other command that does not directly validate the "
        "fix.\n"
        "- Do not analyze unrelated `is_separable` behavior.\n"
        "- Do not run additional ad-hoc verification after the listed relevant pytest command passes.\n"
        "- Do not commit.\n"
        "- After the minimal fix is applied and a relevant pytest command passes, print a one-line summary and exit "
        "immediately.\n\n"
        "Relevant tests to run after the fix:\n"
        f"{tests_block}\n"
    )


def build_claude_command(
    *,
    task: str,
    base_url: str,
    max_turns: int,
    model: str = "default",
    permission_mode: str = "bypassPermissions",
    conda_env: str | None = "testbed",
    disable_web_tools: bool = True,
    disable_slash_commands: bool = True,
) -> str:
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "not-needed",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_FORK_SUBAGENT": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    if conda_env:
        conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)}",
                f"CONDA_PREFIX={shlex.quote(conda_prefix)}",
                f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH",
            ]
        )
    env_prefix = " ".join(env_assignments)
    argv = [
        "/opt/claude-code/bin/claude",
        "-p",
        task,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
    ]
    if disable_slash_commands:
        argv.append("--disable-slash-commands")
    if disable_web_tools:
        argv.extend(["--disallowedTools", "Agent", "Task", "WebFetch", "WebSearch"])
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        "cd /testbed; "
        f"{env_prefix} " + shlex.join(argv)
    )


async def _create_claude_sandbox(
    *,
    image: str,
    sidecar_image: str,
    gateway_url: str,
    max_retries: int = 10,
):
    upstream = extract_upstream(gateway_url) if gateway_url else ""
    return await SandboxClient.create(
        image=image,
        sidecar_image=sidecar_image,
        sidecar_target=TOOL_TARGET,
        upstream=upstream,
        max_retries=int(max_retries),
    )


async def claude_code_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    tool_image: str = DEFAULT_TOOL_IMAGE,
    run_timeout: int = 7200,
    conda_env: str = "testbed",
    sandbox_max_retries: int = 10,
    **kwargs,
) -> None:
    """Run Claude Code inside a sandbox with sidecar tool mount.

    Flow:
        1. Create remote sandbox with the claude-code sidecar
        2. Run the claude binary against the gateway tunnel
        3. Evaluate reward in the same sandbox
        4. Post reward_info for the framework reward path
    """
    tools_kwargs = tools_kwargs or {}
    logger.info("claude_code_runner called, sample_index=%d", sample_index)

    task = build_claude_task(raw_prompt, tools_kwargs)
    env_config = tools_kwargs.get("env", {})
    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")

    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")

    sandbox = await _create_claude_sandbox(
        image=image,
        sidecar_image=tool_image,
        gateway_url=gateway_url,
        max_retries=sandbox_max_retries,
    )

    try:
        post_setup_cmd = env_config.get("post_setup_cmd", "")
        if post_setup_cmd:
            setup_result = await sandbox.run(post_setup_cmd, timeout=120)
            if setup_result.exit_code != 0:
                logger.warning(
                    "post_setup_cmd failed rc=%s: %.300s",
                    setup_result.exit_code,
                    setup_result.stdout + setup_result.stderr,
                )

        claude_base_url = rewrite_gateway_url(gateway_url, strip_v1=True)
        max_turns = int(os.environ.get("AGENT_MAX_TURNS", "100"))
        agent_cmd = build_claude_command(
            task=task,
            base_url=claude_base_url,
            max_turns=max_turns,
            conda_env=conda_env,
        )

        started_at = time.perf_counter()
        result = await sandbox.run(agent_cmd, timeout=int(run_timeout))
        elapsed = time.perf_counter() - started_at
        logger.info("[sample %d] claude-code finished rc=%s elapsed=%.1fs", sample_index, result.exit_code, elapsed)
        if result.exit_code != 0:
            logger.warning(
                "[sample %d] claude-code failed stdout_tail=%r stderr_tail=%r",
                sample_index,
                (result.stdout or "")[-4000:],
                (result.stderr or "")[-4000:],
            )

        metadata, eval_timeout = build_reward_context(tools_kwargs)
        score, eval_result = await evaluate_in_env(SandboxEnvForReward(sandbox), metadata, eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            "reward_score": score,
            "claude_code_exit_code": result.exit_code,
            **eval_result,
        }
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url is empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        await sandbox.cleanup()
