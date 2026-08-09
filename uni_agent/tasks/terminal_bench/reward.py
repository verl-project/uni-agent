"""Run a Terminal-Bench verifier using Harbor's reward-file contract."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import shlex
import time
from typing import Any

from uni_agent.sandbox import SandboxBackend

logger = logging.getLogger(__name__)

_ENV_REFERENCE_RE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_OUTPUT_TAIL_CHARS = 12000


def parse_json_mapping(raw: Any, *, field: str) -> dict[str, Any]:
    """Parse a JSON object stored in a provider-agnostic parquet field."""
    if raw in (None, ""):
        return {}
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def resolve_env_mapping(values: dict[str, Any]) -> dict[str, str]:
    """Resolve full-value ``${VAR}`` and ``${VAR:-default}`` references."""
    resolved: dict[str, str] = {}
    for key, value in values.items():
        text = str(value)
        match = _ENV_REFERENCE_RE.fullmatch(text)
        if match is None:
            resolved[str(key)] = text
            continue

        name, default = match.groups()
        if name in os.environ:
            resolved[str(key)] = os.environ[name]
        elif default is not None:
            resolved[str(key)] = default
        else:
            raise ValueError(f"required environment variable {name!r} is not set")
    return resolved


async def install_archive(
    sandbox: SandboxBackend,
    encoded_archive: str,
    *,
    target_dir: str,
) -> None:
    """Extract one parquet-embedded task archive into the sandbox."""
    if not encoded_archive:
        raise ValueError(f"Terminal-Bench archive for {target_dir} is empty")
    try:
        content = base64.b64decode(encoded_archive, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError(f"Terminal-Bench archive for {target_dir} is invalid") from exc

    archive_path = "/tmp/uni-agent-task-archive.tar.gz"
    await sandbox.write_file(archive_path, content)
    try:
        command = (
            f"rm -rf -- {shlex.quote(target_dir)} && "
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar --no-same-owner -xzf {shlex.quote(archive_path)} -C {shlex.quote(target_dir)}"
        )
        response = await sandbox.exec_shell(command)
        if response.exit_code != 0:
            detail = (response.stderr or response.stdout or "unknown extraction error").strip()[-2000:]
            raise RuntimeError(f"failed to install archive into {target_dir}: {detail}")
    finally:
        await sandbox.exec(["rm", "-f", archive_path])


async def _read_optional_text(sandbox, path: str) -> str | None:
    exists = await sandbox.exec(["test", "-f", path])
    if exists.exit_code != 0:
        return None
    return (await sandbox.read_file(path)).decode("utf-8", errors="replace")


def parse_reward_files(reward_json: str | None, reward_text: str | None) -> tuple[float, dict[str, float]]:
    """Parse verifier output and select its primary reward."""
    if reward_json is not None:
        try:
            payload = json.loads(reward_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid /logs/verifier/reward.json: {exc}") from exc
        if not isinstance(payload, dict) or not payload:
            raise ValueError("/logs/verifier/reward.json must be a non-empty object")
        try:
            rewards = {str(name): float(value) for name, value in payload.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("/logs/verifier/reward.json values must be numeric") from exc
    elif reward_text is not None:
        text = reward_text.strip()
        if not text:
            raise ValueError("/logs/verifier/reward.txt is empty")
        try:
            score = float(text)
        except ValueError as exc:
            raise ValueError(f"invalid /logs/verifier/reward.txt value: {text!r}") from exc
        rewards = {"reward": score}
    else:
        raise ValueError("verifier produced neither reward.json nor reward.txt")

    # Use "reward", or the sole metric when only one exists, as the scalar score.
    if "reward" in rewards:
        return rewards["reward"], rewards
    if len(rewards) == 1:
        return next(iter(rewards.values())), rewards
    raise ValueError("reward.json has multiple metrics but no primary 'reward' key")


async def compute_reward(
    metadata: dict[str, Any],
    sandbox: SandboxBackend,
) -> dict[str, Any]:
    """Upload official tests, execute the verifier, and parse its reward."""
    instance_id = metadata["instance_id"]
    timeout = float(metadata["verifier_timeout"])

    # Step 1: Clear verifier-owned output after the agent phase to prevent a
    # pre-seeded reward from being mistaken for verifier output.
    reset = await sandbox.exec_shell("rm -rf -- /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier")
    if reset.exit_code != 0:
        raise RuntimeError(f"failed to prepare verifier directories for {instance_id}: {reset.stderr.strip()}")

    # Step 2: Upload the official tests after the agent phase.
    await install_archive(sandbox, metadata["tests_archive"], target_dir="/tests")

    # Step 3: Make the test entrypoint executable and resolve verifier env vars.
    chmod = await sandbox.exec(["chmod", "+x", "/tests/test.sh"])
    if chmod.exit_code != 0:
        raise RuntimeError(f"failed to make verifier executable for {instance_id}: {chmod.stderr.strip()}")

    verifier_env = resolve_env_mapping(parse_json_mapping(metadata["verifier_env_json"], field="verifier_env_json"))
    logger.info("running Terminal-Bench verifier for %s (timeout=%.0fs)", instance_id, timeout)

    # Step 4: Run test.sh in the sandbox's configured task workdir and merge
    # stdout/stderr into one verifier log.
    started = time.perf_counter()
    response = await sandbox.exec_shell(
        "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1",
        env=verifier_env or None,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started

    # Step 5: Read the verifier log and both supported reward files.
    test_output = await _read_optional_text(sandbox, "/logs/verifier/test-stdout.txt") or ""
    reward_json = await _read_optional_text(sandbox, "/logs/verifier/reward.json")
    reward_text = await _read_optional_text(sandbox, "/logs/verifier/reward.txt")

    # Step 6: Treat timeout as incomplete; otherwise prefer reward.json and
    # fall back to reward.txt.
    score = 0.0
    rewards = None
    error = None
    if response.exit_code == -1:
        error = f"verifier timed out after {timeout:g}s"
    else:
        try:
            score, rewards = parse_reward_files(reward_json, reward_text)
        except ValueError as exc:
            error = str(exc)

    if error is not None:
        logger.warning("Terminal-Bench verifier failed for %s: %s", instance_id, error)

    # Step 7: Return the scalar reward and verifier diagnostics.
    completed = error is None
    result = {
        "reward": score,
        "resolved": completed and score == 1.0,
        "eval_completed": completed,
        "eval_execution_time": elapsed,
        "eval_report": {
            "test_exit_code": response.exit_code,
            "test_output_tail": test_output[-_OUTPUT_TAIL_CHARS:],
            "stderr_tail": (response.stderr or "")[-2000:],
            "rewards": rewards,
            "error": error,
        },
    }
    logger.info(
        "Terminal-Bench verifier done for %s in %.1fs (exit=%s, reward=%.3f, completed=%s)",
        instance_id,
        elapsed,
        response.exit_code,
        result["reward"],
        result["eval_completed"],
    )
    return result
