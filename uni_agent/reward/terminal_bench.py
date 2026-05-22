import json
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

# Container paths (mirror harbor's EnvironmentPaths convention).
SOLUTION_DIR = "/solution"
TESTS_DIR = "/tests"
AGENT_LOG_DIR = "/logs/agent"
VERIFIER_LOG_DIR = "/logs/verifier"
ORACLE_LOG_PATH = f"{AGENT_LOG_DIR}/oracle.txt"
TEST_STDOUT_PATH = f"{VERIFIER_LOG_DIR}/test-stdout.txt"
REWARD_JSON_PATH = f"{VERIFIER_LOG_DIR}/reward.json"
REWARD_TEXT_PATH = f"{VERIFIER_LOG_DIR}/reward.txt"


@register_reward_spec("terminal_bench_v2")
@register_reward_spec("terminal_bench")
class TerminalBenchRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        metadata: dict,
        env: AgentEnv,
        eval_timeout: int | float = 600,
    ):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.eval_timeout = float(eval_timeout)
        self.logger = get_logger("reward_spec", run_id=run_id)

        self.task_id: str = metadata["task_id"]
        self.task_config: dict = metadata.get("task_config", {})
        self.workdir: str | None = metadata.get("workdir")
        self.solve_relpath: str = metadata.get("solve_relpath", "solve.sh")
        self.test_relpath: str = metadata.get("test_relpath", "test.sh")
        self.agent_timeout = float(self.task_config.get("agent", {}).get("timeout_sec", self.eval_timeout))

    def _in_workdir(self, command: str) -> str:
        """Wrap ``command`` so it runs inside the task workdir (Dockerfile WORKDIR)."""
        if not self.workdir:
            return command
        return f"cd {shlex.quote(self.workdir)} && {command}"

    async def _ensure_runtime_dirs(self) -> None:
        """Create the harbor-style /logs, /tests, /solution dirs (idempotent)."""
        await self.env.communicate(
            f"mkdir -p {AGENT_LOG_DIR} {VERIFIER_LOG_DIR} {TESTS_DIR} {SOLUTION_DIR} "
            f"&& chmod -R 777 /logs {TESTS_DIR} {SOLUTION_DIR} || true",
            check="ignore",
        )

    async def _upload_archive(self, archive_bytes: bytes, target_dir: str) -> None:
        """Write ``archive_bytes`` to a temp tar.gz, ship it to the container and extract.

        The archive is assumed to contain entries with paths *relative* to
        ``target_dir`` (i.e. without a top-level directory wrapper), matching
        what ``_tar_gz_bytes`` in the preprocessing script produces.
        """
        archive_name = f"{Path(target_dir).name or 'archive'}_{uuid.uuid4().hex}.tar.gz"
        container_archive_path = f"/tmp/{archive_name}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_archive_path = Path(tmp_dir) / archive_name
            local_archive_path.write_bytes(archive_bytes)
            await self.env.copy_to_container(local_archive_path, Path(container_archive_path))

        await self.env.communicate(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar -xzf {shlex.quote(container_archive_path)} -C {shlex.quote(target_dir)} && "
            f"rm -f {shlex.quote(container_archive_path)}",
            check="raise",
            error_msg=f"Failed to extract archive into {target_dir}",
        )

    async def _read_rewards(self) -> dict[str, float | int]:
        """Read reward.json if available, else fall back to reward.txt (single float).

        We probe with ``test -f`` first so we don't hit swerex's server-side
        ``FileNotFoundError`` path (which spams a CRITICAL traceback even
        though the caller handles it gracefully).
        """
        probe = await self.env.communicate(
            f"if [ -f {shlex.quote(REWARD_JSON_PATH)} ]; then echo json; "
            f"elif [ -f {shlex.quote(REWARD_TEXT_PATH)} ]; then echo text; "
            f"else echo none; fi",
            check="ignore",
        )
        kind = probe.strip().splitlines()[-1].strip() if probe else "none"

        if kind == "json":
            parsed = json.loads(await self.env.read_file(REWARD_JSON_PATH))
            if not isinstance(parsed, dict):
                raise ValueError("reward.json must contain a JSON object")
            return parsed
        if kind == "text":
            reward_text = (await self.env.read_file(REWARD_TEXT_PATH)).strip()
            return {"reward": float(reward_text)}
        raise FileNotFoundError(f"Neither {REWARD_JSON_PATH} nor {REWARD_TEXT_PATH} exists after running tests")

    @auto_await
    async def apply_gold_solution(self) -> dict:
        """Extract ``solution_archive`` and run the gold ``solve.sh``."""
        result: dict = {
            "gold_completed": False,
            "gold_execution_time": None,
            "solution_output_path": ORACLE_LOG_PATH,
        }

        await self._ensure_runtime_dirs()
        await self._upload_archive(self.metadata["solution_archive"], SOLUTION_DIR)

        solve_path = f"{SOLUTION_DIR}/{self.solve_relpath}"
        await self.env.communicate(f"chmod +x {shlex.quote(solve_path)}", check="ignore")

        started_at = time.perf_counter()
        output = await self.env.communicate(
            self._in_workdir(f"{shlex.quote(solve_path)} > {ORACLE_LOG_PATH} 2>&1"),
            timeout=self.agent_timeout,
            check="ignore",
        )
        result["gold_completed"] = True
        result["gold_execution_time"] = time.perf_counter() - started_at
        result["command_output"] = output
        return result

    @auto_await
    async def apply_gold_patch(self) -> dict:
        """Compatibility alias for SWE-style callers."""
        return await self.apply_gold_solution()

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[float, dict]:
        """Extract ``tests_archive``, run ``test.sh`` and read the reward file.

        The test script is expected to write either
        ``/logs/verifier/reward.json`` (preferred) or ``/logs/verifier/reward.txt``
        — this is the same contract harbor uses, and Terminal-Bench v2's
        ``tests/test.sh`` scripts already follow it.
        """
        result: dict = {
            "eval_completed": False,
            "eval_execution_time": None,
            "rewards": None,
            "reward": 0.0,
            "test_output_path": TEST_STDOUT_PATH,
        }

        await self._ensure_runtime_dirs()
        await self._upload_archive(self.metadata["tests_archive"], TESTS_DIR)

        test_path = f"{TESTS_DIR}/{self.test_relpath}"
        await self.env.communicate(f"chmod +x {shlex.quote(test_path)}", check="ignore")

        started_at = time.perf_counter()
        output = await self.env.communicate(
            self._in_workdir(f"{shlex.quote(test_path)} > {TEST_STDOUT_PATH} 2>&1"),
            timeout=self.eval_timeout,
            check="ignore",
        )
        result["eval_completed"] = True
        result["eval_execution_time"] = time.perf_counter() - started_at
        result["command_output"] = output

        try:
            result["test_output"] = await self.env.read_file(TEST_STDOUT_PATH)
        except Exception as exc:
            result["test_output_read_error"] = str(exc)

        reward_score = 0.0
        try:
            rewards = await self._read_rewards()
            reward_score = float(rewards.get("reward", 0.0))
            result["rewards"] = rewards
            result["reward"] = reward_score
        except Exception as exc:
            self.logger.error(f"Failed to read Terminal-Bench reward: {exc}")
            result["reward_read_error"] = str(exc)

        return reward_score, result
