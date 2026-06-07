"""Reward spec for SWE-bench/SWE-bench_Multilingual.

The dataset spans 7 non-Python languages (c/go/java/js/php/ruby/rust) and is fully
covered by the official ``swebench`` harness, so we grade with it directly instead
of re-implementing per-language logic:

* The eval script is built explicitly in ``_make_eval_script_list`` -- a transcription
  of swebench's ``make_eval_script_list_common`` (+ the JS image-asset step): reset
  *only the test files* to ``base_commit`` -> ``git apply`` the gold ``test_patch`` ->
  optional per-repo ``build`` -> run the repo's ``test_cmd`` wrapped in ``START``/
  ``END`` markers -> reset the test files again. Crucially it never ``git reset --hard``
  the whole repo, so build-time edits and the agent's solution in ``/testbed`` survive.
* The per-repo parser (``MAP_REPO_TO_PARSER``) + official resolution metric
  (``get_eval_tests_report`` / ``get_resolution_status``) decide ``resolved``;
  ``make_test_spec`` still provides those grading inputs (F2P/P2P, parser key).

Instance images are the published ``swebench/sweb.eval.x86_64.<id>`` containers with
the repo checked out at ``/testbed``.
"""

import time
import uuid
from pathlib import Path

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.javascript import get_download_img_commands
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import get_modified_files

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

# Heredoc delimiter the upstream harness uses to inline the test patch into the script.
HEREDOC_DELIMITER = "EOF_114329324912"


@register_reward_spec("swe_bench_multilingual")
class SWEBenchMultilingualRewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout
        self.repo = metadata["repo"]
        self.version = metadata["version"]
        # Still used for grading inputs (F2P/P2P, parser key, language); pure-CPU.
        self.test_spec = make_test_spec(metadata)

    @auto_await
    async def apply_gold_patch(self) -> None:
        """Apply the dataset gold patch to the working tree (used by verifiers)."""
        await self._apply_patch(self.metadata["patch"])

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }
        try:
            script_path = Path(f"/tmp/sbm_eval_{uuid.uuid4().hex}.sh")
            await self.env.write_file(script_path, self._build_eval_script())

            t0 = time.perf_counter()
            # `| cat` makes stdout a pipe (non-TTY), standing in for the official
            # harness's non-TTY `docker exec` so runners don't emit colored/TUI output.
            output = await self.env.communicate(
                f"bash {script_path} 2>&1 | cat",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            eval_report = self._grade(output)
            result["eval_report"] = eval_report
            result["resolved"] = eval_report["resolved"]
            self.logger.info(
                f"SWE-bench-Multilingual eval: instance={self.test_spec.instance_id} "
                f"repo={self.test_spec.repo} lang={self.test_spec.language} "
                f"resolved={eval_report['resolved']} found={eval_report['found_eval_status']} "
                f"time={result['eval_execution_time']:.1f}s"
            )
        except Exception as exc:
            self.logger.error(f"Failed to evaluate SWE-bench-Multilingual instance: {exc}")
            result["error"] = str(exc)
        return result["resolved"], result

    def _make_eval_script_list(self) -> list[str]:
        """Explicit transcription of swebench's ``make_eval_script_list`` for this
        dataset's languages (``make_eval_script_list_common`` + the JS image-asset
        step), kept inline -- like ``swe_bench.py`` -- so the eval flow is visible and
        tweakable instead of hidden behind ``TestSpec.eval_script``.

        Steps: reset *only the test files* to ``base_commit`` (never the whole repo, so
        build-time ``pre_install`` edits and the agent's solution survive) -> ``git
        apply`` the gold ``test_patch`` -> optional per-repo ``build`` -> run ``test_cmd``
        between the ``START``/``END`` markers the parser keys off -> reset the test files
        again so they can't be tampered with.
        """
        instance = self.metadata
        repo_directory = "/testbed"
        base_commit = instance["base_commit"]
        test_patch = instance["test_patch"]
        specs = MAP_REPO_VERSION_TO_SPECS[self.repo][self.version]

        test_files = get_modified_files(test_patch)
        if test_files:
            reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
        else:
            reset_tests_command = "echo 'skip reset'"

        build_commands = list(specs.get("build", []))
        apply_test_patch_command = (
            f"git apply --verbose --reject - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        test_cmd = specs["test_cmd"]
        test_commands = [test_cmd] if isinstance(test_cmd, str) else list(test_cmd)

        eval_commands = [
            "chmod 1777 /tmp 2>/dev/null || true",
            f"cd {repo_directory}",
            f"git config --global --add safe.directory {repo_directory}",
            f"cd {repo_directory}",
            reset_tests_command,
            apply_test_patch_command,
            *build_commands,
            f": '{START_TEST_OUTPUT}'",
            *test_commands,
            f": '{END_TEST_OUTPUT}'",
            reset_tests_command,
        ]
        # JS instances may ship test image fixtures pulled in right after the reset
        # (a no-op unless the instance carries ``image_assets``).
        if MAP_REPO_TO_EXT[self.repo] == "js":
            idx = eval_commands.index(apply_test_patch_command)
            eval_commands[idx:idx] = get_download_img_commands(instance)
        return eval_commands

    def _build_eval_script(self) -> str:
        """Assemble the eval script (same header as ``TestSpec.eval_script``; no
        ``set -e`` on purpose, so the trailing test-file reset always runs)."""
        return "\n".join(["#!/bin/bash", "set -uxo pipefail", *self._make_eval_script_list()]) + "\n"

    def _grade(self, output: str) -> dict:
        """Parse the test region and grade against FAIL_TO_PASS / PASS_TO_PASS."""
        report = {"resolved": False, "found_eval_status": False, "test_status": None}

        parser = MAP_REPO_TO_PARSER[self.test_spec.repo]
        if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
            region = output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        else:
            region = output
        status_map = parser(region, self.test_spec)
        # Fallback: some runners write results outside the markers (e.g. stderr).
        if not status_map:
            status_map = parser(output, self.test_spec)
        if not status_map:
            self.logger.warning(
                "SWE-bench-Multilingual parser matched 0 tests -- the test command likely "
                f"failed to run. Output tail:\n{output[-3000:]}"
            )
            return report

        report["found_eval_status"] = True
        eval_ref = {
            "instance_id": self.test_spec.instance_id,
            "FAIL_TO_PASS": self.test_spec.FAIL_TO_PASS,
            "PASS_TO_PASS": self.test_spec.PASS_TO_PASS,
        }
        eval_type = EvalType.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        tests_status = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        report["test_status"] = tests_status
        report["resolved"] = get_resolution_status(tests_status) == ResolvedStatus.FULL.value
        if not report["resolved"]:
            f2p_missing = tests_status["FAIL_TO_PASS"]["failure"]
            p2p_failed = tests_status["PASS_TO_PASS"]["failure"]
            self.logger.warning(
                f"SWE-bench-Multilingual NOT resolved: FAIL_TO_PASS still failing={f2p_missing[:25]} "
                f"PASS_TO_PASS broke={p2p_failed[:25]}"
            )
        return report

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        """Apply a patch string to the env. Tries multiple apply strategies in order."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await self.env.write_file(patch_path, patch)
        commands = [
            f"cd /testbed && git apply --whitespace=fix {patch_path.as_posix()}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {patch_path.as_posix()}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {patch_path.as_posix()}",
        ]
        last_error: Exception | None = None
        for cmd in commands:
            try:
                await self.env.communicate(cmd, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError("Failed to apply patch with any command") from last_error
