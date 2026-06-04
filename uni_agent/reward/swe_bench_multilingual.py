"""Reward spec for SWE-bench/SWE-bench_Multilingual.

The dataset spans 7 non-Python languages (c/go/java/js/php/ruby/rust) and is fully
covered by the official ``swebench`` harness, so we grade with it directly instead
of re-implementing per-language logic:

* ``make_test_spec`` builds the language-aware eval script (reset *test files only*
  -> apply the gold ``test_patch`` -> run the repo's test command, wrapped in
  ``START/END`` markers). Crucially it never ``git reset --hard`` the whole repo, so
  the agent's solution in ``/testbed`` is preserved.
* The per-repo parser (``MAP_REPO_TO_PARSER``) + official resolution metric
  (``get_eval_tests_report`` / ``get_resolution_status``) decide ``resolved``.

Instance images are the published ``swebench/sweb.eval.x86_64.<id>`` containers with
the repo checked out at ``/testbed``.
"""

import re
import time
import uuid
from pathlib import Path

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.test_spec import make_test_spec

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

# Strip ANSI color escapes and lone CRs so the line-anchored parsers (cargo, go,
# rspec, ...) match cleanly even if a runner forces color over the session PTY.
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _clean(text: str) -> str:
    return re.sub(r"\r\n?", "\n", _ANSI_RE.sub("", text))


@register_reward_spec("swe_bench_multilingual")
class SWEBenchMultilingualRewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout
        # Build once; for non-Python repos this is pure-CPU (no network).
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
            await self.env.write_file(script_path, self.test_spec.eval_script)

            t0 = time.perf_counter()
            # `| cat` forces a non-TTY stdout (matching the upstream harness) so test
            # runners don't emit colored / TUI output that breaks the parsers.
            output = await self.env.communicate(
                f"export TERM=dumb NO_COLOR=1 FORCE_COLOR=0; bash {script_path} 2>&1 | cat",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            eval_report = self._grade(_clean(output))
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
        """Apply a patch to /testbed; tries several apply strategies in order."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/sbm_patch_{uuid.uuid4().hex}.diff")
        await self.env.write_file(patch_path, patch)
        p = patch_path.as_posix()
        commands = [
            f"cd /testbed && git apply -v --3way --recount --whitespace=nowarn {p}",
            f"cd /testbed && git apply --whitespace=fix {p}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {p}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {p}",
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
