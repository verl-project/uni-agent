"""Reward spec for SWE-rebench-V2.

This mirrors the official evaluator at
https://github.com/SWE-rebench/SWE-rebench-V2 (``scripts/eval.py`` +
``lib/agent/log_parsers.py``), adapted to grade a *live* agent container instead
of a fresh ``docker run``.

Differences vs. the SWE-bench / SWE-rebench(-v1) reward spec, all driven by the
V2 instance images (see ``combine.Dockerfile.j2``):

* The repo is checked out at ``/<repo_name>`` (e.g. ``/netcdf-c``), **not**
  ``/testbed``. ``WORKDIR`` is that directory.
* The build-time ``install`` already ran when the image was built, so the reward
  does **not** re-run install and does **not** activate any conda env.
* Tests are language-agnostic: ``install_config.test_cmd`` is run as-is and the
  log is parsed by the named, vendored ``log_parser`` (76 parsers covering
  pytest/go/cargo/maven/jest/...), not a hard-coded pytest parser.
* Grading follows the official metric: the set of ``PASSED`` tests must equal
  ``PASS_TO_PASS ∪ FAIL_TO_PASS`` (after timing-suffix normalization). A more
  lenient SWE-bench-style subset check is also computed and can be selected via
  ``metadata["grading"] = "subset"``.

Unlike the standalone evaluator we must *not* ``git reset --hard`` (that would
wipe the agent's solution from the working tree); we only reset the test files to
``base_commit`` and apply the gold ``test_patch`` on top of the agent's changes.
"""

import re
import shlex
import time
import uuid
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward._swe_rebench_v2 import NAME_TO_PARSER, TestStatus
from uni_agent.reward._swe_rebench_v2 import log_parsers as _v2_log_parsers
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await

# Markers injected around the test command output so the parser only sees test
# logs, not the patch-apply chatter. The V2 evaluator parses the whole log; we
# narrow it for robustness and fall back to the full log if markers are missing.
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# Timing patterns some runners embed in test names; stripped from both actual and
# expected names so run-to-run timing differences don't cause spurious mismatches
# (ported verbatim from SWE-rebench-V2 scripts/eval.py).
_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def get_parser(parser_name: str):
    """Resolve a log parser by name (registry first, then module attribute)."""
    parser = NAME_TO_PARSER.get(parser_name)
    if parser is None:
        parser = getattr(_v2_log_parsers, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown SWE-rebench-V2 log parser: {parser_name!r}")
    return parser


def _normalize_command_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _project_dir(metadata: dict) -> str:
    """Working directory inside the instance image: ``/<repo_name>``."""
    project_dir = metadata.get("project_dir")
    if project_dir:
        return str(project_dir)
    repo = metadata.get("repo", "")
    if "/" not in repo:
        raise ValueError(f"Cannot derive project dir from repo={repo!r}")
    return f"/{repo.split('/')[1]}"


def _test_files_from_patch(patch: str) -> list[str]:
    """Pre-image (``a/``) paths touched by a diff, excluding newly added files."""
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            path = line[len("--- a/") :].strip()
            if path and path != "/dev/null":
                files.append(path)
    return files


@register_reward_spec("swe_rebench_v2")
class SWEReBenchV2RewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout

    @auto_await
    async def apply_gold_patch(self) -> None:
        """Apply the dataset gold patch to the working tree (used by verifiers)."""
        await self._apply_patch(self.metadata["patch"])

    @auto_await
    async def compute_reward(self, interaction_result: dict | None = None, **kwargs) -> tuple[bool, dict]:
        """Apply the gold test patch on top of the agent's solution, run the
        instance test command, parse the log, and grade. Returns (resolved, info).
        """
        result: dict = {
            "eval_completed": False,
            "eval_execution_time": None,
            "resolved": False,
            "passed_match": False,
            "subset_resolved": False,
        }

        try:
            instance = self.metadata
            workdir = _project_dir(instance)
            base_commit = instance["base_commit"]
            test_patch = instance["test_patch"]
            test_cmds = _normalize_command_list(
                instance.get("test_cmd", (instance.get("install_config") or {}).get("test_cmd"))
            )
            if not test_cmds:
                raise ValueError("SWE-rebench-V2 instance missing test_cmd")
            parser_name = instance.get("log_parser") or (instance.get("install_config") or {}).get("log_parser")
            parser = get_parser(parser_name)
            grading = (instance.get("grading") or "strict").lower()
            self.logger.info(
                f"SWE-rebench-V2 eval start: instance={instance.get('instance_id')} "
                f"workdir={workdir} parser={parser_name} grading={grading} test_cmds={test_cmds}"
            )

            eval_script = self._build_eval_script(workdir, base_commit, test_patch, test_cmds)
            script_path = Path(f"/tmp/sbv2_eval_{uuid.uuid4().hex}.sh")
            await self.env.write_file(script_path, eval_script)

            t0 = time.perf_counter()
            output = await self.env.communicate(f"bash {script_path}", timeout=self.eval_timeout, check="ignore")
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            region = self._extract_test_region(output)
            parsed = {_normalize_test_name(k): v for k, v in parser(region).items()}
            passed = {name for name, status in parsed.items() if status == TestStatus.PASSED.value}

            fail_to_pass = {_normalize_test_name(n) for n in instance.get("FAIL_TO_PASS", [])}
            pass_to_pass = {_normalize_test_name(n) for n in instance.get("PASS_TO_PASS", [])}
            expected = fail_to_pass | pass_to_pass

            passed_match = passed == expected
            subset_resolved = fail_to_pass <= passed and pass_to_pass <= passed
            resolved = passed_match if grading == "strict" else subset_resolved

            f2p_resolved = sorted(fail_to_pass & passed)
            f2p_missing = sorted(fail_to_pass - passed)
            p2p_failed = sorted(pass_to_pass - passed)
            unexpected_passed = sorted(passed - expected)

            result.update(
                {
                    "resolved": resolved,
                    "passed_match": passed_match,
                    "subset_resolved": subset_resolved,
                    "grading": grading,
                    "parser": parser_name,
                    "workdir": workdir,
                    "num_parsed": len(parsed),
                    "num_passed_actual": len(passed),
                    "num_expected": len(expected),
                    "fail_to_pass_resolved": f2p_resolved,
                    "fail_to_pass_missing": f2p_missing,
                    "pass_to_pass_failed": p2p_failed,
                    "unexpected_passed": unexpected_passed,
                }
            )
            self.logger.info(
                f"SWE-rebench-V2 eval: resolved={resolved} passed_match={passed_match} "
                f"subset={subset_resolved} f2p={len(f2p_resolved)}/{len(fail_to_pass)} "
                f"p2p_failed={len(p2p_failed)} parsed={len(parsed)} parser={parser_name} "
                f"time={result['eval_execution_time']:.1f}s"
            )
            if not resolved:
                self._log_failure_details(
                    region=region,
                    parsed=parsed,
                    f2p_missing=f2p_missing,
                    p2p_failed=p2p_failed,
                    unexpected_passed=unexpected_passed,
                    result=result,
                )
        except Exception as exc:
            self.logger.error(f"Failed to evaluate SWE-rebench-V2 instance: {exc}")
            result["error"] = str(exc)

        return result["resolved"], result

    # Caps for the failure diagnostics so a broken case doesn't flood the logs.
    _MAX_NAMES_IN_LOG = 25
    _OUTPUT_TAIL_CHARS = 4000

    def _log_failure_details(
        self,
        *,
        region: str,
        parsed: dict,
        f2p_missing: list[str],
        p2p_failed: list[str],
        unexpected_passed: list[str],
        result: dict,
    ) -> None:
        """Emit enough detail to triage a gold-patch failure at a glance."""
        cap = self._MAX_NAMES_IN_LOG

        def fmt(names: list[str]) -> str:
            shown = names[:cap]
            suffix = f" (+{len(names) - cap} more)" if len(names) > cap else ""
            return f"{shown}{suffix}"

        self.logger.warning(
            "SWE-rebench-V2 NOT resolved -- gold-patch grading failed:\n"
            f"  FAIL_TO_PASS still failing ({len(f2p_missing)}): {fmt(f2p_missing)}\n"
            f"  PASS_TO_PASS broke ({len(p2p_failed)}): {fmt(p2p_failed)}\n"
            f"  unexpected PASSED not in expected ({len(unexpected_passed)}): {fmt(unexpected_passed)}"
        )
        if not parsed:
            self.logger.warning(
                "SWE-rebench-V2 parser matched 0 tests -- the test command most likely "
                "failed to run (build error / missing binary / wrong workdir / test_patch "
                "did not apply). Inspect the raw output tail below."
            )
        tail = region[-self._OUTPUT_TAIL_CHARS :]
        result["test_output_tail"] = tail
        self.logger.warning(f"SWE-rebench-V2 raw test-output tail (last {len(tail)} chars):\n{tail}")

    def _build_eval_script(self, workdir: str, base_commit: str, test_patch: str, test_cmds: list[str]) -> str:
        wd = shlex.quote(workdir)
        test_files = _test_files_from_patch(test_patch)
        # Heredoc carries the test patch inline to avoid a second write_file round-trip.
        delimiter = "SBV2_TEST_PATCH_EOF"
        lines = [
            "#!/bin/bash",
            "set -uo pipefail",
            f"cd {wd} || {{ echo 'cd failed'; exit 2; }}",
            f"git config --global --add safe.directory {wd} 2>/dev/null || true",
            "git config --global --add safe.directory '*' 2>/dev/null || true",
        ]
        if test_files:
            quoted = " ".join(shlex.quote(f) for f in test_files)
            # Restore test files to the base revision so the agent can't tamper
            # with them; new test files (added by the patch) simply don't exist yet.
            lines.append(f"git checkout {shlex.quote(base_commit)} -- {quoted} 2>/dev/null || true")
        lines += [
            f"cat > /tmp/sbv2_test_patch.diff <<'{delimiter}'",
            test_patch,
            delimiter,
            "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /tmp/sbv2_test_patch.diff "
            "|| git apply --recount --ignore-space-change --whitespace=nowarn /tmp/sbv2_test_patch.diff "
            "|| patch --batch --fuzz=5 -p1 -i /tmp/sbv2_test_patch.diff || true",
            f"echo {shlex.quote(START_TEST_OUTPUT)}",
        ]
        lines += test_cmds
        lines.append(f"echo {shlex.quote(END_TEST_OUTPUT)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_test_region(output: str) -> str:
        if START_TEST_OUTPUT in output:
            region = output.split(START_TEST_OUTPUT, 1)[1]
            if END_TEST_OUTPUT in region:
                region = region.split(END_TEST_OUTPUT, 1)[0]
            return region
        return output

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        """Apply a patch to the working tree; tries several apply strategies."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        workdir = shlex.quote(_project_dir(self.metadata))
        patch_path = Path(f"/tmp/sbv2_patch_{uuid.uuid4().hex}.diff")
        await self.env.write_file(patch_path, patch)
        p = patch_path.as_posix()
        commands = [
            f"cd {workdir} && git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {p}",
            f"cd {workdir} && git apply --whitespace=fix {p}",
            f"cd {workdir} && git apply --reject --whitespace=nowarn {p}",
            f"cd {workdir} && patch --batch --fuzz=5 -p1 -i {p}",
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
