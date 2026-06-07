"""Reward spec for ``nebius/SWE-rebench-V2`` (Python subset).

SWE-rebench-V2 is a language-agnostic SWE task collection; we currently only
process its Python slice (see ``examples/data_preprocess/swe_rebench_v2.py``).
Unlike SWE-bench it is **not** graded by the ``swebench`` harness -- every
instance ships its own ``install_config`` (``install`` / ``test_cmd`` /
``log_parser``) and a prebuilt Docker image, and grading uses the dataset
authors' own per-framework log parsers. We vendor those parsers
(``swe_rebench_v2_log_parsers`` -- Python ``parse_log_pytest`` for now, with
``NAME_TO_PARSER`` as the extension point for other languages) and re-implement
the official eval flow so results match the upstream harness
(https://github.com/SWE-rebench/SWE-rebench-V2, ``scripts/eval.py``).

Key differences from the SWE-bench reward specs:

* The repo is checked out at ``/<repo_name>`` (e.g. ``/synthetics`` for
  ``elastic/synthetics``), **not** ``/testbed`` -- this mirrors the upstream
  ``combine.Dockerfile.j2`` which clones into ``/{repo.split('/')[1]}``.
* No per-repo ``MAP_REPO_VERSION_TO_SPECS``: the test command comes straight
  from ``install_config.test_cmd`` and the parser from
  ``install_config.log_parser``.
* ``install_config.install`` already ran at image *build* time, so the eval only
  applies the gold ``test_patch`` and runs ``test_cmd``.

Eval flow (transcription of upstream ``run_in_container`` adapted to preserve the
agent's solution instead of ``git reset --hard``): reset *only the test files* to
``base_commit`` -> apply the gold ``test_patch`` -> run ``test_cmd`` between the
``START``/``END`` markers -> reset the test files again. Resolution uses the
standard SWE-bench definition: every ``FAIL_TO_PASS`` passes and every
``PASS_TO_PASS`` still passes.

The grading path stays language-agnostic (parser resolved by name), so enabling
another language only means registering its parser in ``NAME_TO_PARSER``.
"""

import re
import time
import uuid
from pathlib import Path

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward import swe_rebench_v2_log_parsers as _log_parsers
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.reward.swe_rebench_v2_log_parsers import NAME_TO_PARSER, TestStatus
from uni_agent.utils import auto_await

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"


def _get_modified_files(patch: str) -> list[str]:
    """Target (``b/``) paths touched by a unified diff, order-preserved & deduped.

    A tiny self-contained replacement for ``swebench.harness.utils.get_modified_files``
    (only used to reset the test files before/after applying the test patch).
    Files deleted by the patch (``+++ /dev/null``) are skipped.
    """
    files: list[str] = []
    for line in patch.split("\n"):
        if not line.startswith("+++ "):
            continue
        path = line[4:].strip()
        # Drop trailing diff metadata (timestamps) after a tab, if present.
        path = path.split("\t")[0]
        if path == "/dev/null":
            continue
        if path.startswith("b/"):
            path = path[2:]
        if path:
            files.append(path)
    return list(dict.fromkeys(files))


def _resolve_parser(name: str):
    """Resolve a parser by name, mirroring upstream ``scripts/eval.py``:
    prefer the ``NAME_TO_PARSER`` table, then fall back to a module-level
    function of the same name."""
    parser = NAME_TO_PARSER.get(name)
    if parser is None:
        parser = getattr(_log_parsers, name, None)
    if parser is None:
        raise ValueError(f"Unknown SWE-rebench-V2 log parser: {name!r}")
    return parser


@register_reward_spec("swe_rebench_v2")
class SWEREBenchV2RewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 1800):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout

        self.instance_id = metadata["instance_id"]
        self.repo = metadata["repo"]
        # Upstream clones into /{repo.split('/')[1]} (see combine.Dockerfile.j2).
        self.repo_dir = f"/{self.repo.split('/')[1]}"
        self.log_parser_name = metadata["log_parser"]
        test_cmd = metadata["test_cmd"]
        self.test_cmds = [test_cmd] if isinstance(test_cmd, str) else list(test_cmd)

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
            # The test patch is written to its own file (rather than inlined via a
            # heredoc) so arbitrary diff content can't break the shell script.
            test_patch_path = Path(f"/tmp/srb2_test_patch_{uuid.uuid4().hex}.diff")
            await self.env.write_file(test_patch_path, self.metadata["test_patch"])

            script_path = Path(f"/tmp/srb2_eval_{uuid.uuid4().hex}.sh")
            await self.env.write_file(script_path, self._build_eval_script(test_patch_path))

            t0 = time.perf_counter()
            # `| cat` forces a non-TTY stdout (mirrors the upstream non-TTY
            # `docker run`) so runners don't emit colored/TUI output.
            output = await self.env.communicate(
                f"bash {script_path} 2>&1 | cat",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - t0
            result["eval_completed"] = True

            # Strip SGR color codes and CRs; the parsers also handle ANSI, but
            # this keeps line-based parsing robust (matches the other SWE specs).
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            eval_report = self._grade(output)
            result["eval_report"] = eval_report
            result["resolved"] = eval_report["resolved"]
            self.logger.info(
                f"SWE-rebench-V2 eval: instance={self.instance_id} repo={self.repo} "
                f"parser={self.log_parser_name} resolved={eval_report['resolved']} "
                f"found={eval_report['found_eval_status']} time={result['eval_execution_time']:.1f}s"
            )
        except Exception as exc:
            self.logger.error(f"Failed to evaluate SWE-rebench-V2 instance: {exc}")
            result["error"] = str(exc)
        return result["resolved"], result

    def _make_eval_script_list(self, test_patch_path: Path) -> list[str]:
        """Adapted from upstream ``run_in_container``: instead of ``git reset
        --hard`` (which would wipe the agent's fix) we reset *only the test files*
        to ``base_commit``, apply the gold ``test_patch``, run ``test_cmd``
        between the markers, then reset the test files again so they can't be
        tampered with. The per-file ``|| true`` tolerates newly-added test files
        that don't exist at ``base_commit``."""
        base_commit = self.metadata["base_commit"]
        test_files = _get_modified_files(self.metadata["test_patch"])
        reset_cmds = [f'git checkout {base_commit} -- "{f}" 2>/dev/null || true' for f in test_files]

        # Apply with the same lenient flags upstream uses, with fallbacks.
        apply_test_patch = (
            f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {test_patch_path.as_posix()} "
            f"|| git apply -v --recount --whitespace=nowarn {test_patch_path.as_posix()} "
            f"|| patch --batch --fuzz=5 -p1 -i {test_patch_path.as_posix()}"
        )

        return [
            "chmod 1777 /tmp 2>/dev/null || true",
            # Mirror the env the upstream `docker run` sets for (mostly Java) suites.
            "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
            f"cd {self.repo_dir}",
            f"git config --global --add safe.directory {self.repo_dir}",
            *reset_cmds,
            apply_test_patch,
            f"echo '{START_TEST_OUTPUT}'",
            *self.test_cmds,
            f"echo '{END_TEST_OUTPUT}'",
            *reset_cmds,
        ]

    def _build_eval_script(self, test_patch_path: Path) -> str:
        """Assemble the eval script. No ``set -e`` on purpose so the END marker
        and the trailing test-file reset always run even when tests fail; the
        markers are emitted with ``echo`` so they show up without ``set -x``
        (which would otherwise interleave ``+`` traces into JSON test reports)."""
        return "\n".join(["#!/bin/bash", "set -o pipefail", *self._make_eval_script_list(test_patch_path)]) + "\n"

    def _grade(self, output: str) -> dict:
        """Parse the test region and grade against FAIL_TO_PASS / PASS_TO_PASS."""
        report = {"resolved": False, "found_eval_status": False, "test_status": None}

        parser = _resolve_parser(self.log_parser_name)
        if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
            region = output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        else:
            region = output
        status_map = parser(region)
        # Fallback: some runners write results outside the markers (e.g. stderr).
        if not status_map:
            status_map = parser(output)
        if not status_map:
            self.logger.warning(
                "SWE-rebench-V2 parser matched 0 tests -- the test command likely "
                f"failed to run. Output tail:\n{output[-3000:]}"
            )
            return report

        report["found_eval_status"] = True

        passed = {name for name, st in status_map.items() if st == TestStatus.PASSED.value}
        f2p = set(self.metadata.get("FAIL_TO_PASS", []))
        p2p = set(self.metadata.get("PASS_TO_PASS", []))

        f2p_failure = sorted(f2p - passed)
        p2p_failure = sorted(p2p - passed)
        report["test_status"] = {
            "FAIL_TO_PASS": {"success": sorted(f2p & passed), "failure": f2p_failure},
            "PASS_TO_PASS": {"success": sorted(p2p & passed), "failure": p2p_failure},
        }
        # Standard SWE-bench resolution: all F2P pass and all P2P still pass.
        report["resolved"] = len(f2p) > 0 and not f2p_failure and not p2p_failure
        if not report["resolved"]:
            self.logger.warning(
                f"SWE-rebench-V2 NOT resolved: FAIL_TO_PASS still failing={f2p_failure[:25]} "
                f"PASS_TO_PASS broke={p2p_failure[:25]}"
            )
        return report

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        """Apply a patch string to the repo working tree. Tries multiple apply
        strategies in order."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/srb2_patch_{uuid.uuid4().hex}.diff")
        await self.env.write_file(patch_path, patch)
        p = patch_path.as_posix()
        commands = [
            # The lenient flags upstream uses to apply predictions/gold patches.
            f"cd {self.repo_dir} && git apply --whitespace=fix {p}",
            f"cd {self.repo_dir} && git apply --reject --whitespace=nowarn {p}",
            f"cd {self.repo_dir} && patch --batch --fuzz=5 -p1 -i {p}",
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
