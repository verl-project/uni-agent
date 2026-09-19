from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from uni_agent.agents import AgentResult
from uni_agent.sandbox import ExecResult
from uni_agent.tasks.kernel_bench.task import (
    TritonOperatorTask,
    TritonOperatorTaskConfig,
    _read_json,
    _snapshot_implementation_status,
    _train_best,
    _validate_agent_verify_entrypoint,
    _validate_trusted_image,
)
from uni_agent.tasks.kernel_bench.workspace_snapshot import WorkspaceSnapshot, file_limits

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

_CURRENT_IMPL = b"class ModelNew:\n    def forward(self, value):\n        return value\n"
_BEST_IMPL = b"class ModelNew:\n    def forward(self, value):\n        return value + 1\n"
_STAGED_IMPL = b"class ModelNew:\n    def forward(self, value):\n        return value * 2\n"


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _metrics(*, passed: int = 2, total: int = 2, **extra: Any) -> dict[str, Any]:
    return {
        "success": passed == total,
        "ast_check_ok": True,
        "compile_ok": True,
        "correctness_ok": passed == total,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        **extra,
    }


class FakeSandbox:
    def __init__(
        self,
        *,
        has_impl: bool = True,
        agent_files: dict[str, bytes | str] | None = None,
        events: list[str] | None = None,
        label: str = "sandbox",
        uid: int = 1000,
        trusted_file_type: str = "regular file",
        trusted_parent_mode: str = "755",
        verify_entrypoint_target: str = "/opt/triton-agent-tools/verify_once.sh",
        unsafe_directories: set[str] | None = None,
        mtimes: dict[str, int] | None = None,
        agent_finished: bool = True,
    ) -> None:
        self.started = False
        self.stopped = False
        self.cleanup_calls = 0
        self.events = events
        self.label = label
        self.produce_impl = has_impl
        self.agent_files = dict(agent_files or {})
        self.uid = uid
        self.trusted_file_type = trusted_file_type
        self.trusted_parent_mode = trusted_parent_mode
        self.verify_entrypoint_target = verify_entrypoint_target
        self.unsafe_directories = set(unsafe_directories or ())
        self.mtimes = dict(mtimes or {})
        self.agent_finished = agent_finished
        self.files: dict[str, bytes] = {}
        self.file_types: dict[str, str] = {}
        self.read_file_calls: list[str] = []
        self.download_file_calls: list[str] = []
        self.shell_calls: list[str] = []
        self.snapshot_calls: list[str] = []

    async def __aenter__(self):
        self.started = True
        if self.events is not None:
            self.events.append(f"{self.label}:enter")
        return self

    async def __aexit__(self, *exc):
        self.stopped = True
        if self.events is not None:
            self.events.append(f"{self.label}:exit")

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        del timeout, workdir, env
        self.shell_calls.append(script)
        if script == "/trusted/cleanup":
            self.cleanup_calls += 1
        if script.startswith("test -d ") and any(path in script for path in self.unsafe_directories):
            return ExecResult(1, "", "unsafe directory")
        return ExecResult(0, "", "")

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        del timeout, workdir, env
        if argv[:3] == ["python3", "-I", "-c"]:
            request = json.loads(argv[-1])
            kind = request["kind"]
            self.snapshot_calls.append(kind)
            if kind == "recover":
                best_path = "src/smoke_triton_ascend_impl_best.py"
                self.files[f"/workspace/{best_path}"] = self.files[
                    "/workspace/output/verify/smoke_triton_ascend_impl.py"
                ]
                self.files["/workspace/metrics_best.json"] = _json_bytes(
                    {
                        **request["metrics"],
                        "implementation_path": best_path,
                        "implementation_sha256": request["staged_digest"],
                    }
                )
            payload = self.snapshot_payload("evaluate" if kind == "recover" else kind)
            return ExecResult(0, json.dumps(payload), "")
        if argv == ["pwd"]:
            return ExecResult(0, "/workspace\n", "")
        if argv == ["id", "-u"]:
            return ExecResult(0, f"{self.uid}\n", "")
        if argv[:2] == ["readlink", "-f"]:
            target = self.verify_entrypoint_target if argv[-1].endswith("tools/verify_once.sh") else argv[-1]
            return ExecResult(0, f"{target}\n", "")
        if argv[:2] == ["test", "-f"]:
            return ExecResult(int(argv[2] not in self.files), "", "")
        if argv[:2] == ["test", "-s"]:
            return ExecResult(int(not self.files.get(argv[2])), "", "")
        if argv[:2] == ["rm", "-f"]:
            for path in argv[2:]:
                if path != "--":
                    self.files.pop(path, None)
            return ExecResult(0, "", "")
        if argv[:3] == ["rm", "-rf", "--"]:
            prefix = argv[3].rstrip("/") + "/"
            for path in list(self.files):
                if path == argv[3] or path.startswith(prefix):
                    self.files.pop(path, None)
            return ExecResult(0, "", "")
        if argv[:3] == ["stat", "-c", "%F:%s"]:
            content = self.files.get(argv[-1])
            return (
                ExecResult(0, f"{self.file_types.get(argv[-1], 'regular file')}:{len(content)}\n", "")
                if content is not None
                else ExecResult(1, "", "missing")
            )
        if argv[:3] == ["stat", "-c", "%u:%a:%F"]:
            is_tool = argv[3].endswith((".py", ".sh"))
            if is_tool:
                return ExecResult(0, f"0:755:{self.trusted_file_type}\n", "")
            return ExecResult(0, f"0:{self.trusted_parent_mode}:directory\n", "")
        return ExecResult(0, "", "")

    def snapshot_payload(self, kind="evaluate"):
        """Model only the remote transport; host snapshot validation stays real."""
        files = {}
        for relative, limit in file_limits("smoke", kind).items():
            path = f"/workspace/{relative}"
            raw = self.files.get(path)
            if any(path.startswith(f"{directory}/") for directory in self.unsafe_directories):
                files[relative] = {"status": "unsafe_path"}
            elif raw is None:
                files[relative] = {"status": "missing"}
            elif self.file_types.get(path, "regular file") != "regular file":
                files[relative] = {"status": "not_regular"}
            elif len(raw) > limit:
                files[relative] = {"status": "too_large"}
            else:
                mtime = self.mtimes.get(path, 10 if path.endswith("_triton_ascend_impl.py") else 20)
                files[relative] = {
                    "status": "ok",
                    "size": len(raw),
                    "mtime_ns": mtime * 1_000_000_000,
                    "ctime_ns": mtime * 1_000_000_000,
                    "device": 1,
                    "inode": 1,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "data": base64.b64encode(raw).decode(),
                }
        return {
            "version": 1,
            "workspace": "/workspace",
            "op_name": "smoke",
            "kind": kind,
            "files": files,
            "directories": {
                relative: ("/workspace" if relative == "." else f"/workspace/{relative}")
                not in self.unsafe_directories
                for relative in (".", "src", "output", "output/verify")
            },
        }

    async def write_file(self, path, content):
        self.files[path] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path):
        self.read_file_calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def download_file(self, remote_path, local_path):
        self.download_file_calls.append(remote_path)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[remote_path])


class FakeAgent:
    async def run(self, *, sandbox, messages):
        del messages
        if sandbox.produce_impl:
            sandbox.files["/workspace/src/smoke_triton_ascend_impl.py"] = _CURRENT_IMPL
        for path, content in sandbox.agent_files.items():
            sandbox.files[path] = content.encode() if isinstance(content, str) else content
        return AgentResult(info={"exit_code": 0}, finished=sandbox.agent_finished)


def _config(*, artifact_dir: Path | None = None) -> TritonOperatorTaskConfig:
    return TritonOperatorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code", "model": {"base_url": "http://unused", "model_name": "unused"}},
        prompt=[{"role": "user", "content": "implement"}],
        metadata={"uid": "uid-1", "op_name": "smoke", "task_code": "class Model: pass\n"},
        template_dir=None,
        cleanup_command="/trusted/cleanup",
        artifact_dir=str(artifact_dir) if artifact_dir else None,
    )


def _evaluate(
    sandbox: FakeSandbox,
    *,
    initial_current: str | None = None,
    initial_best: str | None = None,
) -> dict[str, Any]:
    config = _config()
    task = TritonOperatorTask(config)
    return asyncio.run(
        task._evaluate_workspace(  # noqa: SLF001 - focused recipe behavior test
            sandbox,
            config,
            "/workspace",
            "smoke",
            initial_impl_digests={"current": initial_current, "best": initial_best},
        )
    )


def test_task_config_uses_only_agent_time_verifier() -> None:
    config = TritonOperatorTaskConfig(sandbox={"provider": "docker"})
    assert not hasattr(config, "verify_command")
    assert not hasattr(config, "verify_timeout")
    assert config.trusted_agent_verify_command == "/opt/triton-agent-tools/verify_once.sh"
    assert all("final_verify" not in path for path in config.trusted_tool_paths)


def test_workspace_template_copy_drops_image_ownership_and_mode() -> None:
    sandbox = FakeSandbox()
    config = _config().model_copy(
        update={
            "template_dir": "/opt/triton-agent-template",
            "enforce_trusted_image": False,
            "npu_lease_required": False,
            "install_transcript_hooks": False,
        }
    )

    asyncio.run(TritonOperatorTask(config)._prepare_workspace(sandbox, config, config.metadata, "smoke"))

    assert any("cp -a --no-preserve=ownership,mode" in command for command in sandbox.shell_calls)


def test_matching_best_pair_has_priority_and_preserves_train_best() -> None:
    sandbox = FakeSandbox(has_impl=False)
    digest = hashlib.sha256(_BEST_IMPL).hexdigest()
    best = _metrics(
        perf_data={"speedup_vs_torch": 2.0},
        reward=999,
        reward_components={"all_correct_bonus": 999},
        assistant_snapshot_impl_sha256=digest,
        assistant_index=2,
        assistant_messages_seen=3,
        assistant_index_source="claude_code_transcript_hook",
    )
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl_best.py": _BEST_IMPL,
            "/workspace/metrics_best.json": _json_bytes(best),
            "/workspace/src/smoke_triton_ascend_impl.py": _CURRENT_IMPL,
            "/workspace/metrics.json": _json_bytes(_metrics(passed=1)),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics_best"
    assert evaluation["best_source"] == "best_pair"
    assert sandbox.snapshot_calls == ["evaluate"]
    assert evaluation["used_best_metrics"] is True
    assert evaluation["metrics"]["metrics_impl_binding"] == "matched"
    assert evaluation["metrics"]["reward"] == 1.0
    assert evaluation["metrics"]["reward_components"] == {
        "ast": 0.05,
        "compile": 0.25,
        "correctness": 0.4,
        "speedup": 0.3,
        "raw_speedup": 2.0,
        "total": 1.0,
    }
    assert evaluation["train_best"]["assistant_index"] == 2


def test_legacy_unbound_best_pair_is_accepted_without_trajectory_hint() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl_best.py": _BEST_IMPL,
            "/workspace/metrics_best.json": _json_bytes(_metrics()),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics_best"
    assert evaluation["metrics"]["metrics_impl_binding"] == "legacy_unbound"
    assert "train_best" not in evaluation


def test_best_digest_mismatch_falls_back_to_current_metrics() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl_best.py": _BEST_IMPL,
            "/workspace/metrics_best.json": _json_bytes(_metrics(assistant_snapshot_impl_sha256="0" * 64)),
            "/workspace/src/smoke_triton_ascend_impl.py": _CURRENT_IMPL,
            "/workspace/metrics.json": _json_bytes(_metrics(passed=1)),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics"
    assert evaluation["used_best_metrics"] is False
    assert evaluation["metrics"]["pass_rate"] == 0.5


def test_verifier_artifacts_recover_the_staged_implementation_as_best() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/output/verify/smoke_triton_ascend_impl.py": _STAGED_IMPL,
            "/workspace/output/verify/verify_result.json": _json_bytes(
                {"total_cases": 2, "passed_cases": 1, "failed_cases": 1, "compile_ok": True}
            ),
            "/workspace/output/verify/verify_result_summary.json": _json_bytes({"success": False}),
            "/workspace/output/verify/perf_result.json": _json_bytes({"speedup_vs_torch": 2.0}),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics_best"
    assert evaluation["best_source"] == "verifier_artifacts"
    assert sandbox.snapshot_calls == ["evaluate", "recover"]
    assert evaluation["used_best_metrics"] is True
    assert evaluation["metrics"]["pass_rate"] == 0.5
    assert evaluation["metrics"]["reward"] == 0.5
    assert evaluation["metrics"]["reward_components"]["raw_speedup"] == 0.0
    assert sandbox.files["/workspace/src/smoke_triton_ascend_impl_best.py"] == _STAGED_IMPL
    recovered = json.loads(sandbox.files["/workspace/metrics_best.json"])
    assert recovered["passed_cases"] == 1
    assert recovered["implementation_sha256"] == hashlib.sha256(_STAGED_IMPL).hexdigest()
    assert "/workspace/metrics.json" not in sandbox.files


def test_full_correctness_artifacts_reuse_valid_performance() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/output/verify/smoke_triton_ascend_impl.py": _STAGED_IMPL,
            "/workspace/output/verify/verify_result.json": _json_bytes(
                {"total_cases": 2, "passed_cases": 2, "failed_cases": 0, "compile_ok": True}
            ),
            "/workspace/output/verify/perf_result.json": _json_bytes({"speedup_vs_torch": 1.0}),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["best_source"] == "verifier_artifacts"
    assert evaluation["metrics"]["reward"] == 0.85
    assert evaluation["metrics"]["reward_components"]["raw_speedup"] == 1.0


def test_verifier_artifact_digest_mismatch_is_not_recovered() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/output/verify/smoke_triton_ascend_impl.py": _STAGED_IMPL,
            "/workspace/output/verify/verify_result.json": _json_bytes(
                {
                    "total_cases": 2,
                    "passed_cases": 2,
                    "failed_cases": 0,
                    "compile_ok": True,
                    "verified_impl_sha256": "0" * 64,
                }
            ),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "not_run_missing_impl"
    assert "/workspace/src/smoke_triton_ascend_impl_best.py" not in sandbox.files


def test_conflicting_verify_and_perf_digests_are_not_recovered() -> None:
    digest = hashlib.sha256(_STAGED_IMPL).hexdigest()
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/output/verify/smoke_triton_ascend_impl.py": _STAGED_IMPL,
            "/workspace/output/verify/verify_result.json": _json_bytes(
                {
                    "total_cases": 2,
                    "passed_cases": 2,
                    "failed_cases": 0,
                    "compile_ok": True,
                    "verified_impl_sha256": digest,
                }
            ),
            "/workspace/output/verify/perf_result.json": _json_bytes(
                {"speedup_vs_torch": 1.0, "verified_impl_sha256": "0" * 64}
            ),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "not_run_missing_impl"
    assert "/workspace/src/smoke_triton_ascend_impl_best.py" not in sandbox.files


def test_staged_implementation_newer_than_results_is_not_recovered() -> None:
    staged_path = "/workspace/output/verify/smoke_triton_ascend_impl.py"
    verify_path = "/workspace/output/verify/verify_result.json"
    sandbox = FakeSandbox(has_impl=False, mtimes={staged_path: 30, verify_path: 20})
    sandbox.files.update(
        {
            staged_path: _STAGED_IMPL,
            verify_path: _json_bytes({"total_cases": 2, "passed_cases": 2, "failed_cases": 0, "compile_ok": True}),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "not_run_missing_impl"
    assert "/workspace/src/smoke_triton_ascend_impl_best.py" not in sandbox.files


@pytest.mark.parametrize(
    "verify",
    [
        {"total_cases": 0, "passed_cases": 0, "failed_cases": 0},
        {"total_cases": 2, "passed_cases": 2, "failed_cases": 1},
        {"total_cases": 2, "passed_cases": True, "failed_cases": 1},
    ],
)
def test_invalid_staged_artifacts_are_missing_impl(verify: dict[str, Any]) -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/output/verify/smoke_triton_ascend_impl.py": _STAGED_IMPL,
            "/workspace/output/verify/verify_result.json": _json_bytes(verify),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "not_run_missing_impl"
    assert evaluation["metrics"]["error_type"] == "missing_impl"


def test_current_metrics_are_the_last_fallback() -> None:
    sandbox = FakeSandbox(has_impl=False)
    digest = hashlib.sha256(_CURRENT_IMPL).hexdigest()
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl.py": _CURRENT_IMPL,
            "/workspace/metrics.json": _json_bytes(_metrics(passed=1, implementation_sha256=digest, reward=1234)),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics"
    assert evaluation["used_best_metrics"] is False
    assert evaluation["metrics"]["metrics_impl_binding"] == "matched"
    assert evaluation["metrics"]["reward"] == 0.5


def test_explicit_ast_failure_is_not_rewritten_as_success() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl.py": _CURRENT_IMPL,
            "/workspace/metrics.json": _json_bytes(_metrics(ast_check_ok=False)),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["metrics"]["ast_check_ok"] is False
    assert evaluation["metrics"]["reward_components"]["ast"] == 0.0
    assert evaluation["metrics"]["reward"] == 0.65


def test_legacy_compile_only_current_metrics_keep_partial_credit() -> None:
    sandbox = FakeSandbox(has_impl=False)
    sandbox.files.update(
        {
            "/workspace/src/smoke_triton_ascend_impl.py": _CURRENT_IMPL,
            "/workspace/metrics.json": _json_bytes({"verify_exit": 1, "compile_ok": True}),
        }
    )

    evaluation = _evaluate(sandbox)

    assert evaluation["selected_metrics_source"] == "metrics"
    assert evaluation["metrics"]["total_cases"] == 0
    assert evaluation["metrics"]["compile_ok"] is True
    assert evaluation["metrics"]["correctness_ok"] is False
    assert evaluation["metrics"]["reward"] == 0.25


def test_missing_impl_and_missing_metrics_are_distinct() -> None:
    missing_impl = _evaluate(FakeSandbox(has_impl=False))
    assert missing_impl["selected_metrics_source"] == "not_run_missing_impl"
    assert missing_impl["metrics"]["error_type"] == "missing_impl"

    sandbox = FakeSandbox(has_impl=False)
    sandbox.files["/workspace/src/smoke_triton_ascend_impl.py"] = _CURRENT_IMPL
    missing_metrics = _evaluate(sandbox)
    assert missing_metrics["selected_metrics_source"] == "missing_metrics"
    assert missing_metrics["metrics"]["error_type"] == "missing_metrics"


def test_task_reuses_agent_metrics_without_final_reverify_or_output_reset(tmp_path: Path, capsys) -> None:
    preserved = b'{"agent-time":true}\n'
    progress = {"verify_count": 5, "correctness_stale_verify_count": 0, "latency_stale_verify_count": 2}
    sandbox = FakeSandbox(
        agent_files={
            "/workspace/metrics.json": _json_bytes(_metrics()),
            "/workspace/output/verify/agent-controlled.json": preserved,
            "/workspace/.triton_verify_patience.json": _json_bytes(progress),
        }
    )
    config = _config(artifact_dir=tmp_path)
    config.metadata["runtime"] = {"sample_index": 2, "session_id": "session-sample-2-rollout-5-test"}
    task = TritonOperatorTask(config)
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    summary = capsys.readouterr().out
    assert summary.count("[triton-result]") == 1
    assert "sample=2 session=session-sample-2-rollout-5-test op=smoke correct=True" in summary
    assert "reward=0.7000 pass_rate=1.0000 source=metrics finished=True" in summary
    assert sandbox.started and sandbox.stopped
    assert sandbox.cleanup_calls == 2
    assert sandbox.snapshot_calls == ["baseline", "evaluate"]
    assert (tmp_path / "uid-1" / "session-sample-2-rollout-5-test" / "metrics.json").is_file()
    assert all("final_verify" not in command for command in sandbox.shell_calls)
    assert sandbox.files["/workspace/output/verify/agent-controlled.json"] == preserved
    assert result.finished is True
    assert result.accuracy == 1.0
    assert result.reward == 0.7
    assert result.extra_info is not None
    assert result.extra_info["selected_metrics_source"] == "metrics"
    assert result.extra_info["agent"]["verify_progress"] == progress
    assert result.extra_info["metrics"]["reward_components"]["total"] == 0.7
    assert "task_code" not in result.extra_info
    settings = json.loads(sandbox.files["/workspace/.claude/settings.json"])
    assert {"PreToolUse", "PostToolUse", "PostToolUseFailure"} <= settings["hooks"].keys()
    assert settings["hooks"]["PostToolUseFailure"] == settings["hooks"]["PostToolUse"]
    policy = json.loads(sandbox.files["/workspace/.claude/hooks/triton_verify_policy.json"])
    assert policy == {"correctness_patience": 7, "correctness_min_reward": 0.15, "latency_patience": 3}


def test_early_stop_is_finished_only_with_a_trainable_best_prefix() -> None:
    def run(sandbox: FakeSandbox):
        task = TritonOperatorTask(_config())
        task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
        task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]
        return asyncio.run(task.run())

    digest = hashlib.sha256(_BEST_IMPL).hexdigest()
    stop = {
        "reason": "no_latency_improvement",
        "mode": "latency",
        "patience": 3,
        "stale_verify_count": 3,
        "verify_count": 5,
    }
    best_metrics = _metrics(
        assistant_index=1,
        assistant_messages_seen=2,
        assistant_index_source="claude_code_transcript_hook",
        assistant_snapshot_impl_sha256=digest,
    )
    best = FakeSandbox(
        agent_finished=False,
        agent_files={
            "/workspace/src/smoke_triton_ascend_impl_best.py": _BEST_IMPL,
            "/workspace/metrics_best.json": _json_bytes(best_metrics),
            "/workspace/.triton_verify_stop.json": _json_bytes(stop),
        },
    )
    best_result = run(best)
    assert best_result.finished is True
    assert best_result.extra_info["agent"]["early_stop"]["reason"] == "no_latency_improvement"

    current = FakeSandbox(
        agent_finished=False,
        agent_files={
            "/workspace/metrics.json": _json_bytes(_metrics()),
            "/workspace/.triton_verify_stop.json": _json_bytes(stop),
        },
    )
    assert run(current).finished is False


@pytest.mark.parametrize("has_impl,error_type", [(False, "missing_impl"), (True, "missing_metrics")])
def test_unsuccessful_task_uses_one_sandbox(has_impl, error_type) -> None:
    events: list[str] = []
    sandbox = FakeSandbox(has_impl=has_impl, events=events, label="only")
    task = TritonOperatorTask(_config())
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert events == ["only:enter", "only:exit"]
    assert result.reward == 0.0
    assert result.extra_info["metrics"]["error_type"] == error_type
    assert set(result.extra_info["timing_ms"]) == {"setup", "agent", "evaluate", "total"}


def _implementation_shape(sandbox: FakeSandbox, path: str) -> dict[str, Any]:
    snapshot = WorkspaceSnapshot(sandbox.snapshot_payload(), "/workspace", "smoke", "evaluate")
    return _snapshot_implementation_status(snapshot, path.removeprefix("/workspace/"), baseline_digest=None)


def test_stub_or_missing_modelnew_is_not_a_generated_implementation() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = b"class ModelNew:\n    raise NotImplementedError('Replace this stub')\n"

    status = _implementation_shape(sandbox, path)
    assert status["substantive"] is False
    assert status["reason"] == "not_implemented"

    sandbox.files[path] = b"def helper():\n    return 1\n"
    status = _implementation_shape(sandbox, path)
    assert status["substantive"] is False
    assert status["reason"] == "missing_model_new"

    sandbox.files[path] = b"class ModelNew:\n    pass\n"
    status = _implementation_shape(sandbox, path)
    assert status["substantive"] is False
    assert status["reason"] == "not_implemented"


def test_implementation_shape_uses_ast_not_comments_or_substrings() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = b'# class ModelNew: pass\ndef helper():\n    return "class ModelNew"\n'
    status = _implementation_shape(sandbox, path)
    assert status["reason"] == "missing_model_new"

    sandbox.files[path] = (
        b"class ModelNew:\n"
        b'    """NotImplementedError is discussed here, not raised."""\n'
        b"    def forward(self, value):\n"
        b"        return value\n"
    )
    status = _implementation_shape(sandbox, path)
    assert status["substantive"] is True


def test_agent_symlink_implementation_is_rejected_without_reading_target() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/src/smoke_triton_ascend_impl.py"
    sandbox.files[path] = _CURRENT_IMPL
    sandbox.file_types[path] = "symbolic link"

    status = _implementation_shape(sandbox, path)

    assert status["substantive"] is False
    assert status["reason"] == "missing_or_empty"
    assert path not in sandbox.read_file_calls


def test_json_reads_reject_symlinks_oversize_and_non_finite_values() -> None:
    sandbox = FakeSandbox()
    path = "/workspace/metrics_best.json"
    sandbox.files[path] = b'{"assistant_index": 1}'
    sandbox.file_types[path] = "symbolic link"
    assert asyncio.run(_read_json(sandbox, path)) is None
    assert path not in sandbox.read_file_calls

    sandbox.file_types[path] = "regular file"
    sandbox.files[path] = b"x" * (1024 * 1024 + 1)
    assert asyncio.run(_read_json(sandbox, path)) is None
    assert path not in sandbox.read_file_calls

    sandbox.files[path] = b'{"speedup": Infinity}'
    assert asyncio.run(_read_json(sandbox, path)) is None


def test_best_hint_requires_consistent_native_hook_metadata() -> None:
    digest = "a" * 64
    valid = {
        "assistant_snapshot_impl_sha256": digest,
        "assistant_index": 2,
        "assistant_messages_seen": 3,
        "assistant_index_source": "claude_code_transcript_hook",
    }
    assert _train_best(valid, "metrics_best", best_impl_digest=digest)["assistant_index"] == 2
    assert _train_best({**valid, "assistant_messages_seen": 4}, "metrics_best", best_impl_digest=digest) == {}
    assert _train_best({**valid, "assistant_index": float("inf")}, "metrics_best", best_impl_digest=digest) == {}


def test_trusted_image_rejects_root_agent_user() -> None:
    with pytest.raises(RuntimeError, match="non-root"):
        asyncio.run(_validate_trusted_image(FakeSandbox(uid=0), ("/opt/triton-agent-tools/verify_once.sh",)))


def test_trusted_image_rejects_symlink_tool_or_writable_parent() -> None:
    with pytest.raises(RuntimeError, match="root-owned, executable"):
        asyncio.run(
            _validate_trusted_image(
                FakeSandbox(trusted_file_type="symbolic link"),
                ("/opt/triton-agent-tools/verify_once.sh",),
            )
        )
    with pytest.raises(RuntimeError, match="trusted tool parent"):
        asyncio.run(
            _validate_trusted_image(
                FakeSandbox(trusted_parent_mode="777"),
                ("/opt/triton-agent-tools/verify_once.sh",),
            )
        )


def test_agent_verify_entrypoint_must_resolve_to_trusted_command() -> None:
    with pytest.raises(RuntimeError, match="immutable trusted command"):
        asyncio.run(
            _validate_agent_verify_entrypoint(
                FakeSandbox(verify_entrypoint_target="/workspace/agent-script.sh"),
                "/workspace",
                "tools/verify_once.sh",
                "/opt/triton-agent-tools/verify_once.sh",
            )
        )


@pytest.mark.parametrize("source,expected_calls", [("best", 1), ("staged", 2)])
def test_workspace_snapshot_executes_real_collector(tmp_path: Path, source: str, expected_calls: int) -> None:
    """Exercise the collector/recovery protocol without Docker or accelerator work."""
    (tmp_path / "src").mkdir()
    (tmp_path / "output/verify").mkdir(parents=True)
    if source == "best":
        (tmp_path / "src/smoke_triton_ascend_impl_best.py").write_bytes(_BEST_IMPL)
        (tmp_path / "metrics_best.json").write_bytes(_json_bytes(_metrics()))
    else:
        (tmp_path / "output/verify/smoke_triton_ascend_impl.py").write_bytes(_STAGED_IMPL)
        (tmp_path / "output/verify/verify_result.json").write_bytes(_json_bytes(_metrics()))

    class LocalSnapshotSandbox:
        calls = 0

        async def exec(self, argv, *, timeout=None):
            self.calls += 1
            result = subprocess.run(
                [sys.executable, *argv[1:]], capture_output=True, text=True, timeout=timeout, check=False
            )
            return ExecResult(result.returncode, result.stdout, result.stderr)

    sandbox = LocalSnapshotSandbox()
    config = _config()
    evaluation = asyncio.run(
        TritonOperatorTask(config)._evaluate_workspace(
            sandbox, config, str(tmp_path), "smoke", initial_impl_digests={"current": None, "best": None}
        )
    )
    assert sandbox.calls == expected_calls
    assert evaluation["selected_metrics_source"] == "metrics_best"
    assert evaluation["metrics"]["reward"] == 0.7
    expected_impl = _BEST_IMPL if source == "best" else _STAGED_IMPL
    assert (tmp_path / "src/smoke_triton_ascend_impl_best.py").read_bytes() == expected_impl
