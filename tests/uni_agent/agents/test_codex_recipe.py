from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

ROOT = Path(__file__).parents[3]
CODEX_DIR = ROOT / "examples" / "codex"
SAMPLE = CODEX_DIR / "train_qwen3p5_codex.sh"


def _sample_environment(tmp_path: Path, **overrides: str) -> tuple[dict[str, str], Path]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    capture = tmp_path / "ray-argv.bin"
    ray_stub = stub_dir / "ray"
    ray_stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "${CAPTURE_PATH}"\n')
    ray_stub.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{stub_dir}:{env['PATH']}",
            "CAPTURE_PATH": str(capture),
            "DATA_DIR": str(tmp_path / "data"),
            "RUNTIME_DIR": str(tmp_path / "runtime"),
            "RUNTIME_ENV": "runtime-env.yaml",
        }
    )
    env.update(overrides)
    return env, capture


def _run_sample(tmp_path: Path, **overrides: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    env, capture = _sample_environment(tmp_path, **overrides)
    result = subprocess.run(
        ["bash", str(SAMPLE)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    args = capture.read_bytes().split(b"\0")[:-1] if capture.exists() else []
    return result, [arg.decode() for arg in args]


def test_codex_task_config_uses_framework_agent_and_sandbox_mount():
    import yaml

    entries = yaml.safe_load((CODEX_DIR / "task_config_codex.yaml").read_text(encoding="utf-8"))
    assert {entry["name"] for entry in entries} == {"swe_rebench", "swe_bench"}
    for entry in entries:
        assert entry["sandbox"]["provider"] == "openyuanrong"
        assert entry["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
        assert entry["agent"]["name"] == "codex"
        mount = entry["sandbox"]["sandbox_kwargs"]["mounts"][0]
        assert mount["target"] == "/opt/codex"
        assert mount["image_url"].endswith("codex-tool:0.147.0-direct-stdin")


def test_codex_sample_uses_community_job_shape_and_required_overrides():
    sample = SAMPLE.read_text(encoding="utf-8")
    assert "ray job submit --no-wait --runtime-env" in sample
    assert "python3 -m verl.trainer.main_ppo" in sample
    assert "examples/codex/task_config_codex.yaml" in sample
    assert "++actor_rollout_ref.rollout.custom.agent_framework.max_tokens_per_turn=${MAX_TOKENS_PER_TURN}" in sample
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True" in sample
    assert "actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser" in sample


def test_codex_sample_default_concurrency_reaches_runner_argv(tmp_path):
    result, args = _run_sample(tmp_path)
    assert result.returncode == 0, result.stderr
    assert args[:3] == ["job", "submit", "--no-wait"]
    assert "--" in args
    assert "python3" in args
    assert "-m" in args
    assert "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=1800" in args
    assert "trainer.save_freq=-1" in args
    assert "verl.trainer.main_ppo" in args
    assert "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=1" in args
    assert "++actor_rollout_ref.rollout.custom.agent_framework.max_tokens_per_turn=8192" in args
    assert "data.max_prompt_length=8192" in args
    assert "data.max_response_length=122880" in args
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.62" in args
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True" in args


def test_codex_sample_custom_concurrency_reaches_runner_argv(tmp_path):
    result, args = _run_sample(tmp_path, CONCURRENCY="3")
    assert result.returncode == 0, result.stderr
    assert "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=3" in args
    assert not any("max_concurrent_sessions=1" in arg for arg in args)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_codex_sample_rejects_invalid_concurrency_before_ray(tmp_path, value):
    result, args = _run_sample(tmp_path, CONCURRENCY=value)
    assert result.returncode == 2
    assert "CONCURRENCY must be a positive integer" in result.stderr
    assert args == []
