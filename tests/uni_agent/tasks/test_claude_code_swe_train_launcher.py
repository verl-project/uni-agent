from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level0,
    pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="requires a POSIX shell"),
]

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/claude_code_swe_task/run_train.sh"


@pytest.fixture
def submit(tmp_path):
    capture = tmp_path / "submission.json"
    ray = tmp_path / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['RAY_SUBMISSION_CAPTURE'], 'w') as f:\n"
        "    json.dump(sys.argv[1:], f)\n"
    )
    ray.chmod(0o700)
    runtime_env = tmp_path / "runtime-env.yaml"
    runtime_env.write_text("{}\n")

    def run(**overrides):
        env = os.environ.copy()
        for key in (
            "ROUTER_CONFIG",
            "ROUTER_LOAD_THRESHOLD",
            "ROUTER_KV_EVENTS",
            "UNI_AGENT_ROUTER_DEBUG",
            "UNI_AGENT_ROUTER_SLOW_CUT",
            "UNI_AGENT_ROUTER_OVERLOAD_MODE",
            "UNI_AGENT_ROUTER_DO_SHORTCUT",
        ):
            env.pop(key, None)
        env.update(
            DATA_DIR=str(tmp_path / "data"),
            RUNTIME_DIR=str(tmp_path / "runtime"),
            RUNTIME_ENV=str(runtime_env),
            MODEL_PATH=str(tmp_path / "model"),
            PATH=f"{tmp_path}{os.pathsep}{env['PATH']}",
            RAY_SUBMISSION_CAPTURE=str(capture),
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        args = json.loads(capture.read_text()) if capture.exists() else None
        return result, args

    return run


def test_training_submission_disables_router_by_default(submit):
    result, args = submit()

    assert result.returncode == 0, result.stderr
    assert args is not None
    assert not any("router_config_path" in arg or "kv-events-config" in arg for arg in args)


def test_training_submission_passes_router_and_kv_event_settings(submit, tmp_path):
    router_config = tmp_path / "router config.yaml"
    router_config.write_text("router: {}\n")

    result, args = submit(
        ROUTER_CONFIG=str(router_config),
        ROUTER_LOAD_THRESHOLD="0.75",
        UNI_AGENT_ROUTER_DEBUG="1",
        UNI_AGENT_ROUTER_SLOW_CUT="least-inflight",
        UNI_AGENT_ROUTER_OVERLOAD_MODE="least-inflight",
        UNI_AGENT_ROUTER_DO_SHORTCUT="0",
    )

    assert result.returncode == 0, result.stderr
    assert args is not None
    assert f'++actor_rollout_ref.rollout.router_config_path="{router_config}"' in args
    assert "++actor_rollout_ref.rollout.custom.agent_framework.router.load_threshold=0.75" in args
    assert "++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-events-config.enable_kv_cache_events=True" in args
    assert '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-events-config.publisher="zmq"' in args
    assert '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-events-config.topic="kv-events"' in args
    assert '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-events-config.endpoint="tcp://*:0"' in args
    assert '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv-events-config.replay_endpoint="tcp://*:0"' in args
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_ROUTER_DEBUG="1"' in args
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_ROUTER_SLOW_CUT="least-inflight"' in args
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_ROUTER_OVERLOAD_MODE="least-inflight"' in args
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_ROUTER_DO_SHORTCUT="0"' in args


def test_training_submission_can_disable_kv_event_settings(submit, tmp_path):
    router_config = tmp_path / "router.yaml"
    router_config.write_text("router: {}\n")

    result, args = submit(ROUTER_CONFIG=str(router_config), ROUTER_KV_EVENTS="False")

    assert result.returncode == 0, result.stderr
    assert args is not None
    assert any("router_config_path" in arg for arg in args)
    assert not any("kv-events-config" in arg for arg in args)


def test_training_submission_rejects_missing_router_config(submit, tmp_path):
    missing_config = tmp_path / "missing-router.yaml"

    result, args = submit(ROUTER_CONFIG=str(missing_config))

    assert result.returncode != 0
    assert f"Router config does not exist or is not readable: {missing_config}" in result.stderr
    assert args is None
