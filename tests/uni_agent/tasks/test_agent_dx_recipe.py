from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.tasks import TaskConfigResolver, get_task

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("task_name", "image", "expected_image"),
    [
        ("swe_bench", "swebench/example", "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/example:v2"),
        (
            "swe_rebench",
            "swerebench/sweb.eval.x86_64.example",
            "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-rebench/example:latest",
        ),
    ],
)
def test_agent_dx_recipe_resolves_to_task_and_provider_without_network(task_name, image, expected_image):
    resolver = TaskConfigResolver.from_file(str(ROOT / "examples/mini_swe_agent/task_config_agent_dx.yaml"))
    task = resolver.resolve(
        {"name": task_name, "sandbox": {"image": image}, "metadata": {"base_commit": "abc"}},
        runtime_model={"base_url": "http://gateway:8000/sessions/test/v1", "model_name": "test", "api_key": "test"},
    )
    config = get_task(task).config
    assert config.sandbox.provider == "agent_dx"
    sandbox = build_sandbox(SandboxConfig.model_validate(task["sandbox"]))
    assert sandbox.supports_reverse_tunnel and sandbox.image == config.sandbox.image == expected_image
    assert config.sandbox.runtime_timeout > config.agent.run_timeout + config.eval_timeout


@pytest.fixture
def launcher(tmp_path):
    ray = tmp_path / "ray"
    capture = tmp_path / "capture.json"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['status']: raise SystemExit(0)\n"
        "args = sys.argv[1:]\n"
        "data = {'args': args}\n"
        "if '--runtime-env' in args:\n"
        "    path = Path(args[args.index('--runtime-env') + 1])\n"
        "    data.update(env=json.loads(path.read_text()), mode=path.stat().st_mode & 0o777, runtime_file=str(path))\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(data))\n"
    )
    ray.chmod(0o700)

    def run(script, *, provider="agent_dx", session_timeout=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OPENYUANRONG_", "ADX_"))
            and key
            not in {"TASK_CONFIG", "RUNTIME_ENV", "RAY_SUBMIT_MODE", "SANDBOX_PROVIDER", "SESSION_TIMEOUT_SECONDS"}
        }
        env.update(
            PATH=f"{tmp_path}:{Path(sys.executable).parent}:{env['PATH']}",
            SANDBOX_PROVIDER=provider,
            RAY_SUBMIT_MODE="job",
            CAPTURE=str(capture),
        )
        if provider == "agent_dx":
            env.update(ADX_SERVER_ADDRESS="adx.invalid:7443", ADX_TOKEN="fixture-not-a-real-token", ADX_TLS="1")
        else:
            env.update(OPENYUANRONG_SERVER_ADDRESS="yuanrong.invalid", OPENYUANRONG_TOKEN="fixture-token")
        if session_timeout is not None:
            env["SESSION_TIMEOUT_SECONDS"] = str(session_timeout)
        process = subprocess.run(
            ["bash", str(ROOT / "examples/mini_swe_agent" / script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert process.returncode == 0, process.stderr
        return json.loads(capture.read_text()), process

    return run


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="requires Bash")
@pytest.mark.parametrize("script", ["run_infer_mini_swe_agent.sh", "run_train.sh"])
def test_agent_dx_launchers_forward_credentials_without_yuanrong_or_token_in_argv(launcher, script):
    data, process = launcher(script)
    assert data["env"]["env_vars"]["ADX_TOKEN"] == "fixture-not-a-real-token"
    assert data["env"]["env_vars"]["ADX_TLS_VERIFY"] == "1"
    assert data["mode"] == 0o600
    assert "fixture-not-a-real-token" not in " ".join(data["args"]) + process.stdout + process.stderr
    assert "task_config_agent_dx.yaml" in " ".join(data["args"])
    assert not Path(data["runtime_file"]).exists()
    if script == "run_train.sh":
        from ray.runtime_env.runtime_env import _merge_runtime_env

        child_vars = {
            value.split("env_vars.", 1)[1].split("=", 1)[0]: "fixture"
            for value in data["args"]
            if value.startswith("+ray_kwargs.ray_init.runtime_env.env_vars.")
        }
        assert not any(key.startswith(("ADX_", "OPENYUANRONG_")) for key in child_vars)
        assert _merge_runtime_env(data["env"], {"env_vars": child_vars}) is not None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="requires Bash")
@pytest.mark.parametrize(
    ("provider", "override", "expected"),
    [("agent_dx", None, 8400), ("agent_dx", 9300, 9300), ("openyuanrong", None, 3600)],
)
def test_training_session_budget_preserves_provider_defaults_and_explicit_override(
    launcher, provider, override, expected
):
    data, _ = launcher("run_train.sh", provider=provider, session_timeout=override)
    assert (
        f"+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds={expected}"
        in data["args"]
    )
