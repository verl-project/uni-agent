"""Check GPU launch commands without submitting jobs or installing GPU wheels."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("recipe", ["train_qwen3_moe.sh", "train_qwen3p5_dense.sh"])
@pytest.mark.parametrize("backend,sandbox", [("vllm", "vefaas"), ("sglang", "modal")])
def test_training_submission_preserves_locked_remote_environment(tmp_path, recipe, backend, sandbox):
    # Intercept the submission before it reaches UV/Ray. Paths with spaces and
    # trailing Hydra arguments must survive both shell command boundaries.
    output = tmp_path / "submission.json"
    binary = tmp_path / "uv"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_SUBMISSION'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    binary.chmod(0o755)
    runtime = str(tmp_path / "private runtime.yaml")
    env = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        CAPTURE_SUBMISSION=str(output),
        DATA_DIR=str(tmp_path / "model data"),
        RUNTIME_DIR=str(tmp_path / "training output"),
        RUNTIME_ENV=runtime,
        ROLLOUT_NAME=backend,
        UV_SANDBOX_EXTRA=sandbox,
    )
    subprocess.run(
        ["bash", str(ROOT / "examples/quickstart/training" / recipe), "trainer.total_epochs=1"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = json.loads(output.read_text())
    split = args.index("ray")
    prefix = args[:split]
    assert prefix[:2] == ["run", "--locked"]
    assert "--all-packages" in prefix
    assert prefix[prefix.index("--python") + 1] == "3.12"
    extras = [prefix[i + 1] for i, item in enumerate(prefix[:-1]) if item == "--extra"]
    assert set(extras) == {backend, "megatron", "swebench", sandbox}
    assert args[args.index("--runtime-env") + 1] == runtime
    assert args[args.index("--working-dir") + 1] == "."
    remote = args[args.index("--") + 1 :]
    assert remote[:3] == ["env", "RAY_OVERRIDE_JOB_RUNTIME_ENV=1", "uv"]
    assert remote[3 : 3 + len(prefix)] == prefix
    assert remote[3 + len(prefix) : 6 + len(prefix)] == ["python", "-m", "verl.trainer.main_ppo"]
    assert args[-1] == "trainer.total_epochs=1"
    assert f"data.train_files={env['DATA_DIR']}/data/uni_agent/swe_rebench_filtered_1150.parquet" in args


def test_training_runtime_uses_project_dependencies():
    config = yaml.safe_load((ROOT / "examples/quickstart/training/runtime_env_uv.yaml").read_text())
    assert config["working_dir"] == "."
    assert not {"pip", "conda", "uv", "py_executable"}.intersection(config)
    assert "/.venv/" in config["excludes"]
