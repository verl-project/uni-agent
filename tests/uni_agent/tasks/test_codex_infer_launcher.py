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
LAUNCHER = ROOT / "examples/blackbox_recipes/codex_swe_task/run_infer_codex.sh"


@pytest.fixture
def launch(tmp_path):
    ray = tmp_path / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['OUTPUT_DIR'])\n"
        "fixture = json.loads(os.environ['RAY_TEST_RESULT'])\n"
        "(root / 'result.json').write_text(json.dumps(fixture['result']))\n"
        "for i, items in enumerate(fixture['trajectories']):\n"
        "    folder = root / 'logs' / str(i)\n"
        "    folder.mkdir(parents=True)\n"
        "    (folder / 'trajectory.json').write_text(json.dumps(\n"
        "        {'num_trajectories': len(items), 'trajectories': items}))\n"
    )
    ray.chmod(0o700)
    runtime = tmp_path / "runtime-env.yaml"
    runtime.write_text("{}\n")
    runtime.chmod(0o600)

    def run(*, prompts=1, sessions=1, limit=1, n=1, trajectories=None, dry_run=False):
        env = os.environ.copy()
        for key in (
            "LOG_DIR",
            "RESULT_PATH",
            "NNODES",
            "N_GPUS_PER_NODE",
            "TENSOR_PARALLEL_SIZE",
            "CONCURRENCY",
            "GATEWAY_COUNT",
            "LANGUAGE_MODEL_ONLY",
            "DISABLE_THINKING",
        ):
            env.pop(key, None)
        env.update(
            DATA_PATH="/unused/prepared.parquet",
            MODEL_PATH="/unused/model",
            OUTPUT_DIR=str(tmp_path / "output"),
            REPO_ROOT=str(ROOT),
            TASK_CONFIG=str(ROOT / "examples/blackbox_recipes/codex_swe_task/task_config_codex.yaml"),
            RAY_RUNTIME_ENV=str(runtime),
            RAY_API_SERVER_ADDRESS="http://example.invalid:8265",
            PYTHON_BIN=sys.executable,
            RAY_BIN=str(ray),
            LIMIT=str(limit),
            N=str(n),
            DRY_RUN="1" if dry_run else "0",
            RAY_TEST_RESULT=json.dumps(
                {
                    "result": {"num_prompts": prompts, "num_scored_sessions": sessions, "scores": [0.0] * sessions},
                    "trajectories": trajectories if trajectories is not None else [[{"finished": True}]] * sessions,
                }
            ),
        )
        return subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, check=False)

    return run


@pytest.mark.parametrize("prompts,n,limit", [(1, 1, 1), (1, 1, 10), (1, 3, 10), (2, 2, 10)])
def test_launcher_counts_sessions_from_selected_prompts(launch, prompts, n, limit):
    result = launch(prompts=prompts, n=n, limit=limit, sessions=prompts * n)
    assert result.returncode == 0, result.stderr
    assert f"validated {prompts * n} scored session(s)" in result.stdout


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"prompts": 2, "sessions": 3, "n": 2, "limit": 10}, "unexpected session count: expected 4, got 3"),
        ({"prompts": -1}, "invalid selected prompt count"),
        ({"prompts": 2, "sessions": 2, "limit": 1}, "invalid selected prompt count"),
        ({"trajectories": []}, "expected 1 trajectory files"),
        ({"trajectories": [[{"finished": False}]]}, "expected one finished trajectory entry"),
        ({"trajectories": [[{"finished": True}, {"finished": True}]]}, "expected one finished trajectory entry"),
    ],
)
def test_launcher_rejects_incomplete_or_inconsistent_results(launch, kwargs, error):
    result = launch(**kwargs)
    assert result.returncode != 0
    assert error in result.stderr


def test_launcher_passes_recipe_defaults_explicitly(launch):
    result = launch(dry_run=True)
    assert result.returncode == 0, result.stderr
    for option in ("limit", "n", "concurrency", "gateway-count", "n-gpus-per-node", "tensor-parallel-size"):
        assert f"--{option} 1 " in result.stdout
    assert "--language-model-only" in result.stdout
    assert "--disable-thinking" in result.stdout
    assert "--prompt-length" not in result.stdout
