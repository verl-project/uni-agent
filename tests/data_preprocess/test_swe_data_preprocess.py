import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PREPROCESS_DIR = _REPO_ROOT / "examples" / "data_preprocess"
_SWE_BENCH_VERIFIED_SCRIPT = _DATA_PREPROCESS_DIR / "swe_bench_verified.py"


def _load_preprocess_script(monkeypatch, script_name: str, deployment: str):
    script_path = _DATA_PREPROCESS_DIR / script_name
    module_name = f"_{script_path.stem}_{deployment}_under_test"
    monkeypatch.setenv("DEPLOYMENT", deployment)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_swe_bench_verified_local_image_name_uses_public_swe_bench_image(monkeypatch):
    module = _load_preprocess_script(monkeypatch, "swe_bench_verified.py", "local")
    image = module.get_image_name("swe-bench-verified", "Django__Django-10914")

    assert image == "swebench/sweb.eval.x86_64.django_1776_django-10914:latest"


def test_swe_rebench_local_image_name_uses_public_swe_rebench_image(monkeypatch):
    module = _load_preprocess_script(monkeypatch, "swe_rebench.py", "local")
    image = module.get_image_name("swe-rebench", "Astropy__Astropy-12907")

    assert image == "swerebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_r2e_gym_local_image_name_reuses_public_volcengine_image(monkeypatch):
    module = _load_preprocess_script(monkeypatch, "r2e_gym_subset_filtered.py", "local")
    image = module.get_image_name("r2e-gym-subset", "repo__abcdef1234")

    assert image == "enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/abcdef1234:latest"


def test_save_parquet_creates_save_dir(monkeypatch, tmp_path):
    module = _load_preprocess_script(monkeypatch, "swe_bench_verified.py", "local")

    class FakeDataset:
        def __init__(self):
            self.saved_path = None

        def to_parquet(self, path):
            self.saved_path = Path(path)

    dataset = FakeDataset()

    module._save_parquet(dataset, str(tmp_path / "missing" / "nested"), "out.parquet")

    assert dataset.saved_path == tmp_path / "missing" / "nested" / "out.parquet"
    assert dataset.saved_path.parent.is_dir()


def test_swe_bench_verified_local_preprocess_builds_local_image(monkeypatch):
    module = _load_preprocess_script(monkeypatch, "swe_bench_verified.py", "local")

    class FakeDataset:
        column_names = ["instance_id", "base_commit", "problem_statement"]

        def __init__(self):
            self.mapped_sample = None
            self.remove_columns = None

        def map(self, transform, remove_columns):
            example = {
                "instance_id": "Django__Django-10914",
                "base_commit": "abc123",
                "problem_statement": "Fix the broken behavior.",
            }
            self.mapped_sample = transform(example)
            self.remove_columns = remove_columns
            return self

    fake_dataset = FakeDataset()

    def fake_load_dataset(data_source, split):
        assert data_source == "princeton-nlp/SWE-bench_Verified"
        assert split == "test"
        return fake_dataset

    monkeypatch.setattr(module, "load_dataset", fake_load_dataset)

    dataset = module.build_swe_bench_verified()

    assert dataset is fake_dataset
    assert dataset.remove_columns == fake_dataset.column_names
    sample = dataset.mapped_sample
    env_kwargs = sample["extra_info"]["tools_kwargs"]["env"]
    assert env_kwargs["image"] == "swebench/sweb.eval.x86_64.django_1776_django-10914:latest"
    assert "git checkout abc123" in env_kwargs["post_setup_cmd"]
    assert sample["extra_info"]["tools_kwargs"]["reward"]["name"] == "swe_bench"


def test_swe_bench_verified_script_help_runs_by_file_path():
    env = {**os.environ, "DEPLOYMENT": "local"}

    result = subprocess.run(
        [sys.executable, str(_SWE_BENCH_VERIFIED_SCRIPT), "--help"],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--local-save-dir" in result.stdout
