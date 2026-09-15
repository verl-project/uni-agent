import sys

import pytest

from examples.inference import parallel_infer_api
from examples.inference.parallel_infer_api import _allocate_worker_concurrency


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("total_concurrency", "num_workers", "expected"),
    [
        (1, 1, [1]),
        (8, 4, [2, 2, 2, 2]),
        (10, 3, [4, 3, 3]),
    ],
)
def test_allocate_worker_concurrency_preserves_global_limit(total_concurrency, num_workers, expected):
    limits = _allocate_worker_concurrency(total_concurrency, num_workers)

    assert limits == expected
    assert sum(limits) == total_concurrency


@pytest.mark.cpu
@pytest.mark.level0
def test_standalone_inference_binds_top_level_source_prompt(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: swe_bench
  sandbox:
    provider: local
  prompt_template:
    - role: user
      content: "Issue: {problem_statement}"
  agent:
    name: react
""".strip()
    )
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]
    sample = {
        "prompt": source_prompt,
        "extra_info": {
            "tools_kwargs": {
                "task": {
                    "name": "swe_bench",
                    "metadata": {"instance_id": "sample-1", "problem_statement": "Metadata problem"},
                }
            }
        },
    }

    class _FakeDataset:
        def to_list(self):
            return [sample]

    class _ResultRef:
        def __init__(self, value):
            self.value = value

    captured_tasks = []

    class _RunSingle:
        def remote(self, task, log_context):
            captured_tasks.append(task)
            return _ResultRef(
                {
                    "instance_id": "sample-1",
                    "reward": 1.0,
                    "resolved": True,
                    "eval_completed": True,
                    "eval_execution_time": 0.1,
                }
            )

    class _Actor:
        run_single = _RunSingle()

    class _RemoteActor:
        def remote(self, max_concurrency):
            return _Actor()

    monkeypatch.setattr(parallel_infer_api, "load_dataset", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(parallel_infer_api.ray, "remote", lambda actor_cls: _RemoteActor())
    monkeypatch.setattr(
        parallel_infer_api.ray,
        "wait",
        lambda refs, num_returns=1: ([refs[0]], refs[1:]),
    )
    monkeypatch.setattr(parallel_infer_api.ray, "get", lambda ref: ref.value)
    monkeypatch.setattr(parallel_infer_api, "NUM_WORKERS", 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallel_infer_api.py",
            "--data-path",
            "unused.parquet",
            "--task-config",
            str(config_path),
            "--base-url",
            "http://model/v1",
            "--model",
            "policy",
            "--concurrency",
            "1",
            "--log-dir",
            "",
        ],
    )

    parallel_infer_api.main()

    assert len(captured_tasks) == 1
    assert captured_tasks[0]["prompt"] == source_prompt
