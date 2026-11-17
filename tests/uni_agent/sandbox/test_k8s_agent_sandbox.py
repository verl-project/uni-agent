"""Unit tests for K8sAgentSandbox, LRU warmpool eviction, and R2E-Gym / SWE task config resolution."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace

import uni_agent.sandbox.k8s_agent_sandbox as k8s_sb_mod
from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.k8s_agent_sandbox import (
    _WRITE_FILE_CHUNK_CHARS,
    K8sAgentSandbox,
    PrewarmBatchIterator,
    _K8sStatefulShell,
    close_lookahead_warmer,
    ensure_lookahead_batch_warm,
    extract_sandbox_images,
    finish_lookahead_batch_async,
    normalize_sandbox_image,
    prefetch_lookahead_batch_async,
)
from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.r2e_gym.reward import compute_reward, parse_log_pytest
from uni_agent.tasks.r2e_gym.task import R2EGymTask


@dataclass
class _FakeTask:
    id: str
    image: str
    metadata: dict


class _FakePlan:
    def __init__(self, images: list[str]) -> None:
        self._entries = {img: SimpleNamespace(image=img) for img in images}

    def for_image(self, image: str):
        return self._entries.get(image)


class _FakeFleet:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self.tasks: list[_FakeTask] = []
        self.plan_: _FakePlan | None = None
        self._warmed: dict[str, int] = {}
        self.warm_history: list[tuple[list[str], int | None]] = []
        self.unwarm_history: list[list[str]] = []
        self.released: list[object] = []

    def _install_teardown_hooks(self) -> None:
        return None

    def load_tasks(self, tasks: list[_FakeTask]) -> list[_FakeTask]:
        self.tasks = list(tasks)
        return self.tasks

    def plan(self) -> _FakePlan:
        self.plan_ = _FakePlan([t.image for t in self.tasks])
        return self.plan_

    def add_tasks(self, tasks: list[_FakeTask]) -> list[_FakeTask]:
        self.tasks.extend(tasks)
        if self.plan_ is not None:
            for t in tasks:
                self.plan_._entries[t.image] = SimpleNamespace(image=t.image)
        return tasks

    def warm_images(self, images: list[str], *, replicas_override: int | None = None, wait: bool = True) -> None:
        del wait
        reps = replicas_override or self.config.max_warmpool_size
        for img in images:
            self._warmed[img] = reps
        self.warm_history.append((list(images), replicas_override))

    def warm_image(self, image: str, *, replicas_override: int | None = None, wait: bool = True) -> None:
        self.warm_images([image], replicas_override=replicas_override, wait=wait)

    def set_pool_replicas(self, image: str, replicas: int) -> None:
        self._warmed[image] = replicas

    def unwarm_images(self, images: list[str]) -> None:
        for img in images:
            self._warmed.pop(img, None)
        self.unwarm_history.append(list(images))

    def unwarm_image(self, image: str) -> None:
        self.unwarm_images([image])

    def wait_pools_ready(
        self, images: list[str], timeout: float | None = None, *, replicas: dict | None = None
    ) -> dict[str, bool]:
        del timeout, replicas
        for img in images:
            assert img in self._warmed, f"wait_pools_ready called on unwarmed image {img}"
        return {img: True for img in images}

    def acquire(self, task: _FakeTask):
        assert task.image in self._warmed, f"acquire called on unwarmed image {task.image}"
        return SimpleNamespace(pod_name=f"pod-{task.image.rsplit(':', 1)[-1]}", _cluster=None)

    def release(self, handle: object) -> None:
        self.released.append(handle)


def test_k8s_agent_sandbox_registered():
    cfg = SandboxConfig(provider="k8s_agent_sandbox", image="swerebench/test:latest", runtime_timeout=600.0)
    sb = build_sandbox(cfg)
    assert isinstance(sb, K8sAgentSandbox)
    assert sb.image == "swerebench/test:latest"


def test_normalize_sandbox_image(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/")
    assert (
        normalize_sandbox_image("namanjain12/numpy_final:abc123")
        == "us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/numpy_final:abc123"
    )
    assert (
        normalize_sandbox_image("pillow_final:def456")
        == "us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/pillow_final:def456"
    )
    assert normalize_sandbox_image("otherorg/repo:1") == "otherorg/repo:1"


def test_lru_warmpool_eviction_and_epoch2_rewarm(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "gar.local/tunix")
    monkeypatch.setenv("SANDBOX_RUN_ID", "jialeic-dsw-aa7ab")
    monkeypatch.setenv("SANDBOX_MAX_WARM_IMAGES", "2")
    monkeypatch.setenv("SANDBOX_PINNED_BATCHES", "1")
    monkeypatch.setenv("SANDBOX_MAX_WARMPOOL_SIZE", "8")

    monkeypatch.setattr(k8s_sb_mod, "ClusterConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "FleetConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "SandboxFleet", _FakeFleet)
    monkeypatch.setattr(k8s_sb_mod, "Task", _FakeTask)
    monkeypatch.setattr(k8s_sb_mod, "TemplateSpec", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "_GLOBAL_FLEET", None)
    monkeypatch.setattr(k8s_sb_mod, "_ACTIVE_CLAIMS", {})
    monkeypatch.setattr(k8s_sb_mod, "_RECENT_BATCH_IMAGES", collections.deque(maxlen=1))
    monkeypatch.setattr(k8s_sb_mod, "_IDLE_LRU", collections.OrderedDict())

    async def _scenario():
        # Batch 1 prewarms img1 + img2; run_isolation must default to 'names' for custom SANDBOX_RUN_ID.
        await k8s_sb_mod.async_prewarm_sandbox_images(
            ["namanjain12/repo:1", "namanjain12/repo:2"],
            replicas_override=16,
        )
        fleet = k8s_sb_mod._GLOBAL_FLEET
        assert fleet.config.run_isolation == "names"
        assert set(fleet._warmed) == {"gar.local/tunix/repo:1", "gar.local/tunix/repo:2"}
        assert fleet._warmed["gar.local/tunix/repo:1"] == 8

        # Hold an active claim on img1 so img1 MUST NOT be evicted even when unpinned.
        sb1 = K8sAgentSandbox(image="namanjain12/repo:1")
        await sb1.start()
        assert k8s_sb_mod._ACTIVE_CLAIMS["gar.local/tunix/repo:1"] == 1

        # Batch 2 prewarms img3: pinned={img3}, active={img1}, so idle img2 gets evicted!
        await k8s_sb_mod.async_prewarm_sandbox_images(["namanjain12/repo:3"], replicas_override=16)
        assert set(fleet._warmed) == {"gar.local/tunix/repo:1", "gar.local/tunix/repo:3"}
        assert fleet.unwarm_history == [["gar.local/tunix/repo:2"]]

        # Batch 3 prewarms img4 while sb1 is still active: pinned={img4}, active={img1}, so img3 gets evicted.
        await k8s_sb_mod.async_prewarm_sandbox_images(["namanjain12/repo:4"], replicas_override=16)
        assert set(fleet._warmed) == {"gar.local/tunix/repo:1", "gar.local/tunix/repo:4"}

        # Release sb1 cleanly (idempotent double stop check).
        await sb1.stop()
        await sb1.stop()
        assert "gar.local/tunix/repo:1" not in k8s_sb_mod._ACTIVE_CLAIMS

        # Epoch 2: img2 is still in fleet.plan_ (for_image != None) and was previously evicted;
        # starting a sandbox on img2 MUST re-warm img2 before acquire()!
        assert fleet.plan_.for_image("gar.local/tunix/repo:2") is not None
        assert "gar.local/tunix/repo:2" not in fleet._warmed
        sb2 = K8sAgentSandbox(image="namanjain12/repo:2")
        await sb2.start()
        assert "gar.local/tunix/repo:2" in fleet._warmed
        await sb2.stop()

    asyncio.run(_scenario())


def test_stateful_shell_persists_cwd_and_conda_path():
    class _DummyK8sSandbox(K8sAgentSandbox):
        def __init__(self):
            super().__init__(image="dummy")
            self.last_cmd = ""

        async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
            del timeout, workdir, env
            self.last_cmd = " ".join(argv)
            if "cd /testbed/sub" in self.last_cmd:
                return ExecResult(exit_code=0, stdout="ok\n__UNI_AGENT_PWD__:/testbed/sub\n", stderr="")
            return ExecResult(exit_code=0, stdout="", stderr="")

    sb = _DummyK8sSandbox()
    shell = _K8sStatefulShell(sb, cwd="/testbed", env={"FOO": "bar"})
    res = asyncio.run(shell.run("cd /testbed/sub && echo ok"))
    assert res.exit_code == 0
    assert res.stdout == "ok"
    assert shell.cwd == "/testbed/sub"
    assert "/opt/miniconda3/envs/testbed/bin" in sb.last_cmd


def test_task_config_react_resolves_r2e_gym_and_swe():
    resolver = TaskConfigResolver.from_file("examples/quickstart/training/task_config_react.yaml")
    for task_name in ("r2e_gym", "swe_rebench", "swe_bench"):
        resolved = resolver.resolve(
            {
                "name": task_name,
                "sandbox": {"image": "r2egym/test:latest"},
                "metadata": {"problem_statement": "Fix bug"},
            }
        )
        assert resolved["sandbox"]["provider"] == "k8s_agent_sandbox"
        assert resolved["agent"]["name"] == "react"
        assert resolved["agent"]["max_steps"] == 30
        if task_name == "r2e_gym":
            assert isinstance(get_task(resolved), R2EGymTask)


def test_r2e_gym_compute_reward():
    class _EvalSandbox(K8sAgentSandbox):
        async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
            del argv, timeout, workdir, env
            out = "=== short test summary info ===\nPASSED test_mod.py::TestCls::test_one\n"
            return ExecResult(exit_code=0, stdout=out, stderr="")

    parsed = parse_log_pytest("=== short test summary info ===\nPASSED test_mod.py::TestCls::test_one\n")
    assert parsed == {"TestCls.test_one": "PASSED"}
    res = asyncio.run(
        compute_reward(
            {"instance_id": "repo__1", "expected_output_json": json.dumps({"TestCls.test_one": "PASSED"})},
            _EvalSandbox(image="dummy"),
        )
    )
    assert res["resolved"] is True


def test_extract_sandbox_images_variants(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "gar.local/tunix")

    # 1. Collated batch dict with top-level tools_kwargs (including duplicates)
    batch_dict = {
        "tools_kwargs": [
            {"sandbox": {"image": "namanjain12/numpy_final:1"}},
            {"sandbox": {"image": "namanjain12/pillow_final:2"}},
            {"sandbox": {"image": "namanjain12/numpy_final:1"}},
        ]
    }
    assert extract_sandbox_images(batch_dict) == [
        "gar.local/tunix/numpy_final:1",
        "gar.local/tunix/pillow_final:2",
        "gar.local/tunix/numpy_final:1",
    ]
    assert extract_sandbox_images(batch_dict, deduplicate=True) == [
        "gar.local/tunix/numpy_final:1",
        "gar.local/tunix/pillow_final:2",
    ]

    # 2. Raw parquet rows where tools_kwargs is nested inside extra_info
    raw_rows = [
        {"extra_info": {"tools_kwargs": {"sandbox": {"image": "namanjain12/pandas_final:10"}}}},
        {"extra_info": {"tools_kwargs": {"sandbox": {"image": "namanjain12/tornado_final:20"}}}},
    ]
    assert extract_sandbox_images(raw_rows, deduplicate=True) == [
        "gar.local/tunix/pandas_final:10",
        "gar.local/tunix/tornado_final:20",
    ]


def test_lookahead_iterator_and_just_in_time_eval_pools(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "gar.local/tunix")
    monkeypatch.setenv("SANDBOX_RUN_ID", "jialeic-dsw-lookahead")
    monkeypatch.setenv("SANDBOX_MAX_WARMPOOL_SIZE", "8")

    monkeypatch.setattr(k8s_sb_mod, "ClusterConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "FleetConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "SandboxFleet", _FakeFleet)
    monkeypatch.setattr(k8s_sb_mod, "Task", _FakeTask)
    monkeypatch.setattr(k8s_sb_mod, "TemplateSpec", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "_GLOBAL_FLEET", None)
    monkeypatch.setattr(k8s_sb_mod, "_GLOBAL_LOOKAHEAD_WARMER", None)
    monkeypatch.setattr(k8s_sb_mod, "_ACTIVE_CLAIMS", {})
    monkeypatch.setattr(k8s_sb_mod, "_RECENT_BATCH_IMAGES", collections.deque(maxlen=2))
    monkeypatch.setattr(k8s_sb_mod, "_IDLE_LRU", collections.OrderedDict())

    try:
        # 3 eval images, warmed just in time around validation instead of for the whole run.
        eval_imgs = [f"namanjain12/pillow_final:eval_{i}" for i in range(3)]
        eval_pools = [f"gar.local/tunix/pillow_final:eval_{i}" for i in range(3)]

        # Create 2 training batches in an epoch:
        # Batch 0 uses train_a (duplicated twice in batch) + train_shared
        # Batch 1 uses train_shared + train_b
        batches = [
            {
                "id": 0,
                "tools_kwargs": [
                    {"sandbox": {"image": "namanjain12/numpy_final:train_a"}},
                    {"sandbox": {"image": "namanjain12/numpy_final:train_a"}},
                    {"sandbox": {"image": "namanjain12/numpy_final:train_shared"}},
                ],
            },
            {
                "id": 1,
                "tools_kwargs": [
                    {"sandbox": {"image": "namanjain12/numpy_final:train_shared"}},
                    {"sandbox": {"image": "namanjain12/numpy_final:train_b"}},
                ],
            },
        ]

        iterator = PrewarmBatchIterator(
            batches,
            ensure_warm_fn=ensure_lookahead_batch_warm,
            prefetch_fn=prefetch_lookahead_batch_async,
            finish_fn=finish_lookahead_batch_async,
        )

        # Prime before training starts -> warms Batch 0 ("train_step_1")
        primed_id = iterator.prime()
        assert primed_id == "train_step_1"
        fleet = k8s_sb_mod._GLOBAL_FLEET
        # No eval pool exists before a validation step asks for one.
        assert not any(img in fleet._warmed for img in eval_pools)
        assert fleet._warmed["gar.local/tunix/numpy_final:train_a"] == 8
        assert fleet._warmed["gar.local/tunix/numpy_final:train_shared"] == 8

        # Step 1: pop Batch 0 and asynchronously prefetch Batch 1 ("train_step_2")
        b0 = iterator.next_batch()
        assert b0["id"] == 0
        # Wait for background prefetch of Batch 1 by ensuring "train_step_2" via warmer
        ensure_lookahead_batch_warm("train_step_2", extract_sandbox_images(batches[1]))
        assert fleet._warmed["gar.local/tunix/numpy_final:train_b"] == 8

        # Active sandbox acquisition on Batch 0 image works without re-warming
        sb = K8sAgentSandbox(image="namanjain12/numpy_final:train_a")
        asyncio.run(sb.start())
        asyncio.run(sb.stop())

        # Step 1 rollout finishes -> unwarm Batch 0 ("train_step_1").
        # train_a is unique to Batch 0 so it MUST be unwarmed;
        # train_shared is needed by Batch 1 so it MUST stay warmed.
        finished_id = iterator.finish_oldest_inflight_batch(wait=True)
        assert finished_id == "train_step_1"
        assert "gar.local/tunix/numpy_final:train_a" not in fleet._warmed
        assert fleet._warmed["gar.local/tunix/numpy_final:train_shared"] == 8
        assert fleet._warmed["gar.local/tunix/numpy_final:train_b"] == 8

        # Step 1 ends in validation: prefetch the eval batch in the background, wait for it,
        # validate, then release it. Training pools MUST be untouched throughout.
        prefetch_lookahead_batch_async("eval_step_1", eval_imgs, replicas_override=1)
        ensure_lookahead_batch_warm("eval_step_1", eval_imgs, replicas_override=1, wait=True)
        for img in eval_pools:
            assert fleet._warmed[img] == 1
        assert fleet._warmed["gar.local/tunix/numpy_final:train_shared"] == 8
        assert fleet._warmed["gar.local/tunix/numpy_final:train_b"] == 8
        finish_lookahead_batch_async("eval_step_1", wait=True)
        assert not any(img in fleet._warmed for img in eval_pools)
        assert fleet._warmed["gar.local/tunix/numpy_final:train_shared"] == 8
        assert fleet._warmed["gar.local/tunix/numpy_final:train_b"] == 8

        # Step 2: pop Batch 1 and prefetch Epoch 2 Batch 0 ("train_step_3")
        b1 = iterator.next_batch()
        assert b1["id"] == 1
        ensure_lookahead_batch_warm("train_step_3", extract_sandbox_images(batches[0]))
        assert fleet._warmed["gar.local/tunix/numpy_final:train_a"] == 8

        # Finish Batch 1 ("train_step_2") -> train_b is unwarmed, train_a + train_shared remain;
        # the released eval pools MUST NOT come back.
        finished_id2 = iterator.finish_oldest_inflight_batch(wait=True)
        assert finished_id2 == "train_step_2"
        assert "gar.local/tunix/numpy_final:train_b" not in fleet._warmed
        assert fleet._warmed["gar.local/tunix/numpy_final:train_a"] == 8
        assert fleet._warmed["gar.local/tunix/numpy_final:train_shared"] == 8
        assert not any(img in fleet._warmed for img in eval_pools)
    finally:
        close_lookahead_warmer()


def test_cross_actor_lookahead_mode_registers_plan_without_warming(monkeypatch):
    """When VERL_SANDBOX_LOOKAHEAD=1 on a rollout worker actor, K8sAgentSandbox registers tasks in
    fleet.plan_ (warm=False) so it claims from the trainer's deterministic warm pool without owning pools.
    """

    class _WorkerFakeFleet(_FakeFleet):
        def acquire(self, task: _FakeTask):
            assert self.plan_ is not None and self.plan_.for_image(task.image) is not None
            return SimpleNamespace(pod_name=f"pod-{task.image.rsplit(':', 1)[-1]}", _cluster=None)

    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "gar.local/tunix")
    monkeypatch.setenv("VERL_SANDBOX_LOOKAHEAD", "1")
    monkeypatch.setattr(k8s_sb_mod, "ClusterConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "FleetConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "SandboxFleet", _WorkerFakeFleet)
    monkeypatch.setattr(k8s_sb_mod, "Task", _FakeTask)
    monkeypatch.setattr(k8s_sb_mod, "TemplateSpec", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(k8s_sb_mod, "_GLOBAL_FLEET", None)
    monkeypatch.setattr(k8s_sb_mod, "_GLOBAL_LOOKAHEAD_WARMER", None)
    monkeypatch.setattr(k8s_sb_mod, "_ACTIVE_CLAIMS", {})
    monkeypatch.setattr(k8s_sb_mod, "_RECENT_BATCH_IMAGES", collections.deque(maxlen=2))
    monkeypatch.setattr(k8s_sb_mod, "_IDLE_LRU", collections.OrderedDict())

    async def _worker_scenario():
        await k8s_sb_mod.async_prewarm_sandbox_images(["namanjain12/repo:1"], replicas_override=8)
        assert k8s_sb_mod._GLOBAL_FLEET is None

        sb = K8sAgentSandbox(image="namanjain12/repo:1")
        await sb.start()
        fleet = k8s_sb_mod._GLOBAL_FLEET
        assert fleet is not None
        assert fleet.plan_.for_image("gar.local/tunix/repo:1") is not None
        assert fleet._warmed == {}
        assert fleet.warm_history == []
        await sb.stop()
        assert [h.pod_name for h in fleet.released] == ["pod-1"]

    asyncio.run(_worker_scenario())


def test_write_file_chunks_large_payload_below_argv_limit(tmp_path):
    class _LocalShellSandbox(K8sAgentSandbox):
        max_script = 0

        async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
            type(self).max_script = max(type(self).max_script, len(script))
            proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            return ExecResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

    sandbox = _LocalShellSandbox(image="python:3.12")
    for size in (10, 3 * 1024 * 1024 + 7):
        data = os.urandom(size)
        target = tmp_path / "nested" / f"f{size}.bin"
        asyncio.run(sandbox.write_file(str(target), data))
        assert target.read_bytes() == data
    # Every exec argument stays far below Linux MAX_ARG_STRLEN (128 KiB).
    assert _LocalShellSandbox.max_script < _WRITE_FILE_CHUNK_CHARS + 1024 < 128 * 1024
