"""Unit tests for GKEAgentSandbox: network isolation, runtime_timeout, image_map, pod_annotations, and lookahead."""

from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import uni_agent.sandbox.gke_agent_sandbox as gke_sb_mod
from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.gke_agent_sandbox import (
    _WRITE_FILE_CHUNK_CHARS,
    GKEAgentSandbox,
    PrewarmBatchIterator,
    SandboxRuntimeTimeoutError,
    _GKEStatefulShell,
    apply_image_map,
    close_lookahead_warmer,
    ensure_lookahead_batch_warm,
    extract_sandbox_images,
    finish_lookahead_batch_async,
    load_image_map_from_task_config,
    normalize_sandbox_image,
    parse_image_map,
    prefetch_lookahead_batch_async,
)
from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.r2e_gym.reward import compute_reward, parse_log_pytest
from uni_agent.tasks.r2e_gym.task import R2EGymTask

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


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


class _FakeCustomApi:
    def __init__(self) -> None:
        self.patches: list[dict] = []

    def patch_namespaced_custom_object(self, **kwargs):
        self.patches.append(kwargs)
        return kwargs


class _FakeFleet:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self.tasks: list[_FakeTask] = []
        self.plan_: _FakePlan | None = None
        self._warmed: dict[str, int] = {}
        self.warm_history: list[tuple[list[str], int | None]] = []
        self.unwarm_history: list[list[str]] = []
        self.released: list[object] = []
        self.acquire_annotations: list[dict | None] = []
        self.custom_api = _FakeCustomApi()

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

    def acquire(self, task: _FakeTask, *, pod_annotations: dict | None = None):
        assert task.image in self._warmed, f"acquire called on unwarmed image {task.image}"
        self.acquire_annotations.append(pod_annotations)
        ns = self.config.clusters[0].namespace if self.config.clusters else "default"
        cluster = SimpleNamespace(
            namespace=ns,
            custom_api=self.custom_api,
            resources=SimpleNamespace(custom_api=self.custom_api),
        )
        return SimpleNamespace(
            pod_name=f"pod-{task.image.rsplit(':', 1)[-1]}",
            claim_name=f"claim-{task.image.rsplit(':', 1)[-1]}",
            _cluster=cluster,
        )

    def release(self, handle: object) -> None:
        self.released.append(handle)


def _reset_fleet_globals(monkeypatch, *, fleet_cls=_FakeFleet) -> None:
    monkeypatch.setattr(gke_sb_mod, "ClusterConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(gke_sb_mod, "FleetConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(gke_sb_mod, "SandboxFleet", fleet_cls)
    monkeypatch.setattr(gke_sb_mod, "Task", _FakeTask)
    monkeypatch.setattr(gke_sb_mod, "TemplateSpec", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(gke_sb_mod, "_GLOBAL_FLEET", None)
    monkeypatch.setattr(gke_sb_mod, "_GLOBAL_LOOKAHEAD_WARMER", None)
    monkeypatch.setattr(gke_sb_mod, "_FLEET_POD_ANNOTATIONS", {})
    monkeypatch.setattr(gke_sb_mod, "_ACTIVE_CLAIMS", {})
    monkeypatch.setattr(gke_sb_mod, "_RECENT_BATCH_IMAGES", collections.deque(maxlen=2))
    monkeypatch.setattr(gke_sb_mod, "_IDLE_LRU", collections.OrderedDict())


def test_gke_agent_sandbox_registered_with_image_map():
    cfg = SandboxConfig(
        provider="gke_agent_sandbox",
        image="upstream-org/test_repo:v1",
        runtime_timeout=600.0,
        image_map=[{"from": "upstream-org/**", "to": "mirror.example.com/team/**"}],
    )
    sb = build_sandbox(cfg)
    assert isinstance(sb, GKEAgentSandbox)
    assert sb.image == "mirror.example.com/team/test_repo:v1"
    assert sb.runtime_timeout == 600.0
    assert sb.network_policy_management == "Managed"
    assert sb.network_policy == {"ingress": [], "egress": []}
    assert sb.pod_annotations == {}


def test_image_map_parsing_and_rewriting(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_IMAGE_MAP", raising=False)
    # Without any image_map configured, images pass through untouched (no hardcoded org rewrites).
    assert normalize_sandbox_image("some-org/numpy_final:abc123") == "some-org/numpy_final:abc123"

    # Shorthand from=to via env var
    monkeypatch.setenv("SANDBOX_IMAGE_MAP", "some-org/**=region-docker.pkg.dev/my-proj/my-repo/**")
    assert (
        normalize_sandbox_image("some-org/numpy_final:abc123")
        == "region-docker.pkg.dev/my-proj/my-repo/numpy_final:abc123"
    )
    assert normalize_sandbox_image("other-org/repo:1") == "other-org/repo:1"

    # JSON list format via env var
    monkeypatch.setenv(
        "SANDBOX_IMAGE_MAP",
        json.dumps([{"from": "org-a/**:latest", "to": "mirror.example.com/a/**:v2"}]),
    )
    assert apply_image_map("org-a/pkg") == "mirror.example.com/a/pkg:v2"

    # Loading from Task Config YAML + env var
    cfg_file = tmp_path / "task_config.yaml"
    cfg_file.write_text(
        "- name: r2e_gym\n"
        "  sandbox:\n"
        "    provider: gke_agent_sandbox\n"
        "    image_map:\n"
        "      - from: 'yaml-org/**'\n"
        "        to: 'mirror.example.com/yaml/**'\n",
        encoding="utf-8",
    )
    rules = load_image_map_from_task_config(str(cfg_file))
    assert len(rules) == 2
    assert apply_image_map("yaml-org/img:42", image_map=rules) == "mirror.example.com/yaml/img:42"
    assert apply_image_map("org-a/img", image_map=rules) == "mirror.example.com/a/img:v2"

    # Malformed rule raises ValueError
    with pytest.raises(ValueError, match="SANDBOX_IMAGE_MAP"):
        parse_image_map("not-a-mapping")


def test_default_network_policy_is_managed_deny_all_and_no_default_annotations(monkeypatch):
    _reset_fleet_globals(monkeypatch)
    monkeypatch.delenv("SANDBOX_POD_ANNOTATIONS", raising=False)
    monkeypatch.delenv("SANDBOX_NETWORK_POLICY_MANAGEMENT", raising=False)
    monkeypatch.setenv("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
    monkeypatch.setenv("NODE_SELECTOR_VAL", "sandbox-cpu-pool")

    async def _scenario():
        sb = GKEAgentSandbox(image="mirror.example.com/repo:1", runtime_timeout=120.0)
        await sb.start()
        fleet = gke_sb_mod._GLOBAL_FLEET
        assert fleet.config.template.network_policy_management == "Managed"
        assert fleet.config.template.extra_template_spec == {"networkPolicy": {"ingress": [], "egress": []}}
        assert fleet.config.template.automount_service_account_token is False
        assert fleet.config.template.share_process_namespace is False
        assert fleet.config.template.extra_pod_spec == {
            "tolerations": [
                {
                    "key": "cloud.google.com/gke-nodepool",
                    "operator": "Equal",
                    "value": "sandbox-cpu-pool",
                    "effect": "NoSchedule",
                }
            ]
        }
        # No hardcoded annotations are stamped by default.
        assert gke_sb_mod._FLEET_POD_ANNOTATIONS == {}
        assert fleet.acquire_annotations == [None]
        await sb.stop()

    asyncio.run(_scenario())


def test_unmanaged_network_policy_fails_closed_without_namespace_egress_policy(monkeypatch):
    _reset_fleet_globals(monkeypatch)
    monkeypatch.setattr(gke_sb_mod, "_load_kube_config_once", lambda: None)

    class _FakeNetApi:
        def __init__(self, items):
            self._items = items

        def list_namespaced_network_policy(self, namespace):
            del namespace
            return SimpleNamespace(items=self._items)

    # 1. Empty policy list -> refused
    monkeypatch.setattr(gke_sb_mod, "k8s_client", SimpleNamespace(NetworkingV1Api=lambda: _FakeNetApi([])))
    sb = GKEAgentSandbox(image="mirror.example.com/repo:1", network_policy_management="Unmanaged")
    with pytest.raises(RuntimeError, match="no egress-restricting NetworkPolicy"):
        asyncio.run(sb.start())

    # 2. Ingress-only or allow-all egress policy -> still refused
    allow_all_egress = SimpleNamespace(
        spec=SimpleNamespace(
            pod_selector=SimpleNamespace(match_labels={"app": "agent-sandbox-rl"}, match_expressions=None),
            policy_types=["Egress"],
            egress=[SimpleNamespace(to=None, ports=None)],
        )
    )
    ingress_only = SimpleNamespace(
        spec=SimpleNamespace(
            pod_selector=SimpleNamespace(match_labels={}, match_expressions=None),
            policy_types=["Ingress"],
            egress=None,
        )
    )
    monkeypatch.setattr(
        gke_sb_mod,
        "k8s_client",
        SimpleNamespace(NetworkingV1Api=lambda: _FakeNetApi([allow_all_egress, ingress_only])),
    )
    with pytest.raises(RuntimeError, match="no egress-restricting NetworkPolicy"):
        asyncio.run(sb.start())

    # 3. Namespace-wide deny-all Egress policy selecting app=agent-sandbox-rl -> allowed
    deny_all_egress = SimpleNamespace(
        spec=SimpleNamespace(
            pod_selector=SimpleNamespace(match_labels={"app": "agent-sandbox-rl"}, match_expressions=None),
            policy_types=["Ingress", "Egress"],
            egress=[],
        )
    )
    monkeypatch.setattr(
        gke_sb_mod,
        "k8s_client",
        SimpleNamespace(NetworkingV1Api=lambda: _FakeNetApi([deny_all_egress])),
    )

    async def _run_allowed():
        await sb.start()
        assert gke_sb_mod._GLOBAL_FLEET.config.template.network_policy_management == "Unmanaged"
        assert not hasattr(gke_sb_mod._GLOBAL_FLEET.config.template, "extra_template_spec")
        await sb.stop()

    asyncio.run(_run_allowed())


def test_opt_in_pod_annotations_applied_to_template_and_acquire(monkeypatch):
    _reset_fleet_globals(monkeypatch)
    monkeypatch.setenv("SANDBOX_POD_ANNOTATIONS", json.dumps({"custom.example.com/owner": "team-a"}))

    class _FakeResources:
        def _template_manifest(self, image: str, template_name: str, template):
            del image, template_name, template
            return {
                "spec": {
                    "podTemplate": {
                        "metadata": {"labels": {"app": "agent-sandbox-rl"}},
                        "spec": {"containers": [{"name": "default-name"}]},
                    }
                }
            }

    monkeypatch.setattr(gke_sb_mod, "asrl_resources", SimpleNamespace(Resources=_FakeResources))
    monkeypatch.setattr(gke_sb_mod, "_TEMPLATES_PATCHED", False)

    async def _scenario():
        sb = GKEAgentSandbox(
            image="mirror.example.com/repo:1",
            pod_annotations={"custom.example.com/role": "worker"},
        )
        await sb.start()
        # Merged env + explicit kwargs annotations
        expected = {"custom.example.com/owner": "team-a", "custom.example.com/role": "worker"}
        assert gke_sb_mod._FLEET_POD_ANNOTATIONS == expected
        assert gke_sb_mod._GLOBAL_FLEET.acquire_annotations == [expected]

        # Template manifest hook stamps only user-provided annotations (no hardcoded keys).
        manifest = _FakeResources()._template_manifest("img", "tpl", None)
        assert manifest["spec"]["podTemplate"]["metadata"]["annotations"] == expected
        assert manifest["spec"]["podTemplate"]["spec"]["containers"][0]["name"] == "agent-runtime"
        await sb.stop()

    asyncio.run(_scenario())

    # Invalid JSON in SANDBOX_POD_ANNOTATIONS raises ValueError
    monkeypatch.setenv("SANDBOX_POD_ANNOTATIONS", "{not-json")
    with pytest.raises(ValueError, match="SANDBOX_POD_ANNOTATIONS"):
        GKEAgentSandbox(image="mirror.example.com/repo:1")


def test_runtime_timeout_validation_expiry_and_claim_lifecycle_patch(monkeypatch):
    for bad in (0, -5, math.nan, math.inf):
        with pytest.raises(ValueError, match="runtime_timeout"):
            GKEAgentSandbox(image="mirror.example.com/repo:1", runtime_timeout=bad)

    _reset_fleet_globals(monkeypatch)

    async def _scenario():
        sb = GKEAgentSandbox(image="mirror.example.com/repo:1", runtime_timeout=0.15)
        await sb.start()
        fleet = gke_sb_mod._GLOBAL_FLEET
        # Best-effort server-side SandboxClaim lifecycle patch was sent on start.
        assert len(fleet.custom_api.patches) == 1
        patch = fleet.custom_api.patches[0]
        assert patch["plural"] == "sandboxclaims"
        assert patch["name"] == "claim-1"
        lifecycle = patch["body"]["spec"]["lifecycle"]
        assert lifecycle["shutdownPolicy"] == "Delete"
        assert lifecycle["shutdownTime"].endswith("Z")
        assert await sb.is_alive() is True

        # Wait past runtime_timeout -> background expiry task tears down sandbox.
        await asyncio.sleep(0.25)
        assert await sb.is_alive() is False
        assert len(fleet.released) == 1

        # Subsequent exec() must raise SandboxRuntimeTimeoutError (not swallowed to exit_code=-1).
        with pytest.raises(SandboxRuntimeTimeoutError):
            await sb.exec(["echo", "hi"])

    asyncio.run(_scenario())


def test_runtime_timeout_clamps_in_flight_exec_and_preserves_command_timeout(monkeypatch):
    _reset_fleet_globals(monkeypatch)

    class _SlowExecSandbox(GKEAgentSandbox):
        def __init__(self, *, sleep_s: float, **kwargs):
            super().__init__(**kwargs)
            self._sleep_s = sleep_s
            self.seen_timeouts: list[float] = []

        def _exec_in_pod_sync(self, argv, timeout, workdir, env, *, stream_grace=10.0):
            del argv, workdir, env, stream_grace
            self.seen_timeouts.append(timeout)
            time.sleep(self._sleep_s)
            raise TimeoutError("command timed out")

    async def _scenario():
        # Case 1: Normal command timeout while sandbox still has plenty of runtime_timeout -> exit_code=-1.
        sb_ok = _SlowExecSandbox(sleep_s=0.02, image="mirror.example.com/repo:1", runtime_timeout=60.0)
        await sb_ok.start()
        res = await sb_ok.exec(["sleep", "5"], timeout=1.0)
        assert res.exit_code == -1
        assert await sb_ok.is_alive() is True
        await sb_ok.stop()

        # Case 2: Command outlives remaining runtime_timeout -> clamped and raises SandboxRuntimeTimeoutError.
        sb_exp = _SlowExecSandbox(sleep_s=0.35, image="mirror.example.com/repo:2", runtime_timeout=0.15)
        await sb_exp.start()
        with pytest.raises(SandboxRuntimeTimeoutError):
            await sb_exp.exec(["sleep", "30"], timeout=30.0)
        assert sb_exp.seen_timeouts and sb_exp.seen_timeouts[0] <= 0.16
        assert await sb_exp.is_alive() is False

    asyncio.run(_scenario())


def test_lru_warmpool_eviction_and_epoch2_rewarm(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_MAP", "upstream/**=mirror.example.com/repo/**")
    monkeypatch.setenv("SANDBOX_RUN_ID", "test-run-1")
    monkeypatch.setenv("SANDBOX_MAX_WARM_IMAGES", "2")
    monkeypatch.setenv("SANDBOX_PINNED_BATCHES", "1")
    monkeypatch.setenv("SANDBOX_MAX_WARMPOOL_SIZE", "8")

    _reset_fleet_globals(monkeypatch)
    monkeypatch.setattr(gke_sb_mod, "_RECENT_BATCH_IMAGES", collections.deque(maxlen=1))

    async def _scenario():
        # Batch 1 prewarms img1 + img2; run_isolation must default to 'names' for custom SANDBOX_RUN_ID.
        await gke_sb_mod.async_prewarm_sandbox_images(
            ["upstream/app:1", "upstream/app:2"],
            replicas_override=16,
        )
        fleet = gke_sb_mod._GLOBAL_FLEET
        assert fleet.config.run_isolation == "names"
        assert set(fleet._warmed) == {"mirror.example.com/repo/app:1", "mirror.example.com/repo/app:2"}
        assert fleet._warmed["mirror.example.com/repo/app:1"] == 8

        # Hold an active claim on img1 so img1 MUST NOT be evicted even when unpinned.
        sb1 = GKEAgentSandbox(image="upstream/app:1")
        await sb1.start()
        assert gke_sb_mod._ACTIVE_CLAIMS["mirror.example.com/repo/app:1"] == 1

        # Batch 2 prewarms img3: pinned={img3}, active={img1}, so idle img2 gets evicted!
        await gke_sb_mod.async_prewarm_sandbox_images(["upstream/app:3"], replicas_override=16)
        assert set(fleet._warmed) == {"mirror.example.com/repo/app:1", "mirror.example.com/repo/app:3"}
        assert fleet.unwarm_history == [["mirror.example.com/repo/app:2"]]

        # Batch 3 prewarms img4 while sb1 is still active: pinned={img4}, active={img1}, so img3 gets evicted.
        await gke_sb_mod.async_prewarm_sandbox_images(["upstream/app:4"], replicas_override=16)
        assert set(fleet._warmed) == {"mirror.example.com/repo/app:1", "mirror.example.com/repo/app:4"}

        # Release sb1 cleanly (idempotent double stop check).
        await sb1.stop()
        await sb1.stop()
        assert "mirror.example.com/repo/app:1" not in gke_sb_mod._ACTIVE_CLAIMS

        # Epoch 2: img2 is still in fleet.plan_ (for_image != None) and was previously evicted;
        # starting a sandbox on img2 MUST re-warm img2 before acquire()!
        assert fleet.plan_.for_image("mirror.example.com/repo/app:2") is not None
        assert "mirror.example.com/repo/app:2" not in fleet._warmed
        sb2 = GKEAgentSandbox(image="upstream/app:2")
        await sb2.start()
        assert "mirror.example.com/repo/app:2" in fleet._warmed
        await sb2.stop()

    asyncio.run(_scenario())


def test_stateful_shell_persists_cwd_and_conda_path():
    class _DummyGKESandbox(GKEAgentSandbox):
        def __init__(self):
            super().__init__(image="dummy")
            self.last_cmd = ""

        async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
            del timeout, workdir, env
            self.last_cmd = " ".join(argv)
            if "cd /testbed/sub" in self.last_cmd:
                return ExecResult(exit_code=0, stdout="ok\n__UNI_AGENT_PWD__:/testbed/sub\n", stderr="")
            return ExecResult(exit_code=0, stdout="", stderr="")

    sb = _DummyGKESandbox()
    shell = _GKEStatefulShell(sb, cwd="/testbed", env={"FOO": "bar"})
    res = asyncio.run(shell.run("cd /testbed/sub && echo ok"))
    assert res.exit_code == 0
    assert res.stdout == "ok"
    assert shell.cwd == "/testbed/sub"
    assert "/opt/miniconda3/envs/testbed/bin" in sb.last_cmd


def test_task_config_react_gke_resolves_r2e_gym_and_swe_without_breaking_default_config():
    gke_resolver = TaskConfigResolver.from_file("examples/quickstart/training/task_config_react_gke.yaml")
    for task_name in ("r2e_gym", "swe_rebench", "swe_bench"):
        resolved = gke_resolver.resolve(
            {
                "name": task_name,
                "sandbox": {"image": "r2egym/test:latest"},
                "metadata": {"problem_statement": "Fix bug"},
            }
        )
        assert resolved["sandbox"]["provider"] == "gke_agent_sandbox"
        assert resolved["agent"]["name"] == "react"
        assert resolved["agent"]["max_steps"] == 30
        if task_name == "r2e_gym":
            assert isinstance(get_task(resolved), R2EGymTask)

    # Upstream default task_config_react.yaml stays intact (not overwritten with gke_agent_sandbox).
    default_resolver = TaskConfigResolver.from_file("examples/quickstart/training/task_config_react.yaml")
    default_resolved = default_resolver.resolve(
        {
            "name": "swe_bench",
            "sandbox": {"image": "swebench/test:latest"},
            "metadata": {"problem_statement": "Fix bug"},
        }
    )
    assert default_resolved["sandbox"]["provider"] != "gke_agent_sandbox"


def test_r2e_gym_compute_reward():
    class _EvalSandbox(GKEAgentSandbox):
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
    monkeypatch.setenv("SANDBOX_IMAGE_MAP", "upstream/**=mirror.example.com/repo/**")

    # 1. Collated batch dict with top-level tools_kwargs (including duplicates)
    batch_dict = {
        "tools_kwargs": [
            {"sandbox": {"image": "upstream/numpy_final:1"}},
            {"sandbox": {"image": "upstream/pillow_final:2"}},
            {"sandbox": {"image": "upstream/numpy_final:1"}},
        ]
    }
    assert extract_sandbox_images(batch_dict) == [
        "mirror.example.com/repo/numpy_final:1",
        "mirror.example.com/repo/pillow_final:2",
        "mirror.example.com/repo/numpy_final:1",
    ]
    assert extract_sandbox_images(batch_dict, deduplicate=True) == [
        "mirror.example.com/repo/numpy_final:1",
        "mirror.example.com/repo/pillow_final:2",
    ]

    # 2. Raw parquet rows where tools_kwargs is nested inside extra_info
    raw_rows = [
        {"extra_info": {"tools_kwargs": {"sandbox": {"image": "upstream/pandas_final:10"}}}},
        {"extra_info": {"tools_kwargs": {"sandbox": {"image": "upstream/tornado_final:20"}}}},
    ]
    assert extract_sandbox_images(raw_rows, deduplicate=True) == [
        "mirror.example.com/repo/pandas_final:10",
        "mirror.example.com/repo/tornado_final:20",
    ]


def test_lookahead_iterator_and_just_in_time_eval_pools(monkeypatch):
    monkeypatch.setenv("SANDBOX_IMAGE_MAP", "upstream/**=mirror.example.com/repo/**")
    monkeypatch.setenv("SANDBOX_RUN_ID", "test-lookahead")
    monkeypatch.setenv("SANDBOX_MAX_WARMPOOL_SIZE", "8")

    _reset_fleet_globals(monkeypatch)

    try:
        eval_imgs = [f"upstream/pillow_final:eval_{i}" for i in range(3)]
        eval_pools = [f"mirror.example.com/repo/pillow_final:eval_{i}" for i in range(3)]

        batches = [
            {
                "id": 0,
                "tools_kwargs": [
                    {"sandbox": {"image": "upstream/numpy_final:train_a"}},
                    {"sandbox": {"image": "upstream/numpy_final:train_a"}},
                    {"sandbox": {"image": "upstream/numpy_final:train_shared"}},
                ],
            },
            {
                "id": 1,
                "tools_kwargs": [
                    {"sandbox": {"image": "upstream/numpy_final:train_shared"}},
                    {"sandbox": {"image": "upstream/numpy_final:train_b"}},
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
        fleet = gke_sb_mod._GLOBAL_FLEET
        assert not any(img in fleet._warmed for img in eval_pools)
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_a"] == 8
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_shared"] == 8

        # Step 1: pop Batch 0 and asynchronously prefetch Batch 1 ("train_step_2")
        b0 = iterator.next_batch()
        assert b0["id"] == 0
        ensure_lookahead_batch_warm("train_step_2", extract_sandbox_images(batches[1]))
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_b"] == 8

        # Active sandbox acquisition on Batch 0 image works without re-warming
        sb = GKEAgentSandbox(image="upstream/numpy_final:train_a")
        asyncio.run(sb.start())
        asyncio.run(sb.stop())

        # Step 1 rollout finishes -> unwarm Batch 0 ("train_step_1").
        finished_id = iterator.finish_oldest_inflight_batch(wait=True)
        assert finished_id == "train_step_1"
        assert "mirror.example.com/repo/numpy_final:train_a" not in fleet._warmed
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_shared"] == 8
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_b"] == 8

        # Step 1 ends in validation: prefetch eval batch, wait, validate, then release.
        prefetch_lookahead_batch_async("eval_step_1", eval_imgs, replicas_override=1)
        ensure_lookahead_batch_warm("eval_step_1", eval_imgs, replicas_override=1, wait=True)
        for img in eval_pools:
            assert fleet._warmed[img] == 1
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_shared"] == 8
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_b"] == 8
        finish_lookahead_batch_async("eval_step_1", wait=True)
        assert not any(img in fleet._warmed for img in eval_pools)
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_shared"] == 8
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_b"] == 8

        # Step 2: pop Batch 1 and prefetch Epoch 2 Batch 0 ("train_step_3")
        b1 = iterator.next_batch()
        assert b1["id"] == 1
        ensure_lookahead_batch_warm("train_step_3", extract_sandbox_images(batches[0]))
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_a"] == 8

        # Finish Batch 1 ("train_step_2") -> train_b is unwarmed, train_a + train_shared remain.
        finished_id2 = iterator.finish_oldest_inflight_batch(wait=True)
        assert finished_id2 == "train_step_2"
        assert "mirror.example.com/repo/numpy_final:train_b" not in fleet._warmed
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_a"] == 8
        assert fleet._warmed["mirror.example.com/repo/numpy_final:train_shared"] == 8
        assert not any(img in fleet._warmed for img in eval_pools)
    finally:
        close_lookahead_warmer()


def test_cross_actor_lookahead_mode_registers_plan_without_warming(monkeypatch):
    """When VERL_SANDBOX_LOOKAHEAD=1 on a rollout worker actor, GKEAgentSandbox registers tasks in
    fleet.plan_ (warm=False) so it claims from the trainer's deterministic warm pool without owning pools.
    """

    class _WorkerFakeFleet(_FakeFleet):
        def acquire(self, task: _FakeTask, *, pod_annotations: dict | None = None):
            del pod_annotations
            assert self.plan_ is not None and self.plan_.for_image(task.image) is not None
            return SimpleNamespace(pod_name=f"pod-{task.image.rsplit(':', 1)[-1]}", _cluster=None)

    monkeypatch.setenv("SANDBOX_IMAGE_MAP", "upstream/**=mirror.example.com/repo/**")
    monkeypatch.setenv("VERL_SANDBOX_LOOKAHEAD", "1")
    _reset_fleet_globals(monkeypatch, fleet_cls=_WorkerFakeFleet)

    async def _worker_scenario():
        await gke_sb_mod.async_prewarm_sandbox_images(["upstream/app:1"], replicas_override=8)
        assert gke_sb_mod._GLOBAL_FLEET is None

        sb = GKEAgentSandbox(image="upstream/app:1")
        await sb.start()
        fleet = gke_sb_mod._GLOBAL_FLEET
        assert fleet is not None
        assert fleet.plan_.for_image("mirror.example.com/repo/app:1") is not None
        assert fleet._warmed == {}
        assert fleet.warm_history == []
        await sb.stop()
        assert [h.pod_name for h in fleet.released] == ["pod-1"]

    asyncio.run(_worker_scenario())


def test_write_file_chunks_large_payload_below_argv_limit(tmp_path):
    class _LocalShellSandbox(GKEAgentSandbox):
        max_script = 0

        async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
            del timeout, workdir, env
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


def test_exec_in_pod_drains_buffered_channels_on_close(monkeypatch):
    class _FakeStreamResp:
        def __init__(self):
            self._open = True
            self._channels = {1: "tail-stdout", 2: "tail-stderr"}
            self.returncode = 0
            self.closed = False

        def is_open(self):
            return self._open

        def update(self, timeout=1.0):
            del timeout
            self._open = False

        def peek_stdout(self):
            return False

        def peek_stderr(self):
            return False

        def close(self):
            self.closed = True

    fake_resp = _FakeStreamResp()
    monkeypatch.setattr(gke_sb_mod, "k8s_stream", lambda *args, **kwargs: fake_resp)

    sb = GKEAgentSandbox(image="python:3.12")
    sb._pod_name = "test-pod"
    sb._handle = SimpleNamespace(
        _cluster=SimpleNamespace(exec_core_api=lambda: SimpleNamespace(connect_get_namespaced_pod_exec=object()))
    )
    res = sb._exec_in_pod_sync(["echo", "hi"], timeout=10.0, workdir=None, env=None)
    assert res.exit_code == 0
    assert res.stdout == "tail-stdout"
    assert res.stderr == "tail-stderr"
    assert fake_resp.closed is True
