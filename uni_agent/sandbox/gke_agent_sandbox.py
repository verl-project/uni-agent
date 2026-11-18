"""GKE / Kubernetes Agent Sandbox provider backed by ``agent_sandbox_rl`` ``SandboxFleet``."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import os
import re
import shlex
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from .base import ExecResult, ImageMap, Sandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.stream import stream as k8s_stream
except ImportError:
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    k8s_stream = None  # type: ignore[assignment]

try:
    import agent_sandbox_rl.resources as asrl_resources
    from agent_sandbox_rl import ClusterConfig, FleetConfig, SandboxFleet, Task
    from agent_sandbox_rl.config import TemplateSpec
    from agent_sandbox_rl.lookahead import LookaheadWarmer
except ImportError:
    ClusterConfig = None  # type: ignore[assignment]
    FleetConfig = None  # type: ignore[assignment]
    SandboxFleet = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]
    TemplateSpec = None  # type: ignore[assignment]
    LookaheadWarmer = None  # type: ignore[assignment]
    asrl_resources = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

RUNTIME_CONTAINER_NAME = "agent-runtime"
_PWD_SENTINEL = "__UNI_AGENT_PWD__"

_DEFAULT_NETWORK_POLICY_MANAGEMENT = "Managed"
_VALID_NETWORK_POLICY_MODES = ("Managed", "Unmanaged")
_DENY_ALL_NETWORK_POLICY: dict[str, list[Any]] = {"ingress": [], "egress": []}
_AGENT_SANDBOX_RL_INSTALL_HINT = (
    "GKEAgentSandbox requires 'agent_sandbox_rl', which is not installed. "
    "Install the 'gke' extra from a uni-agent checkout (pip install -e '.[gke]') and 'agent-sandbox-rl' "
    "(see docs/source/concepts/sandbox.md)."
)

_AGENT_SANDBOX_EXT_GROUP = "extensions.agents.x-k8s.io"
_AGENT_SANDBOX_EXT_VERSION = "v1beta1"
_SANDBOX_CLAIMS_PLURAL = "sandboxclaims"

_FLEET_LOCK = threading.Lock()
_GLOBAL_FLEET: Any = None
_GLOBAL_LOOKAHEAD_WARMER: Any = None
_LOOKAHEAD_FUTURES: dict[int | str, Future[Any]] = {}
_TEMPLATES_PATCHED = False
_FLEET_POD_ANNOTATIONS: dict[str, str] = {}
_ACTIVE_CLAIMS: dict[str, int] = {}
_RECENT_BATCH_IMAGES: collections.deque[set[str]] = collections.deque(maxlen=4)
_IDLE_LRU: collections.OrderedDict[str, None] = collections.OrderedDict()

# Separate executors so cold sandbox claim waits never starve active pod exec calls.
# Initialized lazily so importing this module or pickling Ray actors never touches a ThreadPoolExecutor.
_EXECUTOR_LOCK = threading.Lock()
_LIFECYCLE_EXECUTOR: ThreadPoolExecutor | None = None
_EXEC_EXECUTOR: ThreadPoolExecutor | None = None


class SandboxRuntimeTimeoutError(RuntimeError):
    """Raised when a ``GKEAgentSandbox`` exceeds its overall ``runtime_timeout``."""


def get_lifecycle_executor() -> ThreadPoolExecutor:
    global _LIFECYCLE_EXECUTOR
    if _LIFECYCLE_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _LIFECYCLE_EXECUTOR is None:
                _LIFECYCLE_EXECUTOR = ThreadPoolExecutor(max_workers=128, thread_name_prefix="gke-sb-lifecycle")
    return _LIFECYCLE_EXECUTOR


def get_exec_executor() -> ThreadPoolExecutor:
    global _EXEC_EXECUTOR
    if _EXEC_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXEC_EXECUTOR is None:
                _EXEC_EXECUTOR = ThreadPoolExecutor(max_workers=256, thread_name_prefix="gke-sb-exec")
    return _EXEC_EXECUTOR


def parse_image_map(raw: Any) -> list[ImageMap]:
    """Parse ``ImageMap`` rules from a list/dict, JSON string, or comma-separated ``from=to`` string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in SANDBOX_IMAGE_MAP: {exc}") from exc
            return parse_image_map(parsed)
        rules: list[ImageMap] = []
        for entry in text.split(","):
            item = entry.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"Invalid SANDBOX_IMAGE_MAP entry {item!r}; expected 'from=to' or JSON list of {{from, to}} rules"
                )
            src, dst = item.split("=", 1)
            rules.append(ImageMap(from_=src.strip(), to=dst.strip()))
        return rules
    if isinstance(raw, ImageMap):
        return [raw]
    if isinstance(raw, Mapping):
        return [ImageMap.model_validate(dict(raw))]
    if isinstance(raw, Sequence):
        result: list[ImageMap] = []
        for elem in raw:
            if isinstance(elem, ImageMap):
                result.append(elem)
            elif isinstance(elem, Mapping):
                result.append(ImageMap.model_validate(dict(elem)))
            else:
                raise ValueError(f"Unsupported image_map rule entry: {elem!r}")
        return result
    raise ValueError(f"Unsupported image_map value: {raw!r}")


def load_image_map_from_task_config(task_config_path: str | None) -> list[ImageMap]:
    """Collect ``sandbox.image_map`` rules from a Task Config YAML file, followed by ``SANDBOX_IMAGE_MAP`` env rules."""
    rules: list[ImageMap] = []
    seen: set[tuple[str, str]] = set()
    if task_config_path:
        path = Path(task_config_path).expanduser()
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                sandbox_cfg = entry.get("sandbox")
                if isinstance(sandbox_cfg, Mapping) and sandbox_cfg.get("image_map"):
                    for rule in parse_image_map(sandbox_cfg["image_map"]):
                        key = (rule.from_, rule.to)
                        if key not in seen:
                            seen.add(key)
                            rules.append(rule)
    for rule in parse_image_map(os.getenv("SANDBOX_IMAGE_MAP", "")):
        key = (rule.from_, rule.to)
        if key not in seen:
            seen.add(key)
            rules.append(rule)
    return rules


def apply_image_map(
    image: str,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> str:
    """Apply ``ImageMap`` rewrite rules (explicit or from ``SANDBOX_IMAGE_MAP``) to ``image``. First match wins."""
    if not image:
        return image
    rules = parse_image_map(os.getenv("SANDBOX_IMAGE_MAP", "")) if image_map is None else parse_image_map(image_map)
    for rule in rules:
        mapped = rule.try_map(image)
        if mapped is not None:
            return mapped
    return image


def _read_in_cluster_namespace() -> str | None:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", encoding="utf-8") as f:
            ns = f.read().strip()
            if ns:
                return ns
    except OSError:
        pass
    return None


def _resolve_default_namespace(namespace: str | None = None) -> str:
    if namespace and namespace.strip():
        return namespace.strip()
    for env_key in ("SANDBOX_NAMESPACE", "K8S_NAMESPACE", "POD_NAMESPACE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    return _read_in_cluster_namespace() or "default"


def _resolve_default_node_selector(node_selector: Mapping[str, str] | None = None) -> dict[str, str] | None:
    if node_selector is not None:
        return {str(k): str(v) for k, v in node_selector.items()}
    k, v = os.getenv("NODE_SELECTOR_KEY"), os.getenv("NODE_SELECTOR_VAL")
    return {k: v} if (k and v) else None


def _resolve_default_pod_annotations(pod_annotations: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse optional pod annotations from ``SANDBOX_POD_ANNOTATIONS`` (JSON object) and explicit kwargs."""
    merged: dict[str, str] = {}
    raw_env = os.getenv("SANDBOX_POD_ANNOTATIONS", "").strip()
    if raw_env:
        try:
            parsed = json.loads(raw_env)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in SANDBOX_POD_ANNOTATIONS: {exc}") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            raise ValueError("SANDBOX_POD_ANNOTATIONS must be a JSON object mapping string keys to string values")
        merged.update(parsed)
    if pod_annotations is not None:
        if not isinstance(pod_annotations, Mapping):
            raise ValueError(f"pod_annotations must be a mapping, got {type(pod_annotations).__name__}")
        for k, v in pod_annotations.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("pod_annotations keys and values must be strings")
            merged[k] = v
    return merged


def _getenv_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _require_agent_sandbox_rl() -> None:
    if ClusterConfig is None or FleetConfig is None or SandboxFleet is None or Task is None:
        raise ImportError(_AGENT_SANDBOX_RL_INSTALL_HINT)


def _resolve_network_policy_management(mode: str | None = None) -> str:
    if isinstance(mode, str) and mode.strip():
        resolved = mode.strip()
    else:
        env_val = (os.getenv("SANDBOX_NETWORK_POLICY_MANAGEMENT") or "").strip()
        resolved = env_val or _DEFAULT_NETWORK_POLICY_MANAGEMENT
    if resolved not in _VALID_NETWORK_POLICY_MODES:
        raise ValueError(f"network_policy_management must be one of {_VALID_NETWORK_POLICY_MODES}, got {resolved!r}")
    return resolved


def _resolve_network_policy(network_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if network_policy is None:
        return {
            "ingress": list(_DENY_ALL_NETWORK_POLICY["ingress"]),
            "egress": list(_DENY_ALL_NETWORK_POLICY["egress"]),
        }
    if not isinstance(network_policy, Mapping):
        raise ValueError(f"network_policy must be a mapping, got {type(network_policy).__name__}")
    return dict(network_policy)


def _load_kube_config_once() -> None:
    if k8s_config is None:
        raise ImportError("The 'kubernetes' package is required for GKEAgentSandbox.")
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()


def _pod_selector_covers(selector: Any, labels: Mapping[str, str]) -> bool:
    """Return True if a Kubernetes label selector matches a pod carrying ``labels``."""
    if selector is None:
        return True
    if isinstance(selector, Mapping):
        match_labels = selector.get("matchLabels") or selector.get("match_labels") or {}
        match_exprs = selector.get("matchExpressions") or selector.get("match_expressions") or []
    else:
        match_labels = getattr(selector, "match_labels", None) or {}
        match_exprs = getattr(selector, "match_expressions", None) or []

    for key, expected in match_labels.items():
        if labels.get(key) != expected:
            return False

    for expr in match_exprs:
        if isinstance(expr, Mapping):
            key = expr.get("key")
            op = expr.get("operator")
            values = set(expr.get("values") or ())
        else:
            key = getattr(expr, "key", None)
            op = getattr(expr, "operator", None)
            values = set(getattr(expr, "values", None) or ())
        present = key in labels
        val = labels.get(key)
        if op == "In" and (not present or val not in values):
            return False
        if op == "NotIn" and present and val in values:
            return False
        if op == "Exists" and not present:
            return False
        if op == "DoesNotExist" and present:
            return False
    return True


def _is_allow_all_egress_rule(rule: Any) -> bool:
    if isinstance(rule, Mapping):
        to = rule.get("to")
        ports = rule.get("ports")
    else:
        to = getattr(rule, "to", None)
        ports = getattr(rule, "ports", None)
    return not to and not ports


def _policy_restricts_egress(policy: Any) -> bool:
    spec = policy.get("spec") if isinstance(policy, Mapping) else getattr(policy, "spec", None)
    if spec is None:
        return False
    if isinstance(spec, Mapping):
        policy_types = spec.get("policyTypes") or spec.get("policy_types") or []
        egress = spec.get("egress")
    else:
        policy_types = getattr(spec, "policy_types", None) or []
        egress = getattr(spec, "egress", None)

    applies_to_egress = ("Egress" in policy_types) or (egress is not None)
    if not applies_to_egress:
        return False
    if egress and any(_is_allow_all_egress_rule(rule) for rule in egress):
        return False
    return True


def _verify_unmanaged_network_policy(namespace: str, pod_labels: Mapping[str, str]) -> None:
    """Fail closed when ``networkPolicyManagement='Unmanaged'`` has no namespace-wide egress NetworkPolicy."""
    if k8s_client is None:
        raise RuntimeError(
            "network_policy_management='Unmanaged' requires the 'kubernetes' package to verify that a "
            f"namespace-wide NetworkPolicy restricts egress for pods {dict(pod_labels)} in namespace {namespace!r}. "
            "Use the default 'Managed' mode instead."
        )
    try:
        _load_kube_config_once()
        policies = k8s_client.NetworkingV1Api().list_namespaced_network_policy(namespace).items or []
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to run with network_policy_management='Unmanaged' in namespace {namespace!r}: "
            f"could not verify a namespace-wide NetworkPolicy ({exc}). Use the default 'Managed' mode, "
            "or grant list access on networking.k8s.io/networkpolicies."
        ) from exc

    for policy in policies:
        spec = policy.get("spec") if isinstance(policy, Mapping) else getattr(policy, "spec", None)
        selector = (
            (spec.get("podSelector") if isinstance(spec, Mapping) else getattr(spec, "pod_selector", None))
            if spec is not None
            else None
        )
        if _pod_selector_covers(selector, pod_labels) and _policy_restricts_egress(policy):
            return

    wanted = ",".join(f"{k}={v}" for k, v in sorted(pod_labels.items()))
    raise RuntimeError(
        f"Refusing to run with network_policy_management='Unmanaged': no egress-restricting NetworkPolicy in "
        f"namespace {namespace!r} selects sandbox pods ({wanted}). Apply a namespace-wide deny-all NetworkPolicy "
        "or use the default network_policy_management='Managed' mode."
    )


def patch_agent_sandbox_rl_templates() -> None:
    """Hook ``agent_sandbox_rl.Resources._template_manifest`` to attach opt-in ``pod_annotations`` when configured."""
    global _TEMPLATES_PATCHED
    if asrl_resources is None or _TEMPLATES_PATCHED:
        return
    if not hasattr(asrl_resources.Resources, "_orig_template_manifest") and hasattr(
        asrl_resources.Resources, "_template_manifest"
    ):
        asrl_resources.Resources._orig_template_manifest = asrl_resources.Resources._template_manifest

        def _patched_template_manifest(self: Any, image: str, template_name: str, template: Any) -> dict[str, Any]:
            manifest = self._orig_template_manifest(image, template_name, template)
            spec = manifest["spec"]
            pod_template = spec["podTemplate"]
            if _FLEET_POD_ANNOTATIONS:
                pod_meta = pod_template.setdefault("metadata", {})
                pod_annotations = pod_meta.setdefault("annotations", {})
                for key, value in _FLEET_POD_ANNOTATIONS.items():
                    pod_annotations.setdefault(key, value)
            pod_spec = pod_template["spec"]
            pod_spec.setdefault("automountServiceAccountToken", False)
            pod_spec.setdefault("shareProcessNamespace", False)
            containers = pod_spec.get("containers") or []
            if containers:
                containers[0]["name"] = RUNTIME_CONTAINER_NAME
            return manifest

        asrl_resources.Resources._template_manifest = _patched_template_manifest
        _TEMPLATES_PATCHED = True


def _get_or_init_fleet_locked(
    *,
    namespace: str,
    node_selector: dict[str, str] | None,
    pod_annotations: Mapping[str, str] | None = None,
    network_policy_management: str | None = None,
    network_policy: Mapping[str, Any] | None = None,
) -> Any:
    global _GLOBAL_FLEET, _RECENT_BATCH_IMAGES, _FLEET_POD_ANNOTATIONS
    if _GLOBAL_FLEET is not None:
        return _GLOBAL_FLEET
    _require_agent_sandbox_rl()

    resolved_annotations = _resolve_default_pod_annotations(pod_annotations)
    resolved_np_mgmt = _resolve_network_policy_management(network_policy_management)
    resolved_np = _resolve_network_policy(network_policy)

    if resolved_np_mgmt == "Unmanaged":
        _verify_unmanaged_network_policy(namespace, {"app": "agent-sandbox-rl"})

    _FLEET_POD_ANNOTATIONS = dict(resolved_annotations)
    patch_agent_sandbox_rl_templates()

    adopt_existing = os.getenv("SANDBOX_ADOPT_EXISTING", "0").strip().lower() in ("1", "true", "yes")
    run_id = os.getenv("SANDBOX_RUN_ID", "uni-agent-shared").strip() or None
    default_isolation = "none" if (adopt_existing or run_id in (None, "uni-agent-shared")) else "names"
    run_isolation = os.getenv("SANDBOX_RUN_ISOLATION", default_isolation).strip() or default_isolation
    max_concurrent = max(1, _getenv_int("SANDBOX_MAX_CONCURRENT", 512))
    max_warmpool_size = max(1, _getenv_int("SANDBOX_MAX_WARMPOOL_SIZE", 8))
    colocate_replicas = os.getenv("SANDBOX_COLOCATE_REPLICAS", "1").strip().lower() in ("1", "true", "yes")
    pinned_batches = max(1, _getenv_int("SANDBOX_PINNED_BATCHES", 4))
    _RECENT_BATCH_IMAGES = collections.deque(_RECENT_BATCH_IMAGES, maxlen=pinned_batches)

    fleet_kwargs: dict[str, Any] = {
        "clusters": [
            ClusterConfig(
                name="default",
                namespace=namespace,
                node_selector=node_selector,
                in_cluster=os.getenv("KUBERNETES_SERVICE_HOST") is not None,
            )
        ],
        "max_concurrent": max_concurrent,
        "window_size": 16,
        "max_warmpool_size": max_warmpool_size,
        "warm_per_task": True,
        "adopt_existing": adopt_existing,
        "run_isolation": run_isolation,
        "run_id": run_id,
        "install_teardown_hooks": True,
    }
    template_kwargs: dict[str, Any] = {
        "colocate_replicas": colocate_replicas,
        "network_policy_management": resolved_np_mgmt,
        "automount_service_account_token": False,
        "share_process_namespace": False,
    }
    if node_selector:
        template_kwargs["extra_pod_spec"] = {
            "tolerations": [
                {"key": k, "operator": "Equal", "value": str(v), "effect": "NoSchedule"}
                for k, v in node_selector.items()
            ]
        }
    if resolved_np_mgmt == "Managed":
        template_kwargs["extra_template_spec"] = {"networkPolicy": resolved_np}
    fleet_kwargs["template"] = TemplateSpec(**template_kwargs)

    cfg = FleetConfig(**fleet_kwargs)
    _GLOBAL_FLEET = SandboxFleet(cfg)
    _GLOBAL_FLEET._install_teardown_hooks()
    logger.info(
        "Initialized GKE SandboxFleet (namespace=%s, run_id=%s, run_isolation=%s, max_warmpool_size=%d, "
        "colocate=%s, network_policy_management=%s)",
        namespace,
        run_id,
        run_isolation,
        max_warmpool_size,
        colocate_replicas,
        resolved_np_mgmt,
    )
    return _GLOBAL_FLEET


def _register_and_warm_locked(
    fleet: Any,
    tasks: list[Any],
    *,
    warm: bool = True,
    replicas_override: int | None = None,
) -> None:
    unique_tasks: list[Any] = []
    seen_images: set[str] = set()
    for t in tasks:
        img = getattr(t, "image", None)
        if img and img not in seen_images:
            seen_images.add(img)
            unique_tasks.append(t)
    if not unique_tasks:
        return

    unplanned = [t for t in unique_tasks if fleet.plan_ is None or fleet.plan_.for_image(t.image) is None]
    if unplanned:
        if fleet.plan_ is None:
            fleet.load_tasks(unplanned)
            fleet.plan()
        else:
            fleet.add_tasks(unplanned)

    for t in unique_tasks:
        _IDLE_LRU[t.image] = None
        _IDLE_LRU.move_to_end(t.image)

    if warm:
        warmed_map = getattr(fleet, "_warmed", None)
        # Re-warm images that were evicted across epochs even when still present in fleet.plan_.
        to_warm = (
            [t.image for t in unique_tasks if t.image not in warmed_map]
            if isinstance(warmed_map, dict)
            else [t.image for t in unplanned]
        )
        if to_warm:
            default_reps = getattr(getattr(fleet, "config", None), "max_warmpool_size", None) or 8
            reps = max(1, min(int(replicas_override), default_reps)) if replicas_override is not None else default_reps
            fleet.warm_images(to_warm, replicas_override=reps, wait=False)


def _is_lookahead_mode_enabled() -> bool:
    """Return True when warm-pool lifecycle is managed by a ``LookaheadWarmer`` (local or trainer actor)."""
    return _GLOBAL_LOOKAHEAD_WARMER is not None or os.getenv("VERL_SANDBOX_LOOKAHEAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _evict_excess_warmpools_locked(fleet: Any) -> list[str]:
    """Unwarm oldest zero-claim, unpinned warm pools once active pool count exceeds SANDBOX_MAX_WARM_IMAGES."""
    if _is_lookahead_mode_enabled():
        return []
    if getattr(getattr(fleet, "config", None), "adopt_existing", False):
        return []
    max_warm_images = _getenv_int("SANDBOX_MAX_WARM_IMAGES", 128)
    if max_warm_images <= 0:
        return []
    warmed_map = getattr(fleet, "_warmed", None)
    if not isinstance(warmed_map, dict) or len(warmed_map) <= max_warm_images:
        return []

    for img in warmed_map:
        if img not in _IDLE_LRU:
            _IDLE_LRU[img] = None
    pinned: set[str] = set().union(*_RECENT_BATCH_IMAGES) if _RECENT_BATCH_IMAGES else set()
    excess = len(warmed_map) - max_warm_images
    to_evict: list[str] = []
    for img in list(_IDLE_LRU.keys()):
        if len(to_evict) >= excess:
            break
        if img not in warmed_map:
            _IDLE_LRU.pop(img, None)
            continue
        if _ACTIVE_CLAIMS.get(img, 0) > 0 or img in pinned:
            continue
        to_evict.append(img)

    if to_evict:
        try:
            fleet.unwarm_images(to_evict)
            for img in to_evict:
                if img not in warmed_map:
                    _IDLE_LRU.pop(img, None)
            logger.info(
                "Evicted %d idle sandbox warm pool(s) (active_warm=%d, limit=%d)",
                len(to_evict),
                len(warmed_map),
                max_warm_images,
            )
        except Exception:
            logger.warning("Failed to evict idle sandbox warm pools: %s", to_evict[:3], exc_info=True)
    return to_evict


def _extract_image_from_mapping(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, dict):
        return None
    tools_kwargs = item.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict):
        extra_info = item.get("extra_info")
        if isinstance(extra_info, dict):
            tools_kwargs = extra_info.get("tools_kwargs")
    if isinstance(tools_kwargs, dict):
        task_dict = tools_kwargs.get("task") if isinstance(tools_kwargs.get("task"), dict) else tools_kwargs
    elif isinstance(item.get("task"), dict):
        task_dict = item["task"]
    else:
        task_dict = item
    sandbox_cfg = task_dict.get("sandbox")
    if isinstance(sandbox_cfg, dict):
        img = sandbox_cfg.get("image")
        if isinstance(img, str) and img.strip():
            return img.strip()
    meta_cfg = task_dict.get("metadata")
    if isinstance(meta_cfg, dict):
        img = meta_cfg.get("docker_image")
        if isinstance(img, str) and img.strip():
            return img.strip()
    img = task_dict.get("docker_image") or task_dict.get("image")
    if isinstance(img, str) and img.strip():
        return img.strip()
    return None


def _maybe_dedup_images(images: list[str], *, deduplicate: bool) -> list[str]:
    if not deduplicate:
        return images
    return list(dict.fromkeys(images))


def extract_sandbox_images(
    batch_or_items: Any,
    *,
    deduplicate: bool = False,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> list[str]:
    """Extract normalized sandbox container images from a batch (TensorDict/dict), dataset, or item list."""
    if batch_or_items is None:
        return []
    dataframe = getattr(batch_or_items, "dataframe", None)
    if dataframe is not None:
        batch_or_items = dataframe
    if hasattr(batch_or_items, "to_dict") and hasattr(batch_or_items, "columns"):
        # pandas.DataFrame or HuggingFace Dataset with columns
        try:
            records = batch_or_items.to_dict(orient="records")
        except TypeError:
            records = list(batch_or_items)
        images = [
            apply_image_map(img, image_map=image_map) for item in records if (img := _extract_image_from_mapping(item))
        ]
        return _maybe_dedup_images(images, deduplicate=deduplicate)
    if isinstance(batch_or_items, dict) or hasattr(batch_or_items, "keys"):
        for key in ("tools_kwargs", "extra_info"):
            try:
                col = batch_or_items[key]
            except Exception:
                col = None
            if col is not None:
                if isinstance(col, dict):
                    items = [col]
                elif hasattr(col, "tolist"):
                    items = col.tolist()
                else:
                    items = list(col)
                extracted: list[str] = []
                for elem in items:
                    wrapped = {key: elem} if isinstance(elem, dict) and key not in elem else elem
                    img = _extract_image_from_mapping(wrapped)
                    if img:
                        extracted.append(apply_image_map(img, image_map=image_map))
                if extracted:
                    return _maybe_dedup_images(extracted, deduplicate=deduplicate)
        if isinstance(batch_or_items, dict):
            single = _extract_image_from_mapping(batch_or_items)
            return [apply_image_map(single, image_map=image_map)] if single else []
    if isinstance(batch_or_items, str):
        return [apply_image_map(batch_or_items, image_map=image_map)]
    images = []
    try:
        iterator = iter(batch_or_items)
    except TypeError:
        return images
    for item in iterator:
        img = _extract_image_from_mapping(item)
        if img:
            images.append(apply_image_map(img, image_map=image_map))
    return _maybe_dedup_images(images, deduplicate=deduplicate)


def _get_or_init_lookahead_warmer_locked(fleet: Any) -> Any:
    global _GLOBAL_LOOKAHEAD_WARMER
    if _GLOBAL_LOOKAHEAD_WARMER is not None and getattr(_GLOBAL_LOOKAHEAD_WARMER, "fleet", None) is fleet:
        return _GLOBAL_LOOKAHEAD_WARMER
    cap = getattr(getattr(fleet, "config", None), "max_warmpool_size", None) or 8
    _GLOBAL_LOOKAHEAD_WARMER = LookaheadWarmer(
        fleet,
        cap=cap,
        keep_finished=0,
        max_workers=32,
        fail_fast=False,
        register_images=True,
    )
    return _GLOBAL_LOOKAHEAD_WARMER


def _build_image_counts(
    images: list[str],
    *,
    replicas_override: int | None = None,
    max_replicas: int = 8,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> tuple[list[str], dict[str, int]]:
    normalized = [apply_image_map(img, image_map=image_map) for img in images if img]
    unique_images = list(dict.fromkeys(normalized))
    per_prompt = max(1, min(int(replicas_override), max_replicas)) if replicas_override is not None else max_replicas
    counts: dict[str, int] = collections.OrderedDict()
    for img in normalized:
        counts[img] = min(max_replicas, counts.get(img, 0) + per_prompt)
    return unique_images, counts


def ensure_lookahead_batch_warm(
    batch_id: int | str,
    images: list[str],
    *,
    replicas_override: int | None = None,
    wait: bool = True,
    wait_timeout: float | None = None,
    namespace: str | None = None,
    node_selector: dict[str, str] | None = None,
    pod_annotations: Mapping[str, str] | None = None,
    network_policy_management: str | None = None,
    network_policy: Mapping[str, Any] | None = None,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> None:
    """Ensure ``batch_id`` is registered and warmed (awaiting any prior background prefetch for ``batch_id``)."""
    _require_agent_sandbox_rl()
    resolved_ns = _resolve_default_namespace(namespace)
    resolved_sel = _resolve_default_node_selector(node_selector)
    with _FLEET_LOCK:
        fleet = _get_or_init_fleet_locked(
            namespace=resolved_ns,
            node_selector=resolved_sel,
            pod_annotations=pod_annotations,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
        )
        max_reps = getattr(getattr(fleet, "config", None), "max_warmpool_size", None) or 8
        unique_images, counts = _build_image_counts(
            images,
            replicas_override=replicas_override,
            max_replicas=max_reps,
            image_map=image_map,
        )
        if not unique_images:
            return
        task_objs = [Task(id=img, image=img, metadata={"ds": {"docker_image": img}}) for img in unique_images]
        _register_and_warm_locked(fleet, task_objs, warm=False)
        warmer = _get_or_init_lookahead_warmer_locked(fleet)
        fut = _LOOKAHEAD_FUTURES.pop(batch_id, None)
    if fut is not None:
        fut.result()
        if wait:
            warmer.wait_ready(counts.keys(), timeout=wait_timeout, replicas=counts)
    elif batch_id in getattr(warmer, "_live", {}):
        if wait:
            warmer.wait_ready(counts.keys(), timeout=wait_timeout, replicas=counts)
    else:
        warmer.add(batch_id, counts, wait=wait, wait_timeout=wait_timeout)


def prefetch_lookahead_batch_async(
    batch_id: int | str,
    images: list[str],
    *,
    replicas_override: int | None = None,
    namespace: str | None = None,
    node_selector: dict[str, str] | None = None,
    pod_annotations: Mapping[str, str] | None = None,
    network_policy_management: str | None = None,
    network_policy: Mapping[str, Any] | None = None,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> None:
    """Start warming ``batch_id``'s sandbox pools asynchronously in the background via ``LookaheadWarmer``."""
    _require_agent_sandbox_rl()
    resolved_ns = _resolve_default_namespace(namespace)
    resolved_sel = _resolve_default_node_selector(node_selector)
    with _FLEET_LOCK:
        fleet = _get_or_init_fleet_locked(
            namespace=resolved_ns,
            node_selector=resolved_sel,
            pod_annotations=pod_annotations,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
        )
        max_reps = getattr(getattr(fleet, "config", None), "max_warmpool_size", None) or 8
        unique_images, counts = _build_image_counts(
            images,
            replicas_override=replicas_override,
            max_replicas=max_reps,
            image_map=image_map,
        )
        if not unique_images:
            return
        task_objs = [Task(id=img, image=img, metadata={"ds": {"docker_image": img}}) for img in unique_images]
        _register_and_warm_locked(fleet, task_objs, warm=False)
        warmer = _get_or_init_lookahead_warmer_locked(fleet)
    fut = warmer.add_async(batch_id, counts, wait=False)
    with _FLEET_LOCK:
        _LOOKAHEAD_FUTURES[batch_id] = fut


def finish_lookahead_batch_async(batch_id: int | str, *, wait: bool = False) -> None:
    """Retire ``batch_id`` from ``LookaheadWarmer`` and tear down its unneeded warm pools."""
    with _FLEET_LOCK:
        warmer = _GLOBAL_LOOKAHEAD_WARMER
        _LOOKAHEAD_FUTURES.pop(batch_id, None)
    if warmer is None:
        return
    if wait:
        warmer.finish(batch_id)
        return
    with warmer._lock:
        counts = warmer._live.pop(batch_id, None)
        if counts is None:
            return
        if warmer.keep_finished:
            warmer._finished.append(counts)
        if warmer._executor is None:
            warmer._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookahead-warm")
        ex = warmer._executor
    ex.submit(warmer._reconcile)


def close_lookahead_warmer() -> None:
    """Drain background lookahead futures and close the global ``LookaheadWarmer``."""
    global _GLOBAL_LOOKAHEAD_WARMER
    with _FLEET_LOCK:
        warmer, _GLOBAL_LOOKAHEAD_WARMER = _GLOBAL_LOOKAHEAD_WARMER, None
        _LOOKAHEAD_FUTURES.clear()
    if warmer is not None:
        warmer.close()


class PrewarmBatchIterator:
    """2-slot sliding lookahead iterator.

    Warms Step 1 synchronously on ``prime()``, then on each ``next_batch()`` returns the
    already-warmed batch for Step ``k`` while prefetching and warming Step ``k + 1`` in
    the background. Calling ``finish_oldest_inflight_batch()`` when Step ``k`` completes
    tears down Step ``k``'s pools while keeping Step ``k + 1`` (and any other live batch) warm.
    """

    def __init__(
        self,
        fetch_batch_fn: Callable[[], Any] | Any,
        *,
        extract_images_fn: Callable[[Any], list[str]] = extract_sandbox_images,
        ensure_warm_fn: Callable[..., Any] = ensure_lookahead_batch_warm,
        prefetch_async_fn: Callable[..., Any] | None = None,
        prefetch_fn: Callable[..., Any] | None = None,
        finish_batch_fn: Callable[..., Any] | None = None,
        finish_fn: Callable[..., Any] | None = None,
        replicas_override: int | None = None,
    ) -> None:
        if callable(fetch_batch_fn):
            self._fetch_batch_fn = fetch_batch_fn
        else:
            source = fetch_batch_fn
            it_state = [iter(source)]

            def _next_from_iterable() -> Any:
                try:
                    return next(it_state[0])
                except StopIteration:
                    it_state[0] = iter(source)
                    return next(it_state[0])

            self._fetch_batch_fn = _next_from_iterable
        self._extract_images_fn = extract_images_fn
        self._ensure_warm_fn = ensure_warm_fn
        self._prefetch_async_fn = prefetch_fn or prefetch_async_fn or prefetch_lookahead_batch_async
        self._finish_batch_fn = finish_fn or finish_batch_fn or finish_lookahead_batch_async
        self._replicas_override = replicas_override
        self._next_batch_id = 1
        self._lookahead: tuple[int | str, Any, list[str]] | None = None
        self._inflight_batch_ids: collections.deque[int | str] = collections.deque()

    @property
    def inflight_batch_ids(self) -> list[int | str]:
        return list(self._inflight_batch_ids)

    @property
    def lookahead_batch_id(self) -> int | str | None:
        return self._lookahead[0] if self._lookahead is not None else None

    def prime(self) -> int | str:
        """Pull the first batch into the lookahead slot and warm its sandboxes synchronously."""
        if self._lookahead is not None:
            return self._lookahead[0]
        batch_id = f"train_step_{self._next_batch_id}"
        self._next_batch_id += 1
        batch = self._fetch_batch_fn()
        images = self._extract_images_fn(batch)
        if images:
            self._ensure_warm_fn(batch_id, images, replicas_override=self._replicas_override, wait=True)
        self._lookahead = (batch_id, batch, images)
        return batch_id

    def next_batch(self) -> Any:
        """Return the warmed current batch and start warming the next batch in the background."""
        if self._lookahead is None:
            self.prime()
        assert self._lookahead is not None
        batch_id, batch, images = self._lookahead
        self._lookahead = None
        if images:
            self._ensure_warm_fn(batch_id, images, replicas_override=self._replicas_override, wait=True)

        next_id = f"train_step_{self._next_batch_id}"
        self._next_batch_id += 1
        next_batch_obj = self._fetch_batch_fn()
        next_images = self._extract_images_fn(next_batch_obj)
        self._lookahead = (next_id, next_batch_obj, next_images)
        if next_images:
            self._prefetch_async_fn(next_id, next_images, replicas_override=self._replicas_override)

        self._inflight_batch_ids.append(batch_id)
        return batch

    def finish_oldest_inflight_batch(self, *, wait: bool = False) -> int | str | None:
        """Retire the oldest in-flight batch and unwarm its sandbox pools."""
        if not self._inflight_batch_ids:
            return None
        batch_id = self._inflight_batch_ids.popleft()
        self._finish_batch_fn(batch_id, wait=wait)
        return batch_id


def prewarm_sandbox_images(
    images: list[str],
    *,
    replicas_override: int | None = None,
    namespace: str | None = None,
    node_selector: dict[str, str] | None = None,
    pod_annotations: Mapping[str, str] | None = None,
    network_policy_management: str | None = None,
    network_policy: Mapping[str, Any] | None = None,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> None:
    """Pre-warm a batch of sandbox images in parallel and pin them against LRU eviction."""
    _require_agent_sandbox_rl()
    if _is_lookahead_mode_enabled():
        return
    normalized = [apply_image_map(img, image_map=image_map) for img in images if img]
    unique_images = list(dict.fromkeys(normalized))
    if not unique_images:
        return
    resolved_ns = _resolve_default_namespace(namespace)
    resolved_sel = _resolve_default_node_selector(node_selector)
    task_objs = [Task(id=img, image=img, metadata={"ds": {"docker_image": img}}) for img in unique_images]
    with _FLEET_LOCK:
        fleet = _get_or_init_fleet_locked(
            namespace=resolved_ns,
            node_selector=resolved_sel,
            pod_annotations=pod_annotations,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
        )
        _RECENT_BATCH_IMAGES.append(set(unique_images))
        _register_and_warm_locked(fleet, task_objs, warm=True, replicas_override=replicas_override)
        _evict_excess_warmpools_locked(fleet)


async def async_prewarm_sandbox_images(
    images: list[str],
    *,
    replicas_override: int | None = None,
    namespace: str | None = None,
    node_selector: dict[str, str] | None = None,
    pod_annotations: Mapping[str, str] | None = None,
    network_policy_management: str | None = None,
    network_policy: Mapping[str, Any] | None = None,
    image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> None:
    """Async wrapper running ``prewarm_sandbox_images`` on the dedicated lifecycle executor."""
    if not images:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        get_lifecycle_executor(),
        lambda: prewarm_sandbox_images(
            images,
            replicas_override=replicas_override,
            namespace=namespace,
            node_selector=node_selector,
            pod_annotations=pod_annotations,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
            image_map=image_map,
        ),
    )


def _patch_claim_lifecycle(handle: Any, runtime_timeout: float) -> None:
    """Best-effort server-side TTL patch on ``SandboxClaim.spec.lifecycle`` so expired claims are reaped on crash."""
    claim_name = getattr(handle, "claim_name", None)
    cluster = getattr(handle, "_cluster", None)
    if not claim_name or cluster is None:
        return
    custom_api = getattr(getattr(cluster, "resources", None), "custom_api", None) or getattr(
        cluster, "custom_api", None
    )
    if custom_api is None or not hasattr(custom_api, "patch_namespaced_custom_object"):
        return
    shutdown_time = (datetime.now(timezone.utc) + timedelta(seconds=max(1, math.ceil(runtime_timeout)))).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    body = {
        "spec": {
            "lifecycle": {
                "shutdownTime": shutdown_time,
                "shutdownPolicy": "Delete",
            }
        }
    }
    try:
        custom_api.patch_namespaced_custom_object(
            group=_AGENT_SANDBOX_EXT_GROUP,
            version=_AGENT_SANDBOX_EXT_VERSION,
            namespace=getattr(cluster, "namespace", "default"),
            plural=_SANDBOX_CLAIMS_PLURAL,
            name=claim_name,
            body=body,
        )
    except Exception:
        logger.debug("Best-effort SandboxClaim lifecycle patch skipped for %s", claim_name, exc_info=True)


def _acquire_from_fleet(
    image: str,
    *,
    namespace: str,
    node_selector: dict[str, str] | None,
    runtime_timeout: float,
    pod_annotations: dict[str, str],
    network_policy_management: str,
    network_policy: dict[str, Any],
) -> Any:
    _require_agent_sandbox_rl()
    task_obj = Task(id=image, image=image, metadata={"ds": {"docker_image": image}})
    with _FLEET_LOCK:
        fleet = _get_or_init_fleet_locked(
            namespace=namespace,
            node_selector=node_selector,
            pod_annotations=pod_annotations,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
        )
        _ACTIVE_CLAIMS[image] = _ACTIVE_CLAIMS.get(image, 0) + 1
        try:
            _register_and_warm_locked(fleet, [task_obj], warm=not _is_lookahead_mode_enabled())
            _evict_excess_warmpools_locked(fleet)
        except Exception:
            _ACTIVE_CLAIMS[image] = max(0, _ACTIVE_CLAIMS[image] - 1)
            if _ACTIVE_CLAIMS[image] == 0:
                _ACTIVE_CLAIMS.pop(image, None)
            raise

    try:
        for attempt in range(5):
            try:
                handle = fleet.acquire(task_obj, pod_annotations=pod_annotations or None)
                _patch_claim_lifecycle(handle, runtime_timeout)
                return handle
            except Exception as exc:
                if attempt < 4:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise exc
    except Exception:
        with _FLEET_LOCK:
            _ACTIVE_CLAIMS[image] = max(0, _ACTIVE_CLAIMS[image] - 1)
            if _ACTIVE_CLAIMS[image] == 0:
                _ACTIVE_CLAIMS.pop(image, None)
        raise


def _release_to_fleet(image: str, handle: Any) -> None:
    try:
        if handle is not None and _GLOBAL_FLEET is not None:
            _GLOBAL_FLEET.release(handle)
    finally:
        with _FLEET_LOCK:
            if image in _ACTIVE_CLAIMS:
                _ACTIVE_CLAIMS[image] = max(0, _ACTIVE_CLAIMS[image] - 1)
                if _ACTIVE_CLAIMS[image] == 0:
                    _ACTIVE_CLAIMS.pop(image, None)
            if image in _IDLE_LRU:
                _IDLE_LRU.move_to_end(image)
            if _GLOBAL_FLEET is not None:
                _evict_excess_warmpools_locked(_GLOBAL_FLEET)


class _GKEStatefulShell:
    """Lightweight shell wrapper that persists ``cwd`` and ``env`` without requiring ``tmux``."""

    def __init__(self, sandbox: GKEAgentSandbox, *, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        self._sandbox = sandbox
        self.cwd = cwd or "/testbed"
        self.env = dict(env or {})
        self._env_file = f"/tmp/.uni_agent_env_{uuid.uuid4().hex[:8]}"
        self._initialized = False

    async def run(self, command: str, *, timeout: float | None = 120.0) -> ExecResult:
        if not self._initialized:
            exports = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in self.env.items())
            await self._sandbox.exec_shell(f"cat <<'__EOF__' > {self._env_file}\n{exports}\n__EOF__", timeout=30.0)
            self._initialized = True
        wrapped = (
            "if [ -d /opt/miniconda3/envs/testbed/bin ]; then "
            'export PATH="/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:${PATH}"; fi\n'
            'if [ -d /testbed/.venv/bin ]; then export PATH="/testbed/.venv/bin:${PATH}"; fi\n'
            f"[ -f {self._env_file} ] && . {self._env_file} >/dev/null 2>&1 || true\n"
            f"cd {shlex.quote(self.cwd)} 2>/dev/null || cd /\n"
            f"{command}\n"
            "__ec=$?\n"
            f'printf "\\n{_PWD_SENTINEL}:%s\\n" "$(pwd)"\n'
            f"export -p > {self._env_file} 2>/dev/null || true\n"
            "exit $__ec"
        )
        res = await self._sandbox.exec(["bash", "-c", wrapped], timeout=timeout, workdir=self.cwd, env=self.env)
        m = re.search(rf"\n?{re.escape(_PWD_SENTINEL)}:([^\r\n]+)\r?\n?$", res.stdout or "")
        stdout = res.stdout or ""
        if m:
            self.cwd = m.group(1).strip() or self.cwd
            stdout = stdout[: m.start()]
        return ExecResult(exit_code=res.exit_code, stdout=stdout, stderr=res.stderr or "")

    async def close(self) -> None:
        if self._initialized:
            await self._sandbox.exec(["rm", "-f", self._env_file], timeout=10.0)
            self._initialized = False


@register_sandbox("gke_agent_sandbox")
class GKEAgentSandbox(Sandbox):
    """GKE / Kubernetes sandbox provider backed by ``agent_sandbox_rl`` warm pools."""

    supports_shell: ClassVar[bool] = True

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        runtime_timeout: float = 1800.0,
        namespace: str | None = None,
        node_selector: Mapping[str, str] | None = None,
        pod_annotations: Mapping[str, str] | None = None,
        network_policy_management: str | None = None,
        network_policy: Mapping[str, Any] | None = None,
        image_map: Sequence[ImageMap | Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
    ) -> None:
        timeout_val = float(runtime_timeout)
        if not math.isfinite(timeout_val) or timeout_val <= 0:
            raise ValueError(f"runtime_timeout must be a finite positive number, got {runtime_timeout!r}")
        self.image = apply_image_map(image, image_map=image_map)
        self.runtime_timeout = timeout_val
        self.namespace = _resolve_default_namespace(namespace)
        self.node_selector = _resolve_default_node_selector(node_selector)
        self.pod_annotations = _resolve_default_pod_annotations(pod_annotations)
        self.network_policy_management = _resolve_network_policy_management(network_policy_management)
        self.network_policy = _resolve_network_policy(network_policy)
        self._handle: Any = None
        self._pod_name: str | None = None
        self._started = False
        self._deadline: float | None = None
        self._expired = False
        self._expiry_task: asyncio.Task[None] | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> GKEAgentSandbox:
        return cls(
            image=config.image or "python:3.12",
            runtime_timeout=config.runtime_timeout,
            **config.sandbox_kwargs,
        )

    def _is_fatal_exec_error(self, exc: BaseException) -> bool:
        return isinstance(exc, SandboxRuntimeTimeoutError)

    def _get_core_v1_api(self) -> Any:
        if self._handle is not None and getattr(self._handle, "_cluster", None) is not None:
            return self._handle._cluster.exec_core_api()
        if k8s_client is None or k8s_config is None:
            raise ImportError("The 'kubernetes' package is required for GKEAgentSandbox.")
        _load_kube_config_once()
        return k8s_client.CoreV1Api()

    def _remaining_runtime_seconds(self) -> float | None:
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    def _cancel_expiry_task(self) -> None:
        task, self._expiry_task = self._expiry_task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _expire_after_runtime_timeout(self) -> None:
        try:
            await asyncio.sleep(self.runtime_timeout)
        except asyncio.CancelledError:
            return
        self._expiry_task = None
        self._expired = True
        logger.warning(
            "GKEAgentSandbox exceeded runtime_timeout=%.1fs (pod=%s); releasing sandbox",
            self.runtime_timeout,
            self._pod_name,
        )
        try:
            await self.stop()
        except Exception:
            logger.warning("GKEAgentSandbox lifetime cleanup failed (pod=%s)", self._pod_name, exc_info=True)

    async def start(self) -> None:
        if self._started:
            return
        self._expired = False
        loop = asyncio.get_running_loop()
        self._handle = await loop.run_in_executor(
            get_lifecycle_executor(),
            lambda: _acquire_from_fleet(
                self.image,
                namespace=self.namespace,
                node_selector=self.node_selector,
                runtime_timeout=self.runtime_timeout,
                pod_annotations=self.pod_annotations,
                network_policy_management=self.network_policy_management,
                network_policy=self.network_policy,
            ),
        )
        self._pod_name = getattr(self._handle, "pod_name", None)
        if getattr(self._handle, "_cluster", None) is not None:
            self.namespace = self._handle._cluster.namespace
        self._deadline = time.monotonic() + self.runtime_timeout
        self._started = True
        self._expiry_task = asyncio.create_task(self._expire_after_runtime_timeout())

    async def stop(self) -> None:
        self._cancel_expiry_task()
        self._started = False
        self._deadline = None
        handle, self._handle = self._handle, None
        self._pod_name = None
        if handle is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(get_lifecycle_executor(), _release_to_fleet, self.image, handle)

    async def is_alive(self) -> bool:
        if not self._started or self._expired or self._handle is None:
            return False
        remaining = self._remaining_runtime_seconds()
        return remaining is None or remaining > 0

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> _GKEStatefulShell:
        return _GKEStatefulShell(self, cwd=cwd, env=env)

    def _exec_in_pod_sync(
        self,
        argv: list[str],
        timeout: float,
        workdir: str | None,
        env: dict[str, str] | None,
        *,
        stream_grace: float = 10.0,
    ) -> ExecResult:
        if self._pod_name is None or k8s_stream is None:
            raise RuntimeError("GKEAgentSandbox is not started or kubernetes.stream is unavailable.")
        inner_cmd = " ".join(shlex.quote(a) for a in argv)
        parts = [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        full_cmd = (" && ".join(parts) + f" && exec {inner_cmd}") if parts else inner_cmd
        timeout_int = max(1, math.ceil(timeout))
        wrapped_argv = ["/bin/sh", "-c", f"timeout {timeout_int}s /bin/sh -c {shlex.quote(full_cmd)}"]

        api = self._get_core_v1_api()
        resp = k8s_stream(
            api.connect_get_namespaced_pod_exec,
            self._pod_name,
            self.namespace,
            command=wrapped_argv,
            container=RUNTIME_CONTAINER_NAME,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        deadline = time.monotonic() + timeout + stream_grace
        try:
            while resp.is_open():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    resp.close()
                    raise TimeoutError(f"Pod exec exceeded {timeout}s on {self._pod_name}/{RUNTIME_CONTAINER_NAME}")
                resp.update(timeout=min(1.0, remaining))
                if resp.peek_stdout():
                    stdout_chunks.append(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_chunks.append(resp.read_stderr())
            # Frames that arrive together with the close frame are buffered but never surface through the
            # is_open() loop; drain them, or large outputs (e.g. read_file's base64) come back truncated.
            channels = getattr(resp, "_channels", None)
            if isinstance(channels, dict):
                stdout_chunks.append(channels.pop(1, "") or "")
                stderr_chunks.append(channels.pop(2, "") or "")
            rc = resp.returncode
        finally:
            try:
                resp.close()
            except Exception:
                pass

        exit_code = int(rc) if rc is not None else 0
        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)
        if exit_code == 124:
            raise TimeoutError(stdout_str or stderr_str or f"Command timed out after {timeout}s")
        return ExecResult(exit_code=exit_code, stdout=stdout_str, stderr=stderr_str)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._expired:
            raise SandboxRuntimeTimeoutError(f"GKEAgentSandbox exceeded runtime_timeout of {self.runtime_timeout}s")
        remaining = self._remaining_runtime_seconds()
        if remaining is not None and remaining <= 0:
            self._expired = True
            await self.stop()
            raise SandboxRuntimeTimeoutError(f"GKEAgentSandbox exceeded runtime_timeout of {self.runtime_timeout}s")
        if not self._started or self._handle is None:
            raise RuntimeError("GKEAgentSandbox is not started.")

        requested_timeout = max(float(timeout or 120.0), 1.0)
        clamped_by_runtime = remaining is not None and requested_timeout > remaining
        eff_timeout = min(requested_timeout, remaining) if remaining is not None else requested_timeout
        wait_timeout = (remaining + 0.5) if clamped_by_runtime and remaining is not None else (eff_timeout + 15.0)
        stream_grace = 0.2 if clamped_by_runtime else 10.0

        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    get_exec_executor(),
                    lambda: self._exec_in_pod_sync(argv, eff_timeout, workdir, env, stream_grace=stream_grace),
                ),
                timeout=wait_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            post_remaining = self._remaining_runtime_seconds()
            if self._expired or (post_remaining is not None and post_remaining <= 0):
                self._expired = True
                await self.stop()
                raise SandboxRuntimeTimeoutError(
                    f"GKEAgentSandbox exceeded runtime_timeout of {self.runtime_timeout}s"
                ) from exc
            raise
