"""Runtime patch so multi-policy trainers can coexist on one Ray cluster.

Why this is needed
------------------
verl assumes a single-policy process and hard-codes several Ray-global
identifiers, all of which collide when multiple policies share one Ray
cluster:

- Placement-group names: ``ResourcePoolManager.create_resource_pool`` hard-
  codes the name prefix to the pool spec key ("global_pool"). Ray 2.55.x
  enforces unique placement-group names, so the second policy fails with
  ``name 'global_poolverl_group_4:0' already exists``.
- Reward-worker actor names: ``RewardLoopManager._init_reward_loop_workers``
  hard-codes ``reward_loop_worker_{i}``.
- vLLM server actor names: the creator (``vLLMReplica.launch_servers``) and
  the lookup side (``ServerAdapter._ensure_server_handle``) both hard-code
  ``{prefix}server_{rank}_{node}``, which collides across policies.

This module monkey-patches verl at runtime so every such identifier carries a
policy label (``config.policy_name``):

- ``RayResourcePool.__init__`` appends a unique 8-hex suffix;
- ``PPOTrainer._init_resource_pool_mgr`` records the policy label and injects
  it into ``rollout.custom.policy_name`` (a verl-native extension field that
  survives ``omega_conf_to_dataclass``, unlike ad-hoc config attributes);
- ``ResourcePoolManager.create_resource_pool`` prepends that label;
- ``RewardLoopManager._init_reward_loop_workers`` prefixes actor names;
- both ``vLLMReplica._get_server_name_prefix`` (creator) and
  ``ServerAdapter._get_server_name_prefix`` (lookup) read
  ``config.custom.policy_name``, so both sides build the same policy-unique
  actor name, e.g. ``policy_1_vllm_server_0_0``.

Resulting names: ``policy_1_<hex>verl_group_8_8:0``,
``policy_1_reward_loop_worker_0``, ``policy_1_vllm_server_0_0``. No verl file
is modified. The TaskRunner loads the full patch before trainer construction
through ``config.example_patch_fqn``; Ray's ``worker_process_setup_hook``
loads the lookup-only patch in every Python worker process.
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

logger = logging.getLogger(__name__)

_ORIG_INIT = None
_ORIG_CREATE = None
_ORIG_INIT_RESOURCE_POOL_MGR = None  # lazy-imported from verl trainer_base
_ORIG_INIT_REWARD_LOOP_WORKERS = None  # lazy-imported from verl reward_loop
_ORIG_VLLM_REPLICA_NAME_PREFIX = None  # lazy-imported from verl vllm_async_server
_ORIG_SERVER_ADAPTER_NAME_PREFIX = None  # lazy-imported from verl vllm_rollout
_ORIG_REPLICA_INIT_COLOCATED = None  # lazy-imported from verl workers.rollout.replica
_ORIG_REPLICA_INIT_STANDALONE = None  # lazy-imported from verl workers.rollout.replica
_ORIG_VLLM_REPLICA_LAUNCH_SERVERS = None  # lazy-imported from verl vllm_async_server
_ORIG_SERVER_ADAPTER_INIT = None  # lazy-imported from verl vllm_rollout
_ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS = None  # lazy-imported from verl vllm_rollout
_ORIG_VLLM_HTTP_SERVER_INIT = None  # lazy-imported from verl vllm_async_server
_PATCHED = False
_WORKER_PATCHED = False
_LOOKUP_PATCHED = False


def _policy_label_from_rollout_config(config) -> str | None:
    """Read ``config.custom.policy_name`` (survives dataclass conversion)."""
    try:
        custom = getattr(config, "custom", None) or {}
        if isinstance(custom, dict):
            return custom.get("policy_name")
    except Exception:
        pass
    return None


def _patched_init(
    self,
    process_on_nodes=None,
    use_gpu=True,
    name_prefix=None,
    max_colocate_count=10,
    detached=False,
    accelerator_type=None,
):
    suffix = uuid4().hex[:8]
    if name_prefix is None:
        name_prefix = f"pool_{suffix}"
    else:
        name_prefix = f"{name_prefix}_{suffix}"
    return _ORIG_INIT(
        self,
        process_on_nodes,
        use_gpu,
        name_prefix,
        max_colocate_count,
        detached,
        accelerator_type,
    )


def _patched_init_resource_pool_mgr(self):
    """Run verl's original pool-manager init, record the policy label, and
    inject it into ``rollout.custom.policy_name`` so vLLM server actor names
    (creator and lookup sides) share one source of truth."""
    _ORIG_INIT_RESOURCE_POOL_MGR(self)
    label = None
    try:
        label = self.config.get("policy_name")
    except Exception:
        label = None
    if not label:
        return

    label = str(label)
    self.resource_pool_manager._pool_label = label
    try:
        from omegaconf import OmegaConf, open_dict

        rollout = self.config.actor_rollout_ref.rollout
        custom = rollout.get("custom")
        if custom is None:
            with open_dict(rollout):
                rollout.custom = OmegaConf.create({})
            custom = rollout.custom
        with open_dict(custom):
            custom["policy_name"] = label
    except Exception as exc:
        logger.warning(
            "multi_policy.verl_patch: failed to inject rollout.custom.policy_name (%s); "
            "vLLM server actor names may not be policy-uniqued",
            exc,
        )


def _patched_create_resource_pool(self):
    """Create pools as verl does, then prepend the policy label if known."""
    _ORIG_CREATE(self)
    label = getattr(self, "_pool_label", None)
    for name, pool in self.resource_pool_dict.items():
        if label:
            # Replace verl's redundant "global_pool" prefix with the policy label.
            pool.name_prefix = f"{label}_{uuid4().hex[:8]}"
        else:
            pool.name_prefix = f"{name}_{uuid4().hex[:8]}"


def _patched_init_reward_loop_workers(self):
    """verl hard-codes reward-worker actor names (``reward_loop_worker_{i}``),
    which collide across policies on one Ray cluster. Prefix them with the
    policy label (``{label}_reward_loop_worker_{i}``) when available."""
    import ray
    import ray.util

    label = None
    try:
        label = self.config.get("policy_name")
    except Exception:
        label = None
    if not label:
        return _ORIG_INIT_REWARD_LOOP_WORKERS(self)

    self.reward_loop_workers = []
    num_workers = self.config.reward.num_workers
    node_ids = [
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("CPU", 0) > 0
    ]

    for i in range(num_workers):
        node_id = node_ids[i % len(node_ids)]
        self.reward_loop_workers.append(
            self.reward_loop_workers_class.options(
                name=f"{label}_reward_loop_worker_{i}",
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=True,
                ),
            ).remote(self.config, self.reward_router_address)
        )


def _patched_vllm_replica_get_server_name_prefix(self):
    """Creator side: policy-unique vLLM server actor prefix."""
    label = _policy_label_from_rollout_config(self.config)
    return f"{label}_vllm_" if label else "vllm_"


def _patched_server_adapter_get_server_name_prefix(self):
    """Lookup side: must match the creator-side actor name exactly."""
    label = _policy_label_from_rollout_config(self.config)
    return f"{label}_vllm_" if label else "vllm_"


def _policy_suffixed_replica(self) -> None:
    """Inject the policy label into a RolloutReplica's name_suffix (idempotent).

    verl names checkpoint-engine worker actors as
    ``{rollout_colocate|rollout_standalone}_{replica_rank}{name_suffix}`` without
    any policy label, so two policies on one Ray cluster collide on
    e.g. ``rollout_standalone_1CheckpointEngineWorker_0:0``. Prefixing
    ``name_suffix`` with the policy label makes every such actor unique.
    """
    label = _policy_label_from_rollout_config(self.config)
    if label and not self.name_suffix.startswith(f"_{label}"):
        self.name_suffix = f"_{label}" + self.name_suffix


def _patched_replica_init_colocated(self, *args, **kwargs):
    _policy_suffixed_replica(self)
    return _ORIG_REPLICA_INIT_COLOCATED(self, *args, **kwargs)


def _patched_replica_init_standalone(self, *args, **kwargs):
    _policy_suffixed_replica(self)
    return _ORIG_REPLICA_INIT_STANDALONE(self, *args, **kwargs)


async def _patched_vllm_replica_launch_servers(self):
    """Create vLLM server actors WITHOUT the policy suffix we injected into name_suffix.

    vLLMReplica inherits our patched ``init_standalone``/``init_colocated``, so
    ``name_suffix`` already carries ``_{policy}`` and would leak into the server
    actor name (``policy_1_vllm_server_1_0_policy_1``). The lookup side
    (``ServerAdapter._ensure_server_handle``) never appends ``name_suffix`` and
    builds ``policy_1_vllm_server_1_0`` — the server prefix already carries the
    policy label via ``_get_server_name_prefix``, so the suffix is redundant.
    Strip only our injected ``_{label}`` prefix for the duration of the call.
    """
    label = _policy_label_from_rollout_config(self.config)
    injected = f"_{label}" if label else None
    saved = self.name_suffix
    if injected and saved.startswith(injected):
        self.name_suffix = saved[len(injected) :]
    try:
        return await _ORIG_VLLM_REPLICA_LAUNCH_SERVERS(self)
    finally:
        self.name_suffix = saved


def _make_patched_server_adapter_init(orig):
    """Make the sender-side ZMQ weight-transfer socket path per-policy.

    Captures the original in a closure (rather than a module global) so the
    patched method survives Ray actor serialization into worker processes that
    never called ``apply_patch()``.
    """

    def _patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _policy_suffixed_zmq_handle(self)

    return _patched


def _make_patched_server_adapter_update_weights(orig):
    """Ensure the sender-side ZMQ socket path carries the policy label.

    Patching ``__init__`` via the worker watcher races with construction (the
    method runs immediately after the module import). ``update_weights`` runs
    at weight-sync time, long after the watcher has fired, so it is the
    race-free hook for the sender-side label.
    """

    async def _patched(
        self, weights, global_steps=None, wire_format="named_tensors", **kwargs
    ):
        _policy_suffixed_zmq_handle(self)
        return await orig(
            self, weights, global_steps=global_steps, wire_format=wire_format, **kwargs
        )

    return _patched


def _policy_suffixed_zmq_handle(self) -> None:
    """Make a sender-side ZMQ weight-transfer path per-policy (idempotent).

    verl builds the path as
    ``/tmp/rl-colocate-zmq-{ray_job_id}-replica-{replica_rank}-rank-{local_rank}.sock``
    on both sides. The receiver (vLLM worker) derives its copy from the
    ``VERL_RAY_JOB_ID`` env var, which we override to ``{ray_job_id}_{policy}``
    in ``_make_patched_vllm_http_server_init`` so the override survives the
    spawned vLLM worker process (multiprocessing ``spawn`` re-imports modules,
    so a method patch would be lost there). This helper applies the SAME
    ``_{policy}`` suffix to the sender's job-id component so both ends agree.
    """
    label = _policy_label_from_rollout_config(getattr(self, "config", None))
    if (
        label
        and hasattr(self, "zmq_handle")
        and f"_{label}-replica-" not in self.zmq_handle
    ):
        self.zmq_handle = self.zmq_handle.replace("-replica-", f"_{label}-replica-", 1)


def _make_patched_vllm_http_server_init(orig):
    """Forward the policy label into the vLLM worker subprocess environment.

    The vLLM engine workers are multiprocessing-``spawn``ed children, so they
    re-import verl modules and cannot inherit method patches. They DO inherit
    the environment, so:
    - ``VERL_POLICY_LABEL`` lets any env-based receiver-side logic know the
      policy;
    - overriding ``VERL_RAY_JOB_ID`` (which verl's ``_get_zmq_handle`` reads to
      build the weight-transfer socket path) to ``{ray_job_id}_{policy}`` makes
      the receiver's path per-policy without touching verl.
    """

    def _patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        label = _policy_label_from_rollout_config(getattr(self, "config", None))
        if label:
            os.environ["VERL_POLICY_LABEL"] = label
            job_id = os.environ.get("VERL_RAY_JOB_ID")
            if job_id and not job_id.endswith(f"_{label}"):
                os.environ["VERL_RAY_JOB_ID"] = f"{job_id}_{label}"

    return _patched


def _patch_server_adapter_in_module(module) -> None:
    """Patch ServerAdapter._get_server_name_prefix + __init__ once the module is loaded."""
    global _ORIG_SERVER_ADAPTER_NAME_PREFIX, _LOOKUP_PATCHED, _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS
    if _LOOKUP_PATCHED:
        return
    server_adapter = getattr(module, "ServerAdapter", None)
    if server_adapter is None:
        return
    if _ORIG_SERVER_ADAPTER_NAME_PREFIX is None:
        _ORIG_SERVER_ADAPTER_NAME_PREFIX = getattr(
            server_adapter, "_get_server_name_prefix", None
        )
    server_adapter._get_server_name_prefix = (
        _patched_server_adapter_get_server_name_prefix
    )
    # Worker-side sender patch: ServerAdapter.__init__ appends the policy label
    # to the ZMQ weight-transfer socket path. This process (a Ray worker) never
    # ran apply_patch(), so the closure factory captures the original locally.
    global _ORIG_SERVER_ADAPTER_INIT
    if _ORIG_SERVER_ADAPTER_INIT is None:
        _ORIG_SERVER_ADAPTER_INIT = server_adapter.__init__
        server_adapter.__init__ = _make_patched_server_adapter_init(
            server_adapter.__init__
        )
    if _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS is None:
        _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS = server_adapter.update_weights
        server_adapter.update_weights = _make_patched_server_adapter_update_weights(
            server_adapter.update_weights
        )
    _LOOKUP_PATCHED = True


def _apply_worker_lookup_patch_now() -> None:
    """Directly patch ServerAdapter._get_server_name_prefix in this process.

    Safe to call from worker training code (e.g. ActorRolloutRefWorker.init_model),
    where Ray has already set CUDA_VISIBLE_DEVICES. Must NOT be called from the
    worker_process_setup_hook, which runs before Ray configures the GPU env.
    """
    global _ORIG_SERVER_ADAPTER_NAME_PREFIX
    global _LOOKUP_PATCHED
    if _LOOKUP_PATCHED:
        return
    try:
        from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
    except Exception as exc:
        logger.warning(
            "multi_policy.verl_patch: worker lookup patch skipped, "
            "cannot import ServerAdapter (%s)",
            exc,
        )
        return
    if _ORIG_SERVER_ADAPTER_NAME_PREFIX is None:
        _ORIG_SERVER_ADAPTER_NAME_PREFIX = getattr(
            ServerAdapter, "_get_server_name_prefix", None
        )
    ServerAdapter._get_server_name_prefix = (
        _patched_server_adapter_get_server_name_prefix
    )
    _LOOKUP_PATCHED = True
    logger.info("multi_policy.verl_patch: worker lookup patch enabled")


def apply_worker_patch() -> None:
    """Install a lightweight watcher that patches ServerAdapter once verl loads.

    The worker_process_setup_hook runs before Ray configures the per-worker GPU
    environment, so it must NOT import verl (that would initialize CUDA with the
    full device list and break FSDP). A daemon thread polls ``sys.modules`` and
    patches ``ServerAdapter._get_server_name_prefix`` the moment worker code
    imports ``verl.workers.rollout.vllm_rollout.vllm_rollout`` (by then verl has
    set up the CUDA environment).
    """
    global _WORKER_PATCHED
    if _WORKER_PATCHED:
        return

    import sys
    import threading
    import time

    target = "verl.workers.rollout.vllm_rollout.vllm_rollout"

    def _wait_and_patch():
        while target not in sys.modules:
            time.sleep(0.05)
        # "in sys.modules" fires while exec_module is still running (before the
        # ServerAdapter class exists). Wait for the class attribute to appear so
        # we patch the fully-defined class, not a half-loaded module.
        module = sys.modules[target]
        while not hasattr(module, "ServerAdapter"):
            time.sleep(0.05)
        try:
            _patch_server_adapter_in_module(module)
            logger.info("multi_policy.verl_patch: worker lookup patch enabled")
        except Exception:
            pass

    threading.Thread(
        target=_wait_and_patch, daemon=True, name="ma-lookup-patch"
    ).start()
    _WORKER_PATCHED = True


def apply_patch() -> None:
    """Install the multi-policy uniqueness patches (idempotent)."""
    global _PATCHED, _ORIG_INIT, _ORIG_CREATE, _ORIG_INIT_RESOURCE_POOL_MGR, _ORIG_INIT_REWARD_LOOP_WORKERS, _ORIG_VLLM_REPLICA_NAME_PREFIX, _ORIG_REPLICA_INIT_COLOCATED, _ORIG_REPLICA_INIT_STANDALONE, _ORIG_VLLM_REPLICA_LAUNCH_SERVERS, _ORIG_SERVER_ADAPTER_INIT, _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS, _ORIG_VLLM_HTTP_SERVER_INIT
    if _PATCHED:
        return

    from verl.single_controller.ray.base import RayResourcePool
    from verl.single_controller.ray.base import ResourcePoolManager

    try:
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    except Exception as exc:  # optional heavy deps (megatron/flashinfer) unavailable
        logger.warning(
            "multi_policy.verl_patch: trainer_base import failed (%s); "
            "installing unique-name patch only (no policy labels)",
            exc,
        )
        PPOTrainer = None

    try:
        from verl.experimental.reward_loop.reward_loop import RewardLoopManager
    except Exception as exc:  # optional heavy deps unavailable
        logger.warning(
            "multi_policy.verl_patch: reward_loop import failed (%s); "
            "reward-worker actor names will not be policy-uniqued",
            exc,
        )
        RewardLoopManager = None

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
        from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
    except Exception as exc:  # optional heavy deps (vllm) unavailable
        logger.warning(
            "multi_policy.verl_patch: vllm rollout import failed (%s); "
            "vLLM server actor names will not be policy-uniqued",
            exc,
        )
        vLLMReplica = None
        ServerAdapter = None

    try:
        from verl.workers.rollout.replica import RolloutReplica
    except Exception as exc:  # optional heavy deps (vllm) unavailable
        logger.warning(
            "multi_policy.verl_patch: rollout replica import failed (%s); "
            "checkpoint-engine worker actor names will not be policy-uniqued",
            exc,
        )
        RolloutReplica = None

    _ORIG_INIT = RayResourcePool.__init__
    _ORIG_CREATE = ResourcePoolManager.create_resource_pool
    RayResourcePool.__init__ = _patched_init
    ResourcePoolManager.create_resource_pool = _patched_create_resource_pool
    if PPOTrainer is not None:
        _ORIG_INIT_RESOURCE_POOL_MGR = PPOTrainer._init_resource_pool_mgr
        PPOTrainer._init_resource_pool_mgr = _patched_init_resource_pool_mgr
    if RewardLoopManager is not None:
        _ORIG_INIT_REWARD_LOOP_WORKERS = RewardLoopManager._init_reward_loop_workers
        RewardLoopManager._init_reward_loop_workers = _patched_init_reward_loop_workers
    if vLLMReplica is not None:
        _ORIG_VLLM_REPLICA_NAME_PREFIX = vLLMReplica._get_server_name_prefix
        vLLMReplica._get_server_name_prefix = (
            _patched_vllm_replica_get_server_name_prefix
        )
        _ORIG_VLLM_REPLICA_LAUNCH_SERVERS = vLLMReplica.launch_servers
        vLLMReplica.launch_servers = _patched_vllm_replica_launch_servers
    if ServerAdapter is not None:
        apply_worker_patch()
        _ORIG_SERVER_ADAPTER_INIT = ServerAdapter.__init__
        ServerAdapter.__init__ = _make_patched_server_adapter_init(
            ServerAdapter.__init__
        )
    if vLLMReplica is not None:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        _ORIG_VLLM_HTTP_SERVER_INIT = vLLMHttpServer.__init__
        vLLMHttpServer.__init__ = _make_patched_vllm_http_server_init(
            vLLMHttpServer.__init__
        )
    if RolloutReplica is not None:
        _ORIG_REPLICA_INIT_COLOCATED = RolloutReplica.init_colocated
        _ORIG_REPLICA_INIT_STANDALONE = RolloutReplica.init_standalone
        RolloutReplica.init_colocated = _patched_replica_init_colocated
        RolloutReplica.init_standalone = _patched_replica_init_standalone
    _PATCHED = True
    logger.info(
        "multi_policy.verl_patch: unique + policy-labeled placement-group "
        "name prefixes, reward-worker actor names, vLLM server actor names, and "
        "rollout checkpoint-engine worker names enabled"
    )


def restore() -> None:
    """Restore verl's original methods (idempotent)."""
    global _PATCHED, _WORKER_PATCHED, _LOOKUP_PATCHED, _ORIG_INIT, _ORIG_CREATE, _ORIG_INIT_RESOURCE_POOL_MGR, _ORIG_INIT_REWARD_LOOP_WORKERS, _ORIG_VLLM_REPLICA_NAME_PREFIX, _ORIG_SERVER_ADAPTER_NAME_PREFIX, _ORIG_REPLICA_INIT_COLOCATED, _ORIG_REPLICA_INIT_STANDALONE, _ORIG_VLLM_REPLICA_LAUNCH_SERVERS, _ORIG_SERVER_ADAPTER_INIT, _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS, _ORIG_VLLM_HTTP_SERVER_INIT
    if _PATCHED:
        from verl.single_controller.ray.base import RayResourcePool
        from verl.single_controller.ray.base import ResourcePoolManager

        RayResourcePool.__init__ = _ORIG_INIT
        ResourcePoolManager.create_resource_pool = _ORIG_CREATE
        _ORIG_INIT = None
        _ORIG_CREATE = None
        if _ORIG_INIT_RESOURCE_POOL_MGR is not None:
            from verl.trainer.ppo.v1.trainer_base import PPOTrainer

            PPOTrainer._init_resource_pool_mgr = _ORIG_INIT_RESOURCE_POOL_MGR
            _ORIG_INIT_RESOURCE_POOL_MGR = None
        if _ORIG_INIT_REWARD_LOOP_WORKERS is not None:
            from verl.experimental.reward_loop.reward_loop import RewardLoopManager

            RewardLoopManager._init_reward_loop_workers = _ORIG_INIT_REWARD_LOOP_WORKERS
            _ORIG_INIT_REWARD_LOOP_WORKERS = None
        if _ORIG_VLLM_REPLICA_NAME_PREFIX is not None:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

            vLLMReplica._get_server_name_prefix = _ORIG_VLLM_REPLICA_NAME_PREFIX
            _ORIG_VLLM_REPLICA_NAME_PREFIX = None
            if _ORIG_VLLM_REPLICA_LAUNCH_SERVERS is not None:
                vLLMReplica.launch_servers = _ORIG_VLLM_REPLICA_LAUNCH_SERVERS
                _ORIG_VLLM_REPLICA_LAUNCH_SERVERS = None
        if (
            _ORIG_REPLICA_INIT_COLOCATED is not None
            or _ORIG_REPLICA_INIT_STANDALONE is not None
        ):
            from verl.workers.rollout.replica import RolloutReplica

            if _ORIG_REPLICA_INIT_COLOCATED is not None:
                RolloutReplica.init_colocated = _ORIG_REPLICA_INIT_COLOCATED
                _ORIG_REPLICA_INIT_COLOCATED = None
            if _ORIG_REPLICA_INIT_STANDALONE is not None:
                RolloutReplica.init_standalone = _ORIG_REPLICA_INIT_STANDALONE
                _ORIG_REPLICA_INIT_STANDALONE = None
        if _ORIG_SERVER_ADAPTER_INIT is not None:
            from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

            ServerAdapter.__init__ = _ORIG_SERVER_ADAPTER_INIT
            _ORIG_SERVER_ADAPTER_INIT = None
        if _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS is not None:
            from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

            ServerAdapter.update_weights = _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS
            _ORIG_SERVER_ADAPTER_UPDATE_WEIGHTS = None
        if _ORIG_VLLM_HTTP_SERVER_INIT is not None:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (
                vLLMHttpServer,
            )

            vLLMHttpServer.__init__ = _ORIG_VLLM_HTTP_SERVER_INIT
            _ORIG_VLLM_HTTP_SERVER_INIT = None
        _PATCHED = False

    if _WORKER_PATCHED:
        if _ORIG_SERVER_ADAPTER_NAME_PREFIX is not None:
            from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

            ServerAdapter._get_server_name_prefix = _ORIG_SERVER_ADAPTER_NAME_PREFIX
            _ORIG_SERVER_ADAPTER_NAME_PREFIX = None
        _LOOKUP_PATCHED = False
        _WORKER_PATCHED = False
