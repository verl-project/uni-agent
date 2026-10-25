from __future__ import annotations

import json
import logging
import os
import posixpath
import shutil
from collections.abc import Mapping
from concurrent.futures import Executor, wait
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from packaging.version import InvalidVersion, Version
from torchdata.stateful_dataloader import StatefulDataLoader

if TYPE_CHECKING:
    from examples.multi_policy.trainer.single_async_policy_trainer import (
        SingleAsyncPPOTrainer,
    )
    from examples.multi_policy.trainer.single_policy_trainer import SinglePPOTrainer

    PolicyTrainer = SinglePPOTrainer | SingleAsyncPPOTrainer


logger = logging.getLogger(__name__)


def supports_transfer_queue_checkpoint() -> bool:
    """Return whether the installed TransferQueue supports state snapshots."""
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


class MultiPolicyCheckpointManager:
    """Own the atomic save and restore lifecycle of a multi-policy checkpoint."""

    def __init__(
        self,
        *,
        config: DictConfig,
        policy_configs: Mapping[str, DictConfig],
        trainer_mode: str,
    ) -> None:
        self.config = config
        self.policy_configs = policy_configs
        self.trainer_mode = trainer_mode
        self.validate_save_config()

    def validate_save_config(self) -> None:
        checkpoint_configs = (
            (
                "actor",
                "actor_rollout_ref.actor.checkpoint.async_save",
                "actor_rollout_ref.actor.strategy",
            ),
            ("critic", "critic.checkpoint.async_save", "critic.strategy"),
        )
        for policy_name, policy_config in self.policy_configs.items():
            for component, async_path, strategy_path in checkpoint_configs:
                if not OmegaConf.select(policy_config, async_path, default=False):
                    continue
                strategy = str(
                    OmegaConf.select(policy_config, strategy_path, default="unknown")
                ).lower()
                if strategy != "megatron":
                    backend = "FSDP" if strategy == "fsdp" else strategy
                    raise ValueError(
                        f"Policy {policy_name!r} {component} uses {backend} with async_save=true; "
                        "checkpoint async_save is only implemented by the Megatron backend"
                    )
                raise ValueError(
                    f"Policy {policy_name!r} {component} enables async_save=true, but the multi-policy "
                    "outer completion barrier is not implemented yet; set async_save=false"
                )

    def load(
        self,
        *,
        policy_trainers: Mapping[str, PolicyTrainer],
        train_dataloader: StatefulDataLoader,
    ) -> int | None:
        checkpoint_dir = self.resolve_checkpoint_dir()
        if checkpoint_dir is None:
            return None

        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        step_text = checkpoint_name.removeprefix("global_step_")
        if checkpoint_name == step_text or not step_text.isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir}")
        self._validate_manifest(checkpoint_dir, policy_trainers)

        policy_paths = {}
        for policy_name, trainer in policy_trainers.items():
            policy_checkpoint_dir = os.path.join(
                checkpoint_dir, "policies", policy_name
            )
            actor_path = os.path.join(policy_checkpoint_dir, "actor")
            if not os.path.isdir(actor_path):
                raise FileNotFoundError(
                    f"Checkpoint for policy {policy_name!r} is missing actor directory: {actor_path}"
                )

            critic_path = None
            if trainer.use_critic:
                critic_path = os.path.join(policy_checkpoint_dir, "Critic")
                if not os.path.isdir(critic_path):
                    raise FileNotFoundError(
                        f"Checkpoint for policy {policy_name!r} is missing critic directory: {critic_path}"
                    )
            policy_paths[policy_name] = (actor_path, critic_path)

        del_local_after_load = bool(self.config.trainer.del_local_ckpt_after_load)
        for policy_name, trainer in policy_trainers.items():
            actor_path, critic_path = policy_paths[policy_name]
            trainer.actor_rollout_wg.load_checkpoint(
                local_path=actor_path,
                del_local_after_load=del_local_after_load,
            )
            if trainer.use_critic:
                trainer.critic_wg.load_checkpoint(
                    local_path=critic_path,
                    del_local_after_load=del_local_after_load,
                )

        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        if os.path.exists(dataloader_path):
            train_dataloader.load_state_dict(
                torch.load(dataloader_path, weights_only=False)
            )
        else:
            logger.warning(
                "No dataloader state found at %s; starting its state from scratch",
                dataloader_path,
            )

        if self.trainer_mode != "sync" and supports_transfer_queue_checkpoint():
            transfer_queue_path = os.path.join(checkpoint_dir, "transfer_queue")
            if os.path.exists(transfer_queue_path):
                logger.info("Loading TransferQueue state from %s", transfer_queue_path)
                tq.load_checkpoint(transfer_queue_path)
        return int(step_text)

    def resolve_checkpoint_dir(self) -> str | None:
        resume_mode = self.config.trainer.resume_mode
        if resume_mode == "disable":
            return None
        if resume_mode == "resume_path":
            checkpoint_dir = self.config.trainer.resume_from_path
            if not checkpoint_dir:
                raise ValueError(
                    "trainer.resume_from_path is required when trainer.resume_mode='resume_path'"
                )
            checkpoint_dir = os.path.abspath(os.path.normpath(str(checkpoint_dir)))
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint directory does not exist: {checkpoint_dir}"
                )
            return checkpoint_dir
        if resume_mode == "auto":
            checkpoint_root = self.config.trainer.default_local_dir
            if not checkpoint_root:
                return None
            checkpoint_root = os.path.abspath(str(checkpoint_root))
            if not os.path.isdir(checkpoint_root):
                return None
            tracker_path = os.path.join(
                checkpoint_root, "latest_checkpointed_iteration.txt"
            )
            if not os.path.isfile(tracker_path):
                logger.info(
                    "No checkpoint tracker found at %s; training from scratch",
                    tracker_path,
                )
                return None
            with open(tracker_path, encoding="utf-8") as file:
                step_text = file.read().strip()
            if not step_text.isdigit():
                raise ValueError(
                    f"Invalid checkpoint step in {tracker_path}: {step_text!r}"
                )
            checkpoint_dir = os.path.join(checkpoint_root, f"global_step_{step_text}")
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint tracker points to a missing directory: {checkpoint_dir}"
                )
            return checkpoint_dir
        raise ValueError(f"Unknown trainer.resume_mode: {resume_mode}")

    def save(
        self,
        *,
        global_step: int,
        policy_trainers: Mapping[str, PolicyTrainer],
        train_dataloader: StatefulDataLoader,
        policy_pool: Executor,
    ) -> None:
        self.validate_save_config()
        checkpoint_root = self.config.trainer.default_local_dir
        if not checkpoint_root:
            raise ValueError(
                "trainer.default_local_dir is required to save a multi-agent checkpoint"
            )
        checkpoint_dir = os.path.join(
            str(checkpoint_root), f"global_step_{global_step}"
        )
        policies_dir = os.path.join(checkpoint_dir, "policies")
        os.makedirs(policies_dir, exist_ok=True)

        default_hdfs_dir = self.config.trainer.default_hdfs_dir
        remove_previous = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated; use max_actor_ckpt_to_keep=1 "
                "and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            1
            if remove_previous
            else self.config.trainer.get("max_actor_ckpt_to_keep", None)
        )
        max_critic_ckpt_to_keep = (
            1
            if remove_previous
            else self.config.trainer.get("max_critic_ckpt_to_keep", None)
        )
        retention_limits = [
            limit
            for limit in (
                max_actor_ckpt_to_keep,
                (
                    max_critic_ckpt_to_keep
                    if any(trainer.use_critic for trainer in policy_trainers.values())
                    else None
                ),
            )
            if isinstance(limit, int) and limit > 0
        ]
        outer_retention_limit = min(retention_limits) if retention_limits else None
        futures = [
            policy_pool.submit(
                self._save_policy_checkpoint,
                global_step=global_step,
                policy_name=policy_name,
                trainer=trainer,
                policies_dir=policies_dir,
                default_hdfs_dir=default_hdfs_dir,
            )
            for policy_name, trainer in policy_trainers.items()
        ]
        wait(futures)
        for future in futures:
            future.result()

        torch.save(
            train_dataloader.state_dict(), os.path.join(checkpoint_dir, "data.pt")
        )

        transfer_queue_saved = (
            self.trainer_mode != "sync" and supports_transfer_queue_checkpoint()
        )
        if transfer_queue_saved:
            tq.save_checkpoint(
                os.path.join(checkpoint_dir, "transfer_queue"),
                metadata={"global_steps": global_step},
            )

        manifest = self._build_manifest(
            global_step=global_step,
            policy_trainers=policy_trainers,
            transfer_queue_saved=transfer_queue_saved,
        )
        self._atomic_write_text(
            os.path.join(checkpoint_dir, "checkpoint_manifest.json"),
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        self._publish_tracker(str(checkpoint_root), global_step)
        try:
            self._prune_committed_checkpoints(
                str(checkpoint_root), outer_retention_limit
            )
        except OSError as exc:
            logger.warning("Failed to prune old committed checkpoints: %s", exc)

    @staticmethod
    def _save_policy_checkpoint(
        *,
        global_step: int,
        policy_name: str,
        trainer: PolicyTrainer,
        policies_dir: str,
        default_hdfs_dir: str | None,
    ) -> None:
        policy_checkpoint_dir = os.path.join(policies_dir, policy_name)
        os.makedirs(policy_checkpoint_dir, exist_ok=True)

        actor_remote_path = (
            None
            if default_hdfs_dir is None
            else posixpath.join(
                str(default_hdfs_dir),
                f"global_step_{global_step}",
                "policies",
                policy_name,
                "actor",
            )
        )
        trainer.actor_rollout_wg.save_checkpoint(
            os.path.join(policy_checkpoint_dir, "actor"),
            actor_remote_path,
            global_step,
            max_ckpt_to_keep=None,
        )

        if trainer.use_critic:
            critic_remote_path = (
                None
                if default_hdfs_dir is None
                else posixpath.join(
                    str(default_hdfs_dir),
                    f"global_step_{global_step}",
                    "policies",
                    policy_name,
                    "Critic",
                )
            )
            trainer.critic_wg.save_checkpoint(
                os.path.join(policy_checkpoint_dir, "Critic"),
                critic_remote_path,
                global_step,
                max_ckpt_to_keep=None,
            )

    def _build_manifest(
        self,
        *,
        global_step: int,
        policy_trainers: Mapping[str, PolicyTrainer],
        transfer_queue_saved: bool,
    ) -> dict[str, Any]:
        policies = {}
        for policy_name, trainer in policy_trainers.items():
            policy_path = f"policies/{policy_name}"
            policies[policy_name] = {
                "actor": {"path": f"{policy_path}/actor", "status": "complete"},
                "critic": (
                    {"path": f"{policy_path}/Critic", "status": "complete"}
                    if trainer.use_critic
                    else None
                ),
            }
        return {
            "schema_version": 1,
            "global_step": global_step,
            "trainer_mode": self.trainer_mode,
            "status": "complete",
            "policies": policies,
            "dataloader": {"path": "data.pt", "status": "complete"},
            "transfer_queue": (
                {"path": "transfer_queue", "status": "complete"}
                if transfer_queue_saved
                else None
            ),
        }

    @staticmethod
    def _atomic_write_text(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.{uuid4().hex}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def _publish_tracker(cls, checkpoint_root: str, global_step: int) -> None:
        cls._atomic_write_text(
            os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt"),
            str(global_step),
        )

    @staticmethod
    def _prune_committed_checkpoints(
        checkpoint_root: str, max_ckpt_to_keep: int | None
    ) -> None:
        if max_ckpt_to_keep is None:
            return

        checkpoint_root = os.path.realpath(checkpoint_root)
        committed_checkpoints: list[tuple[int, str]] = []
        for entry in os.scandir(checkpoint_root):
            if not entry.is_dir(follow_symlinks=False) or not entry.name.startswith(
                "global_step_"
            ):
                continue
            step_text = entry.name.removeprefix("global_step_")
            if not step_text.isdigit():
                continue
            manifest_path = os.path.join(entry.path, "checkpoint_manifest.json")
            try:
                with open(manifest_path, encoding="utf-8") as file:
                    manifest = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            step = int(step_text)
            if (
                manifest.get("status") == "complete"
                and manifest.get("global_step") == step
            ):
                committed_checkpoints.append((step, entry.path))

        committed_checkpoints.sort()
        for _, checkpoint_path in committed_checkpoints[:-max_ckpt_to_keep]:
            resolved_path = os.path.realpath(checkpoint_path)
            if os.path.dirname(resolved_path) != checkpoint_root:
                raise RuntimeError(
                    f"Refusing to prune checkpoint outside root: {resolved_path}"
                )
            shutil.rmtree(resolved_path)

    def _validate_manifest(
        self,
        checkpoint_dir: str,
        policy_trainers: Mapping[str, PolicyTrainer],
    ) -> None:
        manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
        if not os.path.isfile(manifest_path):
            logger.warning(
                "Checkpoint %s has no outer manifest; loading it as a legacy checkpoint",
                checkpoint_dir,
            )
            return

        try:
            with open(manifest_path, encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid checkpoint manifest at {manifest_path}: {exc}"
            ) from exc

        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        expected_step = int(checkpoint_name.removeprefix("global_step_"))
        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                f"Unsupported checkpoint manifest schema at {manifest_path}: "
                f"{manifest.get('schema_version')!r}"
            )
        if manifest.get("status") != "complete":
            raise RuntimeError(
                f"Checkpoint is uncommitted: manifest status is {manifest.get('status')!r} at {manifest_path}"
            )
        if manifest.get("global_step") != expected_step:
            raise RuntimeError(
                f"Checkpoint manifest step {manifest.get('global_step')!r} does not match directory step "
                f"{expected_step}"
            )
        if manifest.get("trainer_mode") != self.trainer_mode:
            raise RuntimeError(
                f"Checkpoint trainer_mode {manifest.get('trainer_mode')!r} does not match current "
                f"trainer_mode {self.trainer_mode!r}"
            )

        manifest_policies = manifest.get("policies")
        if not isinstance(manifest_policies, dict):
            raise RuntimeError(
                f"Checkpoint manifest has no policies mapping: {manifest_path}"
            )
        expected_policy_names = set(policy_trainers)
        if set(manifest_policies) != expected_policy_names:
            raise RuntimeError(
                "Checkpoint policies do not match configured policies: "
                f"manifest={sorted(manifest_policies)}, configured={sorted(expected_policy_names)}"
            )
        for policy_name, trainer in policy_trainers.items():
            policy_manifest = manifest_policies.get(policy_name)
            if not isinstance(policy_manifest, dict):
                raise RuntimeError(
                    f"Checkpoint manifest is missing policy {policy_name!r}"
                )
            expected_components = {"actor": f"policies/{policy_name}/actor"}
            if trainer.use_critic:
                expected_components["critic"] = f"policies/{policy_name}/Critic"
            for component, relative_path in expected_components.items():
                component_manifest = policy_manifest.get(component)
                if (
                    not isinstance(component_manifest, dict)
                    or component_manifest.get("status") != "complete"
                ):
                    raise RuntimeError(
                        f"Checkpoint manifest has incomplete {component} for policy {policy_name!r}"
                    )
                if component_manifest.get("path") != relative_path:
                    raise RuntimeError(
                        f"Checkpoint manifest has invalid {component} path for policy {policy_name!r}: "
                        f"{component_manifest.get('path')!r}"
                    )
                component_path = os.path.join(checkpoint_dir, *relative_path.split("/"))
                if not os.path.isdir(component_path):
                    raise FileNotFoundError(
                        f"Checkpoint for policy {policy_name!r} is missing {component} directory: "
                        f"{component_path}"
                    )
            if not trainer.use_critic and policy_manifest.get("critic") is not None:
                raise RuntimeError(
                    f"Checkpoint manifest contains a critic for policy {policy_name!r}, "
                    "but the current policy has no critic"
                )

        dataloader_manifest = manifest.get("dataloader")
        if (
            not isinstance(dataloader_manifest, dict)
            or dataloader_manifest.get("status") != "complete"
        ):
            raise RuntimeError("Checkpoint manifest has incomplete dataloader state")
        if dataloader_manifest.get("path") != "data.pt":
            raise RuntimeError(
                f"Checkpoint manifest has invalid dataloader path: {dataloader_manifest.get('path')!r}"
            )
        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        if not os.path.isfile(dataloader_path):
            raise FileNotFoundError(
                f"Checkpoint manifest references missing dataloader state: {dataloader_path}"
            )

        transfer_queue_manifest = manifest.get("transfer_queue")
        if (
            self.trainer_mode != "sync"
            and supports_transfer_queue_checkpoint()
            and transfer_queue_manifest is None
        ):
            raise RuntimeError("Checkpoint manifest is missing TransferQueue state")
        if self.trainer_mode != "sync" and transfer_queue_manifest is not None:
            if (
                not isinstance(transfer_queue_manifest, dict)
                or transfer_queue_manifest.get("status") != "complete"
            ):
                raise RuntimeError(
                    "Checkpoint manifest has incomplete TransferQueue state"
                )
            if transfer_queue_manifest.get("path") != "transfer_queue":
                raise RuntimeError(
                    "Checkpoint manifest has invalid TransferQueue path: "
                    f"{transfer_queue_manifest.get('path')!r}"
                )
            transfer_queue_path = os.path.join(checkpoint_dir, "transfer_queue")
            if not os.path.exists(transfer_queue_path):
                raise FileNotFoundError(
                    f"Checkpoint manifest references missing TransferQueue state: {transfer_queue_path}"
                )
