"""veRL replay-buffer adapters for exporting prompt metrics exactly once."""

from __future__ import annotations

from collections.abc import Mapping

from uni_agent.metrics.prompt import PROMPT_METRICS_EXPORT_OWNER_FIELD, TRAINER_METRICS_EXPORT_OWNER
from uni_agent.metrics.trainer import aggregate_prompt_metrics_for_tracking
from verl.trainer.ppo.v1 import replay_buffer as _verl_replay_buffer


class PromptMetricsReplayBufferMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        export_config = self.sampler_kwargs.get("agent_metrics", {}) or {}
        if not isinstance(export_config, Mapping):
            raise TypeError("sampler_kwargs.agent_metrics must be a mapping")
        mode = export_config.get("mode", "off")
        if mode not in {"off", "shadow", "primary"}:
            raise ValueError(f"Unknown agent metrics export mode: {mode!r}")
        self._agent_metrics_export_enabled = mode == "primary"
        self._pending_agent_metrics: dict[str, int | float] = {}
        self._last_agent_metrics_incomplete_reasons: tuple[str, ...] = ()
        self._selected_prompt_tags: dict[str, Mapping] = {}
        self._selected_metrics_uids: tuple[str, ...] = ()

    @property
    def last_agent_metrics_incomplete_reasons(self) -> tuple[str, ...]:
        return self._last_agent_metrics_incomplete_reasons

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        selection = super()._select_prompt_uids(partition_id, sampleable_keys, batch_size)
        selected_prompt_uids = selection[0]
        if self._agent_metrics_export_enabled:
            partitions = _verl_replay_buffer.tq.kv_list() or {}
            partition_tags = partitions.get(partition_id, {})
            if not isinstance(partition_tags, Mapping):
                partition_tags = {}
            self._selected_prompt_tags = {
                uid: tag
                for uid in selected_prompt_uids
                if isinstance((tag := partition_tags.get(uid)), Mapping)
                and tag.get(PROMPT_METRICS_EXPORT_OWNER_FIELD) == TRAINER_METRICS_EXPORT_OWNER
            }
            self._selected_metrics_uids = tuple(self._selected_prompt_tags)
        return selection

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        if self._agent_metrics_export_enabled and self._selected_metrics_uids:
            export = aggregate_prompt_metrics_for_tracking(
                self._selected_prompt_tags,
                self._selected_metrics_uids,
                partition_id=partition_id,
            )
            self._pending_agent_metrics = export.metrics
            self._last_agent_metrics_incomplete_reasons = export.incomplete_reasons
        return super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        self._pending_agent_metrics = {}
        self._last_agent_metrics_incomplete_reasons = ()
        self._selected_prompt_tags = {}
        self._selected_metrics_uids = ()
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        collisions = self._pending_agent_metrics.keys() & metrics.keys()
        if collisions:
            raise RuntimeError(f"Agent metrics collide with trainer metrics: {sorted(collisions)}")
        return batch, {**metrics, **self._pending_agent_metrics}


class PromptMetricsReplayBuffer(PromptMetricsReplayBufferMixin, _verl_replay_buffer.ReplayBuffer):
    """Sync replay buffer with primary-mode prompt metrics export."""


class PromptMetricsReplayBufferAsync(PromptMetricsReplayBufferMixin, _verl_replay_buffer.ReplayBufferAsync):
    """Async replay buffer with primary-mode prompt metrics export."""
