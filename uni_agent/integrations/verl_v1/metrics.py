"""Score interrupted uni-agent validation without changing verl's metric policy."""

from __future__ import annotations

from collections import Counter

from verl.trainer.ppo.metric_utils import process_validation_metrics as summarize_metrics


def process_validation_metrics(data_sources, sample_uids, infos_dict, *, expected_acc_counts=None, seed=42):
    """Count missing scored sessions as zero for this integration's canonical metrics.

    Padding exists only in metric inputs, never in TransferQueue trajectories.
    Preserve observed-only statistics and all noncanonical metrics. A source
    acquires acc/score only when a real observation defines that metric.
    """
    count = len(data_sources)
    if len(sample_uids) != count or any(len(values) != count for values in infos_dict.values()):
        raise ValueError("Validation metric columns must have equal lengths")
    observed = summarize_metrics(data_sources, sample_uids, infos_dict, seed=seed)
    result = {source: {name: dict(values) for name, values in metrics.items()} for source, metrics in observed.items()}
    if not expected_acc_counts:
        return result

    keys = list(zip(data_sources, sample_uids, strict=True))
    actual_counts = Counter(keys)
    for key, actual in actual_counts.items():
        if key not in expected_acc_counts or actual > expected_acc_counts[key]:
            raise ValueError(f"Unexpected validation observation count for {key!r}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in expected_acc_counts.values()
    ):
        raise ValueError("Expected session counts must be positive integers")

    canonical = [name for name in ("reward", "acc", "score") if name in infos_dict]
    defined = {
        name: {source for source, value in zip(data_sources, infos_dict[name], strict=True) if value is not None}
        for name in canonical
    }
    if "reward" in defined:
        defined["reward"].update(source for source, _ in expected_acc_counts)
    padded_sources, padded_uids = list(data_sources), list(sample_uids)
    padded = {
        name: [
            0.0 if value is None and source in defined[name] else value
            for source, value in zip(data_sources, infos_dict[name], strict=True)
        ]
        for name in canonical
    }
    if "pred" in infos_dict:
        padded["pred"] = list(infos_dict["pred"])
    for (source, uid), expected in expected_acc_counts.items():
        for _ in range(expected - actual_counts[(source, uid)]):
            padded_sources.append(source)
            padded_uids.append(uid)
            for name in canonical:
                padded[name].append(0.0 if source in defined[name] else None)
            if "pred" in padded:
                padded["pred"].append(None)

    all_attempts = summarize_metrics(padded_sources, padded_uids, padded, seed=seed)
    for source, metrics in all_attempts.items():
        target = result.setdefault(source, {})
        for name in canonical:
            if name not in metrics:
                continue
            if name in observed.get(source, {}):
                target[f"{name}_success_only"] = dict(observed[source][name])
            target[name] = dict(metrics[name])
    return result
