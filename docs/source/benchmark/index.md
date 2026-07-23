# Benchmark

This section collects reproducible reference results for agent inference, verification, and reinforcement learning.

## Results

- [Inference and Verification](inference.md) covers SWE-Bench Verified, SWE-Bench Multilingual, and Terminal-Bench v2.
- [RL Training](rl-training.md) covers Base-to-RL improvements, asynchronous rollout performance, and representative training curves.

## Metric Conventions

- **Avg@N** is the average pass rate across `N` independently sampled rollouts per task.
- **Pass@1** reports one rollout per task.
- **Base** is the model score before Uni-Agent RL training.
- **RL** is the score after training with the listed recipe.

## How to Read the Results

!!! note "Reference results, not a controlled leaderboard"
    The results were produced across different model sizes, context lengths, turn limits, sampling settings, hardware, and code revisions. Compare rows only after checking the full setting.

Inference and training evolve quickly. A complete result should record:

- Model checkpoint and tokenizer.
- Benchmark version, split, and preprocessing revision.
- Agent, Task, Sandbox, and reward configuration.
- Sampling parameters, context length, turn limit, and rollout count.
- Inference engine, tensor parallelism, nodes, and GPUs.
- Uni-Agent and verl commits.
- Exact command, configuration files, and result artifacts.

## Reproduce

Inference entry points live under `examples/inference/`. Quickstart Task Configs and Runtime Env examples live under `examples/quickstart/inference/`.

RL recipes live under `examples/quickstart/training/`. Each published result should link back to a runnable recipe and retain its validation curves and configuration.
