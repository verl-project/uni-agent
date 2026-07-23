# RL Training

Uni-Agent trains agents with the same interaction stack used for inference. Each rollout can launch a sandbox, execute a long-horizon Agent, compute a Task reward, and return a token-level trajectory to verl.

## Reference Results

| Model | Dataset | Method | Setting | Base | RL | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen3-30B-A3B-Instruct | R2E-Gym | GSPO | Fully Async, 100 turns, 128K | 22.2 | **36.8** | +14.6 |
| Qwen3-Coder-30B-A3B-Instruct | R2E-Gym | GSPO | Fully Async, 100 turns, 128K | 46.2 | **52.0** | +5.8 |
| Qwen3.5-9B | SWE-reBench | GRPO | Fully Async, 100 turns, 128K | 53.8 | **59.2** | +5.4 |

The Base and RL columns use the task's validation metric. Consult the linked recipe before comparing results across datasets or models.

## Fully Asynchronous Rollouts

Agent rollouts have highly variable latency because tasks use different numbers of turns, commands, tests, and sandbox operations. Fully asynchronous training allows rollout workers and training workers to progress independently.

The following experiment compares rollout scheduling strategies over 200 training steps on 8 × A100 nodes:

![Fully asynchronous agent training comparison](../assets/async_comp.png){ width="900" }

The partial-rollout setting reduced total training time from 95.6 hours to 45.8 hours, a **2.1× speedup**, while benchmark Pass@1 remained within a narrow range.

## Representative Training Curves

The following plots are representative runs used to inspect reward, validation score, and episode length during training. Summary table values may come from a later or differently configured run.

### Qwen3-30B-A3B-Instruct

This run uses R2E-Gym-Subset for training and SWE-Bench Verified for validation, with 100 turns and a 128K context window.

![Qwen3-30B-A3B-Instruct training curves](../assets/results_qwen3_30b.png){ width="1000" }

### Qwen3.5-9B

This run uses SWE-reBench for training and SWE-Bench Verified for validation, with 100 turns and a 128K context window.

![Qwen3.5-9B training curves](../assets/results_qwen3p5_9b.png){ width="1000" }

## Reproduce

Training recipes live under `examples/quickstart/training/`.

For every published result, retain:

- The exact launch script and Task Config.
- Model, optimizer, and rollout configuration.
- Dataset preprocessing revision.
- Hardware topology and inference engine.
- Reward and validation curves.
- Average turns and rollout completion statistics.
- Uni-Agent and verl commits.

Use [Train an Agent with RL](../quickstart/rl-training.md) as the entry point for running these recipes.
