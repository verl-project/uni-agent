# Inference and Verification

Uni-Agent reuses the same Task, Agent, Sandbox, and reward definitions for parallel inference and verification. Each task runs in its own stateful sandbox, and scores are produced by the task's verifier.

## SWE-Bench Verified

| Model | Score | Rollouts | Setting |
| --- | ---: | :---: | --- |
| Qwen3-Coder-30B-A3B-Instruct | **49.2** | Avg@4 | 100 turns, 128K context |
| Qwen3-Coder-480B-A35B-Instruct | **64.2** | Avg@4 | 500 turns, 256K context |
| Qwen3-Coder-Next | **67.6** | Avg@4 | 300 turns, 128K context |
| Qwen3.5-4B | **45.2** | Avg@1 | 100 turns, 64K context |
| Qwen3.5-9B | **53.8** | Avg@1 | 100 turns, 64K context |
| Qwen3.5-9B | **65.6** | Avg@1 | 200 turns, 128K context |
| Qwen3.5-35B-A3B | **68.4** | Avg@1 | 200 turns, 128K context |

The Qwen3-Coder runs use temperature `0.8` and top-p `0.9`. The Qwen3.5 runs use task-specific sampling configurations; consult the associated recipe before comparing rows.

## SWE-Bench Multilingual

| Model | Score | Rollouts | Setting |
| --- | ---: | :---: | --- |
| Qwen3-Coder-30B-A3B-Instruct | **32.3** | Avg@1 | 200 turns, 128K context |

## Terminal-Bench v2

| Model | Score | Rollouts | Setting |
| --- | ---: | :---: | --- |
| Qwen3.6-35B-A3B | **42.53** | Avg@1 | 200K context |

## Run the Evaluation

Prepare the dataset and run either inference path described in [Run Agent Inference](../quickstart/agent-inference.md):

- External API mode for direct endpoint evaluation.
- verl-managed rollout mode for training-path parity and token-level trajectory collection.

SWE-Bench verification is implemented by the Task reward function. The task executes benchmark tests inside the sandbox and reports `resolved`, evaluation status, runtime, and a detailed report.

## Reporting Checklist

When adding a new row, include:

- Exact model identifier.
- Agent implementation or harness.
- Sandbox backend.
- Temperature, top-p, and top-k.
- Max turns and token budget.
- Number of rollouts.
- Benchmark revision and verifier.
- Result JSON or evaluation log.
