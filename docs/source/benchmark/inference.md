# Inference and Verification

Uni-Agent reuses the same Task, Agent, Sandbox, and reward definitions for parallel inference and verification. Each task runs in its own stateful sandbox, and scores are produced by the task's verifier.

## SWE-Bench Verified

| Agent | Model | Rollouts | Setting | Score |
| --- | --- | :---: | --- | ---: |
| ReAct | Qwen3-Coder-30B-A3B-Instruct | Avg@4 | 100 turns, 128K context | **49.2** |
| ReAct | Qwen3-Coder-480B-A35B-Instruct | Avg@4 | 500 turns, 256K context | **64.2** |
| ReAct | Qwen3-Coder-Next | Avg@4 | 300 turns, 128K context | **67.6** |
| ReAct | Qwen3.5-4B | Avg@1 | 100 turns, 64K context | **45.2** |
| ReAct | Qwen3.5-9B | Avg@1 | 100 turns, 64K context | **53.8** |
| ReAct | Qwen3.5-9B | Avg@1 | 200 turns, 128K context | **63.8** |
| Claude Code | Qwen3.5-9B | Avg@1 | 200 turns, 128K context | **51.0** |
| ReAct | Qwen3.5-35B-A3B | Avg@1 | 200 turns, 128K context | **68.4** |

The Qwen3-Coder runs use temperature `0.8` and top-p `0.9`. The Qwen3.5 runs use task-specific sampling configurations; consult the associated recipe before comparing rows. An Agent value of `—` means the original result summary did not record the Agent implementation.

## SWE-Bench Multilingual

| Agent | Model | Rollouts | Setting | Score |
| --- | --- | :---: | --- | ---: |
| ReAct | Qwen3-Coder-30B-A3B-Instruct | Avg@1 | 200 turns, 128K context | **35.0** |

## Terminal-Bench

Harbor is listed as a task format, not an Agent implementation.

| Version | Task Format | Agent | Model | Rollouts | Setting | Score |
| --- | --- | --- | --- | :---: | --- | ---: |
| v2.0 | Native | ReAct | Qwen3.6-35B-A3B | Avg@1 | 256K context | **42.5** |
| v2.1 | Harbor | Claude Code | GLM5.2-733B | — | 256K context | **67.4** |

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
