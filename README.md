# Uni-Agent: Build and Train Agents at Scale

[![Docs](https://img.shields.io/badge/docs-Read%20the%20Docs-8A2BE2)](https://uni-agent.readthedocs.io/en/latest/index.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

Uni-Agent is a unified framework for long-horizon, tool-using agents. Bring existing agent harnesses into reinforcement learning, build transparent agents with composable tools and sandboxes, and generate training-grade trajectories at scale.

The same task, agent, sandbox, and reward definition can be reused across parallel evaluation and RL training with [verl](https://github.com/verl-project/verl).

## Highlights ✨

### Bring existing agents into RL

Connect agent harnesses such as Claude Code and Mini-SWE-Agent, or any harness that can redirect its OpenAI, or Anthropic-compatible model endpoint—to the Uni-Agent Gateway. Uni-Agent captures token-level trajectories and task rewards without requiring you to rewrite the agent loop.

### Build transparent and extensible agents

Build white-box agents by composing reusable `Agent`, `Tool`, `Task`, and `Sandbox` abstractions. Start with built-in coding agents and tools, register your own tools and environments, and extend the same framework to code, search, GUI, and other interactive scenarios.

### Generate training-grade trajectories at scale

Run 1000+ long-horizon, stateful agent sessions with distributed workers, pooled gateway sessions, isolated sandboxes, and asynchronous scheduling. Uni-Agent keeps each session's trajectory and reward correctly associated while reusing the same execution stack across evaluation and RL training.

## Quickstart 🚀

Choose the path that matches your goal:

- **Explore the execution stack:** run the [sandbox and tools demo](./examples/agent_env/README.md).
- **Evaluate with an existing model endpoint:** use [`parallel_infer_api.py`](./examples/agent_interaction/parallel_infer_api.py).
- **Evaluate through the training rollout stack:** use [`parallel_infer_verl.py`](./examples/agent_interaction/parallel_infer_verl.py).
- **Train an agent with RL:** start from the recipes in [`examples/agent_train`](./examples/agent_train).

## Architecture 🧩

<img src="./assets/uni-agent.png" width="100%" alt="Uni-Agent architecture overview">

Uni-Agent separates rollout orchestration from agent and environment execution, with the **Uni-Agent Gateway** acting as the boundary between agent runtimes and the RL system:

1. **Framework Workers** receive prompts and start parallel sessions through the Gateway.
2. A **Task Runner** launches the selected agent—such as Claude Code, Mini-SWE-Agent, or ReAct—against a configurable sandbox backend.
3. The agent sends model requests through its session-scoped Gateway endpoint. The Gateway records the interaction as a token-level trajectory while the task computes its reward.
4. When the session ends, the trajectory and reward are written to the trajectory pool for evaluation or RL training.

For black-box agents, integration can be as small as redirecting the model endpoint to the Gateway. For white-box agents, Uni-Agent provides composable agents, tools, tasks, and sandboxes that can be customized independently.

## Installation 📦

Uni-Agent builds on top of latest `verl` release and can use it as a normal Python package.

```bash
git submodule update --init --recursive
pip install --no-deps -e ./verl

# Other Dependencies
pip install swe-rex loguru pydantic pydantic_settings aiohttp
```

See the full installation guide in the docs: [Installation](https://uni-agent.readthedocs.io/en/latest/start/installation.html).


## Live Dashboard 👀

<img src="./assets/dashboard.png" width="100%" alt="Uni-Agent Dashboard overview">

Uni-Agent includes a lightweight dashboard for monitoring large parallel runs in real time. It is designed for workloads such as parallel inference and reinforcement learning.

Start the dashboard from the repository root:

```bash
python -m dashboard.server --log-dir /tmp/swebench_qwen3_coder --port 8765
```

See [`dashboard/README.md`](./dashboard/README.md) for more details.


## Results 📊

### Parallel Inference & Verification

We compare Uni-Agent with existing agent systems on parallel inference and verification workloads.


| Model            | Benchmark          | OpenHands | Uni-Agent | Setting |
| ---------------- | ------------------ |:---------:|:---------:| ------- |
| Qwen3-Coder-30B  | SWE-Bench Verified | -         | **49.2**  | Avg@4, 100 turns, 128K |
| Qwen3-Coder-480B | SWE-Bench Verified | 62.4      | **64.2**  | Avg@4, 500 turns, 256K |
| Qwen3-Coder-Next | SWE-Bench Verified | 66.6      | **67.6**  | Avg@4, 300 turns, 128K |
| Qwen3.5-35B-A3B  | SWE-Bench Verified | 62.0      | **68.4**  | Avg@1, 200 turns, 128K |
| Qwen3.6-35B-A3B  | Terminal-Bench v2  | -         | **42.5**  | Avg@1, 200K |


### Agent Reinforcement Learning

Uni-Agent supports agent RL training with the same interaction stack used at inference time. We provide fully async training recipes across multiple tasks, models and datasets, with GRPO/GSPO-style objectives and partial rollout support.
Example scripts are available in [examples/agent_train](examples/agent_train).


| Model                        | Dataset      | Method | Setting | Base | RL |
| ---------------------------- | ------------ | ------ | ------- |:----:|:--:|
| Qwen3-30B-A3B-Instruct       | R2E-Gym      | GSPO   | Fully Async, 100 turns, 128K | 22.2 | **36.8** |
| Qwen3-Coder-30B-A3B-Instruct | R2E-Gym      | GSPO   | Fully Async, 100 turns, 128K | 46.2 | **52.0** |
| Qwen3.5-9B                   | SWE-reBench  | GRPO   | Fully Async, 100 turns, 128K | 53.8 | **59.2** |

More training dynamics, including reward, validation score, and average-turn curves, are available in the [agent training guide](https://uni-agent.readthedocs.io/en/latest/start/agent_train.html).



## Roadmap 🗺️

The roadmap below highlights the next major directions for Uni-Agent.

**Environment Support**

- [x] Local deployment support.
- [x] Modal deployment support.
- [ ] More cloud deployment backends (e.g., Yuanrong Sandbox Management System).

**Tool and Task Support**

- [ ] GUI tool support.
- [x] Integration of Skills.
- [ ] More built-in tools and task patterns.

**Model Support**

- [ ] DeepSeek model support.
- [ ] Multimodal model support.

**Agent Integration**

- [x] Black-box integration of additional third-party agents (Ref: [RFC #5790](https://github.com/verl-project/verl/issues/5790)).

**Performance Optimization**

- [ ] Optimize Agentic RL rollout performance (Ref: [Issue #6383](https://github.com/verl-project/verl/issues/6383)).

## Acknowledgement 🙏

Uni-Agent's large-scale parallel interaction and verification rely on remote sandbox backends. We gratefully acknowledge:

- **[veFaaS](https://www.volcengine.com/product/vefaas)**: Volcengine Function-as-a-Service, used as a serverless backend for elastically launching agent sandboxes at scale.
- **[Modal](https://modal.com)**: serverless cloud compute used to spin up isolated, reproducible sandbox environments for agent execution and evaluation.

## Citation 📚

If you find the project helpful, please cite:

```
@misc{uniagent_github,
  author       = {Yuyang Ding and Bo Wen and Xubo Cao and Zhiqiang Zhai and Guangming Sheng and Xibin Wu and Juntao Li and Min Zhang and Uni-Agent Contributors},
  title        = {Uni-Agent: Build, Run, and Train Agents at Scale},
  year         = {2026},
  howpublished = {\url{https://github.com/verl-project/uni-agent}},
  note         = {GitHub repository. Supervisor: Xibin Wu and Juntao Li},
  urldate      = {2026-03-27}
}
```

## Contributing 🤝

Community contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to get started.
