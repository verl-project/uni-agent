# Build a Simple Arxiv Search Agent

Suppose you want an agent that can search arXiv, read recent paper abstracts, and return a shortlist of the most relevant results.

With Uni-Agent, this is a small customization task rather than a framework rewrite. In this example, we build an arXiv search agent from scratch and use it to search recent papers, read their abstracts, and produce a ranked paper list.

The runnable code lives in `examples/search_arxiv/demo.py`.

We will go through the process in three simple steps:

1. Customize a search tool
2. Launch a model service
3. Run interaction

---

## Step 1: Customize a Search Tool

In this example, we customize the tool `search_arxiv`, defined under `uni_agent/tools/search_arxiv`.

This tool contains two files, and they have different roles:

- `uni_agent/tools/search_arxiv/__init__.py`:
  - This file registers the tool and defines its schema.
  - The schema tells the agent model what the tool does and how to call it.
  - The recommended pattern is to define an `Arguments(BaseModel)` class for the tool parameters and then call `AbstractTool.build_tool_schema(...)` to generate the tool schema.
- `uni_agent/tools/search_arxiv/search_arxiv`:
  - This is the executable script that is copied into the environment and actually runs the search.
  - It is installed into the environment and can be executed directly by the agent.

The main parameters of `search_arxiv` are:

| Parameter | Type | Meaning | Example |
|-----------|------|---------|---------|
| `query` | `string` | Topic or keyword query for searching arXiv papers. | `"Reinforcement Learning"` |
| `max_results` | `integer` | Maximum number of paper candidates returned by the tool. | `5` |
| `days` | `integer` | Recency window in days. Only papers updated within this range are kept. | `30` |

You can validate the tool locally:

```bash
chmod +x uni_agent/tools/search_arxiv/search_arxiv
uni_agent/tools/search_arxiv/search_arxiv --query "Reinforcement Learning" --max_results 5 --days 30
```

This command will print a list of recent arXiv papers, including the title, authors, published time, abstract URL, PDF URL, and abstract text. If these fields are returned correctly, then the tool itself is working as expected.

For example, the output may look like:

```text
[1] Some Recent Paper Title
Authors: Author A, Author B
Published: 2026-03-20T12:34:56Z
Abstract URL: http://arxiv.org/abs/2603.xxxxx
PDF URL: https://arxiv.org/pdf/2603.xxxxx
Abstract: This paper studies ...
```

This tool gives the agent a strong starting point: instead of reasoning from scratch, the model can quickly ground itself on a set of recent papers, read the abstracts, and organize the information into a useful shortlist. In other words, the tool provides retrieval, while the agent provides ranking, filtering, and recommending.

---

## Step 2: Launch a Model Service

In Uni-Agent, the Agent Model is a separate module. This design makes it easy to switch between different model backends without changing the rest of the interaction pipeline.

For example, you can connect the agent to:

- a local serving backend such as vLLM or SGLang
- an internal inference gateway
- any other OpenAI-compatible chat-completions service

For example, you can start a local vLLM service like this:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --enable-auto-tool-choice --tool-call-parser qwen3_coder --tensor-parallel-size 4
```

---

## Step 3: Run Interaction

Once the tool and model service are ready, you can run the full demo directly.

Before running, set the environment-side credentials:

```bash
# deployment env_vars
export DEPLOYMENT=vefaas
export VOLCE_ACCESS_KEY=xxxxxxxxxx
export VOLCE_SECRET_KEY=xxxxxxxxxx
export VEFAAS_FUNCTION_ID=xxxxxxxxxx
export VEFAAS_FUNCTION_ROUTE=xxxxxxxxxx
# model service env_vars
export BASE_URL=http://localhost:8000/v1
export MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct
```

Then run the demo from the repository root:

```bash
DEBUG_MODE=1 python examples/search_arxiv/demo.py
```

Setting `DEBUG_MODE=1` is recommended while developing. It prints the full runtime information to the current terminal, which makes it much easier to inspect environment startup, tool installation, model interaction, and final execution results.

If everything works as expected, you will get something like the following result:

```text
[1/4] Starting environment...
[2/4] Installing tools...
/usr/local/bin/search_arxiv
/usr/local/bin/finish

[3/4] Running interaction...

[4/4] Final status:
exit_reason: finished
done: True

Final result:
Observation:
Based on the arXiv papers from the last month, here are the 5 most relevant papers about 'Agent Reinforcement Learning' with one-sentence reasons:

1. **ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model** (http://arxiv.org/abs/2603.22281v1)
   This paper addresses agent-based prediction by combining dense-frame dynamics modeling with long-horizon semantic guidance, which is crucial for reinforcement learning agents that need to plan and reason about future states.

2. **WorldCache: Content-Aware Caching for Accelerated Video World Models** (http://arxiv.org/abs/2603.22286v1)
   This work enhances world models used in reinforcement learning by improving computational efficiency through intelligent feature caching, directly impacting the training and deployment of agent-based systems.

3. **End-to-End Training for Unified Tokenization and Latent Denoising** (http://arxiv.org/abs/2603.22283v1)
   This research advances latent world models that are fundamental to agent reinforcement learning by enabling unified training of tokenization and generation processes, reducing complexity in agent architectures.

4. **The Dual Mechanisms of Spatial Reasoning in Vision-Language Models** (http://arxiv.org/abs/2603.22278v1)
   While focused on vision-language models, this paper provides insights into spatial reasoning capabilities essential for agents to understand and interact with environments in reinforcement learning settings.

5. **Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels** (http://arxiv.org/abs/2603.22276v1)
   This work improves the efficiency of adapting large models, which is critical for deploying reinforcement learning agents in practical scenarios with limited computational resources.

These papers were selected based on their direct relevance to agent-based reinforcement learning systems, particularly those involving world modeling, spatial reasoning, and efficient model adaptation.
```

After that, it helps to understand what the script is doing internally. The demo mainly combines four parts: the environment, the tool list, the model wrapper, and the interaction loop.

### Environment

The script follows the same environment pattern as `examples/agent_env/demo.py` and uses `AgentEnv`:

```python
env_config = {
    "deployment": {
        "type": "vefaas",
        "image": ".../python:3.12",
        "command": "curl -fsSL ... | bash -s -- {token}",
        "timeout": 300.0,
        "startup_timeout": 180.0,
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    },
}
env = AgentEnv(run_id=run_id, env_config=AgentEnvConfig(**env_config))
```

### Tool List

The tool list tells Uni-Agent which tools should be installed into the environment and exposed to the model:

```python
tools_manager = ToolsManager(
    ToolsManagerConfig(
        tools=[
            ToolConfig(name="search_arxiv"),
            ToolConfig(name="finish"),
        ]
    )
)
```

### Model Wrapper

The model wrapper lets Uni-Agent talk to your external model service:

```python
model = OpenAICompatibleChatModel(
    base_url=model_base_url,
    api_key=model_api_key,
    model_name=model_name,
    sampling_params={"temperature": 0.0, "max_tokens": 8192},
)
model.set_tools_schemas(tools_manager.tools_schemas)
```

### Task And Interaction

The task is defined as a normal chat prompt. In this example, the agent is asked to search recent arXiv papers, read the abstracts, and return a ranked list:

```python
user_request = (
    "Please search arXiv papers from the last month about 'Agent Reinforcement Learning', "
    "read the abstracts, and give me a ranked list of the 5 most relevant papers with "
    "one-sentence reasons. For each paper, include the arXiv abstract URL."
)
```

The system prompt tells the model how to behave in the loop. In this demo, we explicitly require a tool call in every assistant turn:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a simple arXiv paper search agent. "
            "Every assistant response MUST contain EXACTLY ONE tool call. "
            "Do not reply with plain text without a tool call."
        ),
    },
    {"role": "user", "content": user_request},
]
```

Finally, pass everything into `AgentInteraction`:

```python
interaction = AgentInteraction(
    run_id=run_id,
    env=env,
    model=model,
    tools_manager=tools_manager,
    messages=messages,
    action_timeout=60,
    max_turns=20,
)
```

At runtime, the script does four things in order:

1. start the environment
2. install the tools into the environment
3. run the multi-turn interaction loop
4. print the final status and final result

That is the key idea of Uni-Agent in practice: once the tool and model are defined, running the agent is mostly just composition.

If you want to extend this example later, the most common changes are:

1. replace `search_arxiv` with another search backend
2. add richer filtering fields to the tool schema
3. change the prompt to target another search scenario
4. switch to a different OpenAI-compatible model service
5. change the finish format for your own task
