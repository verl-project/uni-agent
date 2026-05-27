# Agent Reinforcement Learning

In the previous pages, we focused on inference-time agent behavior: how an agent calls tools, interacts with environments, and solves tasks through multi-turn interaction. A major advantage of Uni-Agent is that the same interaction stack can be connected directly to training engines such as `verl`, so you can move from running agents to training them without any modification.

That is where agent reinforcement learning becomes interesting. Once the agent can already run real rollouts, the next step is to optimize it with large-scale training. This is also where the systems challenge appears: each sample may involve multi-turn reasoning, tool calls, environment interaction, and test execution, and episode latency can vary a lot across tasks.

This page introduces the training scripts under `examples/agent_train`, explains how **agent config** is defined and used, and compares the **synchronous** and **fully asynchronous** training recipes built on top of `verl`.

The training launchers live under `examples/agent_train`.

---

## Two Training Paths

Uni-Agent currently has two agent RL rollout paths.

The default path uses `uni_agent.agent_loop.UniAgentLoop`. This is the path used
by the existing training launchers and agent config YAMLs. It is best when you
want Uni-Agent to own the model, tools, environment, reward, and interaction
loop through `AgentInteraction`.

The gateway framework path uses
`uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter`. It is best
when you already have an OpenAI-compatible agent runner or need per-session
gateway isolation. In that path, the trainer creates gateway sessions, the
agent runner talks to `session.base_url`, and
`OpenAICompatibleAgentFramework.generate_sequences(...)` writes rollout data
back to the `verl` sync trainer.

These paths coexist. A single training run normally chooses one rollout manager:
either the `UniAgentLoop` path through `agent_loop_config_path`, or the gateway
framework path through `actor_rollout_ref.rollout.agent.agent_loop_manager_class`.

---

## Training Overview

The two launcher scripts are:

| Script | Mode| Best for |
|--------|------|----------|
| `train_sync.sh` | Synchronous | First runs, simpler debugging, predictable update rhythm |
| `train_fully_async.sh` | Fully asynchronous | Large-scale runs, better utilization when episode latency is uneven |

Both scripts train an agent policy with the same overall components:

1. Load prompts from a Parquet dataset.
2. Run multi-turn agent rollouts in parallel sandboxes.
3. Compute rewards from the task outcome.
4. Update the policy with GRPO-style training.
5. Periodically evaluate on the validation set.

Both scripts also share the same idea of **agent configuration**:

- The training script sets high-level trainer, rollout, model, and cluster parameters.
- `AGENT_CONFIG_PATH` points to a YAML file that defines the agent loop itself: interaction limits, sandbox config, tools, and reward settings.
- Sample-specific fields from the dataset, such as environment image or reward metadata, are merged in at runtime.

That separation is important: the shell script controls the **training system**, while the YAML controls the **agent interaction behavior inside each rollout**.

---

## Before You Launch

Both scripts are designed to be launched from the repository root so Ray can package both `verl/` and `uni_agent/`.

The common path variables are:

| Variable | Meaning | Default |
|----------|---------|---------|
| `RAY_DATA_HOME` | Root directory for models, data, checkpoints, and runtime env files | `${HOME}/verl` |
| `MODEL_PATH` | Policy model checkpoint | `${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Instruct-xml-template` |
| `CKPTS_DIR` | Output directory for training logs and checkpoints | `${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}` |
| `TRAIN_FILE` | Training dataset in Parquet format | `${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet` |
| `TEST_FILE` | Validation dataset in Parquet format | `${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet` |
| `RUNTIME_ENV` | Ray runtime environment YAML | `${RAY_DATA_HOME}/data/swe_agent/runtime_env.yaml` |
| `AGENT_CONFIG_PATH` | Agent loop config YAML | `examples/agent_interaction/agent_config.yaml` |

Prepare the training and validation datasets first:

```bash
export RAY_DATA_HOME=~/verl

# Training set: r2e-gym-subset
python examples/data_preprocess/r2e_gym_subset_filtered.py \
    --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"

# Validation set: SWE-bench Verified
python examples/data_preprocess/swe_bench_verified.py \
    --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"
```

This writes:

- `${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet` as `TRAIN_FILE`
- `${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet` as `TEST_FILE`

XML tool-call template

If you use Qwen3-30B-A3B-Instruct for agent training, replace the `chat_template` field in `tokenizer_config.json` with the XML version below.
The original template uses JSON-style tool calls, but Uni-Agent expects XML-style tool calls during rollout.

```text
"chat_template": "{% macro render_extra_keys(json_dict, handled_keys) %}\n    {%- if json_dict is mapping %}\n        {%- for json_key in json_dict if json_key not in handled_keys %}\n            {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}\n                {{- '\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}\n            {%- else %}\n                {{-'\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}\n            {%- endif %}\n        {%- endfor %}\n    {%- endif %}\n{% endmacro %}\n\n{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{%- if not tools is defined %}\n    {%- set tools = [] %}\n{%- endif %}\n\n{%- if system_message is defined %}\n    {{- \"<|im_start|>system\\n\" + system_message }}\n{%- else %}\n    {%- if tools is iterable and tools | length > 0 %}\n        {{- \"<|im_start|>system\\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.\" }}\n    {%- endif %}\n{%- endif %}\n{%- if tools is iterable and tools | length > 0 %}\n    {{- \"\\n\\n# Tools\\n\\nYou have access to the following functions:\\n\\n\" }}\n    {{- \"<tools>\" }}\n    {%- for tool in tools %}\n        {%- if tool.function is defined %}\n            {%- set tool = tool.function %}\n        {%- endif %}\n        {{- \"\\n<function>\\n<name>\" ~ tool.name ~ \"</name>\" }}\n        {%- if tool.description is defined %}\n            {{- '\\n<description>' ~ (tool.description | trim) ~ '</description>' }}\n        {%- endif %}\n        {{- '\\n<parameters>' }}\n        {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}\n            {%- for param_name, param_fields in tool.parameters.properties|items %}\n                {{- '\\n<parameter>' }}\n                {{- '\\n<name>' ~ param_name ~ '</name>' }}\n                {%- if param_fields.type is defined %}\n                    {{- '\\n<type>' ~ (param_fields.type | string) ~ '</type>' }}\n                {%- endif %}\n                {%- if param_fields.description is defined %}\n                    {{- '\\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}\n                {%- endif %}\n                {%- set handled_keys = ['name', 'type', 'description'] %}\n                {{- render_extra_keys(param_fields, handled_keys) }}\n                {{- '\\n</parameter>' }}\n            {%- endfor %}\n        {%- endif %}\n        {% set handled_keys = ['type', 'properties'] %}\n        {{- render_extra_keys(tool.parameters, handled_keys) }}\n        {{- '\\n</parameters>' }}\n        {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}\n        {{- render_extra_keys(tool, handled_keys) }}\n        {{- '\\n</function>' }}\n    {%- endfor %}\n    {{- \"\\n</tools>\" }}\n    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n{%- endif %}\n{%- if system_message is defined %}\n    {{- '<|im_end|>\\n' }}\n{%- else %}\n    {%- if tools is iterable and tools | length > 0 %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in loop_messages %}\n    {%- if message.role == \"assistant\" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}\n            {{- '\\n' + message.content | trim + '\\n' }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n            {%- if tool_call.arguments is defined %}\n                {%- for args_name, args_value in tool_call.arguments|items %}\n                    {{- '<parameter=' + args_name + '>\\n' }}\n                    {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}\n                    {{- args_value }}\n                    {{- '\\n</parameter>\\n' }}\n                {%- endfor %}\n            {%- endif %}\n            {{- '</function>\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"user\" or message.role == \"system\" or message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.previtem and loop.previtem.role != \"tool\" %}\n            {{- '<|im_start|>user\\n' }}\n        {%- endif %}\n        {{- '<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>\\n' }}\n        {%- if not loop.last and loop.nextitem.role != \"tool\" %}\n            {{- '<|im_end|>\\n' }}\n        {%- elif loop.last %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- else %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
```

---

## Launch Sync/Async Training

Then launch one of the scripts from the repo root:

```bash
bash examples/agent_train/train_sync.sh
```

or:

```bash
bash examples/agent_train/train_fully_async.sh
```

If you use VEFAAS or other remote environments, make sure the required credentials and deployment settings are already present in `runtime_env.yaml` or the job environment. See the environment setup document for the sandbox-side details.


Agent training uses `actor_rollout_ref.rollout.agent.agent_loop_config_path` to locate the YAML file that defines the agent loop. In the default scripts, this path is:

```bash
examples/agent_interaction/agent_config.yaml
```

At runtime, the config is consumed in layers:

1. The training script passes `agent_loop_config_path` into the rollout config.
2. `uni_agent/agent_loop.py` loads the YAML file and reads the first agent definition from it.
3. The trainer injects rollout-side model objects such as the client, tokenizer, and sampling parameters.
4. Per-sample fields from the dataset, especially `tools_kwargs.env` and `tools_kwargs.reward`, are merged into the config before each rollout starts.

This means the YAML defines the **base agent template**, while the dataset can still customize the sandbox image, setup commands, and reward metadata for each sample.

### Annotated agent config

Below is the default config used by the training scripts, with explanations for each section:

```yaml
# examples/agent_interaction/agent_config.yaml

- name: xxx_agent
  # Agent name. This should match the `agent_name` field in the dataset.
  # Each sample uses this name to select the corresponding agent config.

  _target_: uni_agent.agent_loop.UniAgentLoop
  # Agent loop class. Keep this value unless you are replacing the rollout logic.

  concurrency: 512
  # Global concurrency budget inside the agent loop. The implementation divides
  # this across rollout workers to cap how many environments run at once.

  log_dir: /tmp/swebench_qwen3_coder
  # Base directory for per-run logs such as trajectories and interaction results.

  interaction:
    action_timeout: 300
    # Max time for one tool/action call inside a rollout.

    max_turns: 100
    # Max number of model-environment turns per episode.

  env:
    deployment:
      type: vefaas
      command: curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}
      timeout: 600
    env_variables:
      PIP_PROGRESS_BAR: "off"
      PIP_CACHE_DIR: "~/.cache/pip"
      PAGER: "cat"
      MANPAGER: "cat"
      LESS: "-R"
      TQDM_DISABLE: "1"
      GIT_PAGER: "cat"
  # Sandbox template. The dataset can still override pieces such as image and
  # post-setup commands for each sample.

  tools:
    - name: str_replace_editor
    - name: execute_bash
    - name: submit
  # Tools installed into each sandbox before interaction starts.

  reward:
    eval_timeout: 600
  # Base reward config. Dataset-provided reward metadata is merged into this.
```

### What each top-level section means

| Key | Purpose | Typical tuning advice |
|-----|---------|-----------------------|
| `name` | Logical name of the agent loop entry | Usually keep as is unless you define multiple agent types |
| `_target_` | Python class that implements the loop | Change only when extending the framework |
| `concurrency` | Cap on total simultaneous agent episodes | Raise when your backend can sustain more sandboxes |
| `log_dir` | Directory for run logs and cached rollout outputs | Point to fast, large local storage |
| `interaction` | Turn limit and action timeout | Increase `max_turns` for harder tasks; keep timeouts conservative |
| `env` | Sandbox deployment template and env vars | Match your actual deployment backend and credentials |
| `tools` | Tool list exposed to the model | Keep only tools the task truly needs |
| `reward` | Reward-side settings shared by all samples | Most often `eval_timeout` and reward backend options |

### How this differs from trainer config

It helps to separate three config layers:

| Layer | Controlled by | Examples |
|-------|---------------|----------|
| Training system | `train_sync.sh` / `train_fully_async.sh` | node counts, batch sizes, optimizer, rollout engine, parallelism |
| Agent loop | `AGENT_CONFIG_PATH` YAML | tools, sandbox, interaction limits, reward base config |
| Per-sample task data | Dataset row under `extra_info.tools_kwargs` | container image, repo reset command, reward metadata |

If you want to change **how many GPUs or workers** the run uses, edit the shell script. If you want to change **how the agent behaves inside each environment**, edit the YAML. If you want task-specific sandbox or reward details, edit the dataset generation pipeline.

---

## Gateway Framework Training

Gateway framework training is configured through the `verl` trainer config rather
than an `AGENT_CONFIG_PATH` YAML. The important config entries are:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter
    custom:
      agent_framework:
        agent_runner_fqn: your_package.your_recipe.agent_runner.your_agent_runner
        gateway_count: 8
        tool_config_path: path/to/tool_config.yaml

reward:
  custom_reward_function:
    path: pkg://your_package.your_recipe.reward
    name: compute_score
```

The importable recipe code and runnable DeepEyes example live together under
`examples/agent_train/deepeyes_gateway/`:

```bash
bash examples/agent_train/deepeyes_gateway/run_deepeyes_gateway_grpo.sh
```
