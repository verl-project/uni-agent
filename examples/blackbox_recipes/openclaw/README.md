# OpenClaw black-box recipe

本 recipe 通过 uni_agent.agents.openclaw.agent.OpenClawAgent 调用固定 OpenClaw 2026.9.2，使用通用 Task、Sandbox 和 typed AgentResult，不复制旧 SessionHandle reward 回传机制。

## 镜像

已发布挂载式 sidecar：

    swr.cn-east-3.myhuaweicloud.com/openyuanrong/openclaw-tool@sha256:90cef4641f82a955692385c48cb4db01a3bd73fc6f68f37962a58717ce547c13

这是 scratch 工具镜像，根目录应挂载至 /opt/openclaw，不能 docker run 直接执行。目标任务镜像需提供 glibc >=2.35、python3、git 及任务依赖。Agent 明确设置 /opt/openclaw/bin PATH；不依赖 runtime 镜像的 ENV。

构建可执行验证镜像：

    DOCKER_BUILDKIT=0 docker build --target runtime -t openclaw-recipe-runtime:2026.9.2 -f examples/blackbox_recipes/openclaw/Dockerfile.openclaw-tool examples/blackbox_recipes/openclaw

构建挂载工具镜像：将上述 --target runtime 改为 --target tool，并设置目标 tag。凭据只通过 password-stdin 注入，不写入本 recipe。

## 配置与任务

config/openclaw_terminal_bench.yaml 使用通用 terminal_bench 任务协议，官方 tests_archive 在 Agent 运行后才注入 sandbox，并由通用 reward 模块执行。Agent finished 与 verifier reward 是不同指标。样本应由 uni_agent.tasks.terminal_bench.preprocess 产生，非仅含 prompt 的任意 JSON。

    CONDA_DEFAULT_ENV=<专用环境> DATA_PATH=<数据.parquet> BASE_URL=<sandbox可访问的v1地址> OUTPUT_DIR=<仓库外新目录> bash examples/blackbox_recipes/openclaw/run_infer.sh

必须实际激活独立 Conda 环境。API_KEY 使用受保护环境变量，不通过 shell 命令字面值传入。文本 Qwen3.5-9B vLLM 服务必须使用 --language-model-only。默认 n=1，不用同题多 rollout 拼接结果。Docker 默认网络不能访问宿主 127.0.0.1，应提供可路由 endpoint 或明确配置网络；host-network 仅用于受控验证，不作为不可信任务的安全隔离承诺。

## 外部数据验证基线

retry30 是当前唯一的单机八卡 rollout 基线。使用外部任务验证时只替换 `TRAIN_DATA`、`VAL_DATA` 和任务产物，保持下面参数不变：

- Qwen3.5-9B，V1 `colocate_async`，单机 `1x8`，训练/rollout TP=8、PP=1、CP=1。
- vLLM 内部 rollout，`language_model_only=True`、prompt/response `4096/4096`、`max_model_len=8192`、`max_num_seqs=1`、`max_num_batched_tokens=8192`、`enforce_eager=True`、cudagraph `NONE`、GPU memory utilization `0.2`。
- 单样本单并发：`N=1`、`GATEWAY_COUNT=1`、`MAX_CONCURRENT_SESSIONS=1`、`NUM_AGENT_WORKERS=8`；`OFFLOAD=True`、`OFFLOAD_FRACTION=1.0`；PPO mini/micro batch 都是 `1`。
- `TRAIN_MAX_SAMPLES=1`、`VAL_MAX_SAMPLES=1`、train/val batch 都是 `1`、`TOTAL_TRAINING_STEPS=1`、`VAL_BEFORE_TRAIN=false`，并使用 `actor_rollout_ref.rollout.checkpoint_engine.backend=naive`。

正式入口仍是 `run_train.sh`，由 verl trainer 在 Ray 内部管理 vLLM；不要启动独立 `vllm serve`。如果默认 Ray 控制面被无关集群占用，只能隔离 Ray 端口/临时目录，不能改变上述模型、rollout、并发、长度、offload 和 batch 参数。验收只看题目完成、verifier/reward、`finished=true` 以及单题单 session 单 trajectory。

## 已验证范围

- 真实 OpenClaw + 脚本化 mock endpoint，写文件答案42、两次请求、一个工具调用。
- 真实 Qwen3.5-9B 单机八卡 V1 `colocate_async` rollout：内部 vLLM TP=8，TerminalBench `openclaw-merge-intervals-v1` verifier `exit=0`、reward=1.0、finished=true，单 session 单 trajectory。
- 通用 DockerSandbox + 主机 Agent 调用、SQLite 单链审计与容器回收。
- 从实际 scratch sidecar 提取文件树、只读挂载 /opt/openclaw 的验证通过；未验证 OpenYuanrong 远端挂载服务。
- SQLite 拒绝分叉、compaction/reset、替代 session、不同 model/run、不完整工具配对；未知事件 fail closed。此审计证明记录的轨迹结构，不保证无未记录 transport retry。
- Agent --local 没有公开 max-turns，agent_max_turns 非空时显式拒绝，不声称限制已生效。

## 尚未完成

公开任务正确率、多样本/异常 sandbox 验证、OpenYuanrong 远端挂载、发布0.1.0rc1兼容矩阵和完整质量检查仍待补充。当前验收以单题 rollout、reward 和单条完整轨迹为准；PPO optimizer update 不属于本轮判定。

## 单题解题验收入口

单题独立推理可使用 launch_vllm.sh（前台运行；默认 GPU0,1/TP2，始终 language-model-only）。训练路径使用 run_train.sh，由 verl trainer 在 Ray V1 `colocate_async` 内部管理 vLLM，不要另起 `vllm serve`。

    python examples/blackbox_recipes/openclaw/dataset.py /outside/task.json
    python examples/blackbox_recipes/openclaw/infer_one.py --task /outside/task.json --base-url http://127.0.0.1:18090/v1 --output /outside/result.json

该题是recipe自建区间合并编程验收，有106项测试，不代表公开Terminal-Bench成绩。verifier在Agent结束后才注入。结果必须同时满足reward=1、finished=true、trajectory_audit.verified=true；轨迹自动保存到结果目录下trajectories。训练parquet使用dataset.py --format parquet；每条prompt放在标准extra_info.tools_kwargs.task结构中。

run_train.sh 默认按单机八卡 `colocate_async` 启动；`RAY_SUBMIT_MODE=local` 用于本机验收，`RAY_SUBMIT_MODE=job` 对接已有 Ray Jobs endpoint。需显式 `TRAIN_DATA`、`VAL_DATA`、`CKPTS_DIR` 并激活专用 Conda。`DRY_RUN=1` 只展开命令，不启动 Ray 或模型。

已验证的训练入口示例（Qwen3.5-9B，单题单样本）：

    export RAY_SUBMIT_MODE=local
    export TRAIN_DATA=/outside/train.parquet
    export VAL_DATA=/outside/val.parquet
    export CKPTS_DIR=/outside/checkpoints
    bash examples/blackbox_recipes/openclaw/run_train.sh \
      actor_rollout_ref.rollout.checkpoint_engine.backend=naive

retry30 运行结果保存在仓库外的 `train_run_20260908_single_retry30`：rollout 已完成题目并得到 reward=1.0；后续 optimizer step 的单机资源错误不影响上述 rollout 验收。

## 发布基线插件兼容验证

使用专用环境的python -I以避免checkout遮蔽已安装包，执行check_release.py的绝对路径及仓库外新目录。该入口要求已安装uni-agent版本0.1.0rc1，仅加载新增OpenClaw插件，Task/Sandbox使用site-packages发布基线。正确候选reward1、错误候选reward0的真实OpenClaw/mock端到端验证已通过；这不是rc1训练/Gateway或真实模型兼容性的证明。
