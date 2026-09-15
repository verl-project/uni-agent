# Hermes Agent black-box recipe

This recipe runs the real NousResearch Hermes loop inside the OpenYuanRong SWE-bench
task sandbox. Uni-Agent owns the gateway session, sandbox lifecycle, reward verifier
and trajectory materialization; the sidecar owns only the Hermes runtime and its
`terminal`/`file` tools.

首版范围是单题 SWE-bench Verified rollout：`N=1`、`CONCURRENCY=1`、一个 gateway
session，关闭 memory、delegation、skills、context compression 和 micro-compaction。
`finished` 是 Hermes episode 是否明确完成，题目是否解决仍只由同一 sandbox 的
SWE-bench verifier 决定。训练更新、多题吞吐和子代理不在首版条件内。

## Fixed versions

- Uni-Agent: `verl-project/uni-agent` main at
  `a6e8b08e1d58aa145bc5a22df659f2a00281b8b4`
- Hermes: `NousResearch/hermes-agent` at
  `5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04` (`0.21.2`)
- Tool image default: `swr.cn-east-3.myhuaweicloud.com/openyuanrong/hermes-agent-tool:20260915`
- OpenYuanRong service compatibility: legacy `akernel_sdk==0.9.22` /
  `openyuanrong-sdk==0.7.61rc1` through the isolated Python 3.11 helper;
  `openyuanrong-sandbox==0.10.3` remains recorded for the current API.

实际镜像 digest、verl gitlink checkout、模型路径和运行结果写入每次 run 的
`manifest.json`；不要把 registry auth 文件放进 build context 或 artifact。

## Build the sidecar

在目标 checkout 根目录执行。默认只 build，不 push：

```bash
bash examples/blackbox_recipes/hermes/build_tool.sh \
  --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/
```

若 OpenYuanRong 服务需要从 SWR 拉取，显式指定已授权的 namespace：

```bash
export DOCKER_CONFIG=/home/zxh/.config/hermesrecipe/docker
bash examples/blackbox_recipes/hermes/build_tool.sh \
  --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

构建结果是 `FROM scratch` filesystem payload，挂载到 sandbox 的 `/opt/hermes`，
不是独立任务容器。runner 用 `/opt/hermes/bin/python`，Hermes terminal 通过
`PATH` 使用 `/opt/miniconda3/envs/testbed`；两套 Python 不应混用。

正式 OpenYuanRong 运行环境在项目 Python 中安装
`openyuanrong-sandbox==0.10.3`（提供 `yr_sandbox`）；launcher 会在提交作业前检查该
依赖。目标服务若仍提供旧 legacy API，则 provider 会通过 `/home/zxh/miniconda3/envs/uni-agent`
的 Python 3.11 helper 调用 `akernel_sdk`，不会把旧 SDK 混进 Python 3.12 的 Ray/vLLM
进程。Ray Job API server 也必须已可用，或通过 Ray 的地址环境变量指向目标集群。

## One-shot inference

准备好 remote186 的 OpenYuanRong 凭据、预处理 parquet 和 Qwen3.5-9B 权重后，旧实验
采用的两个变量应保存在权限为 `600` 的 operator-owned env 文件中，例如
`/home/zxh/.config/uni-agent/openyuanrong.env`，不要写入命令行、Ray runtime-env JSON
或 artifact。推荐这样加载后启动：

```bash
set -a
. /home/zxh/.config/uni-agent/openyuanrong.env
set +a
export OPENYUANRONG_ENV_FILE=/home/zxh/.config/uni-agent/openyuanrong.env
export MODEL_PATH=/home/zxh/models/Qwen3.5-9B
export DATA_PATH=/home/zxh/data/swe_agent/swe_bench_verified.parquet
export SAMPLE_INDEX=0
bash examples/blackbox_recipes/hermes/run_infer_hermes.sh
```

如果没有受保护的 env 文件，再由已授权操作者在当前 shell 中设置
`OPENYUANRONG_SERVER_ADDRESS` 和 `OPENYUANRONG_TOKEN`；launcher 会将它们留在进程
内存中，并让 Ray Job driver 在作业内通过受保护文件 wrapper 重新加载，避免出现在
提交命令和 runtime-env JSON 中。

也可以指定 `INSTANCE_ID`，脚本会在源 parquet 中精确选择这一行；未指定时使用
固定的 `SAMPLE_INDEX`。脚本计算 recipe 根目录为 `examples/blackbox_recipes/hermes`
向上三级，因此不要从旧的 two-level launcher 路径复制。

每次运行使用 `/home/zxh/hermesrecipe/artifacts/<run_id>`（可由 `OUTPUT_DIR`
覆盖），并等待 Ray job 结束后执行 `check_acceptance.py`。成功必须同时满足：

- `num_prompts=1`、`num_scored_sessions=1`、`scores=[1.0]`；
- 一个 task log、一个 `trajectory.json` 且 `num_trajectories=1`；
- trajectory 的 `finished=True`、reward 为 `1.0`，token/mask/logprob 数组对齐；
- task log 中 verifier 为 `resolved=True`，Hermes runner 结果为 `completed`。

`TOOL_PARSER` 默认是 `qwen3_coder`，这是与具体 Qwen tokenizer/chat template
绑定的可调整参数；在 W1/W4 的真实多轮 tool-call round trip 前不要把它当作已经
通过的结论。Hermes 产品名不会自动决定 gateway parser。

## Artifacts

```text
<OUTPUT_DIR>/
  manifest.json
  resolved_config.yaml
  command.txt                 # credentials redacted
  prepared_sample.parquet
  sample_metadata.json
  logs/<session>/task.log
  logs/<session>/framework.log
  logs/<session>/trajectory.json
  logs/<session>/trajectory.npz
  hermes/<run-id>/{result,messages,runner}.json/log
  result.json
  acceptance.json
  handoff.md
```

`messages.json` 是 Hermes 诊断快照；训练轨迹只认 Uni-Agent gateway/framework
写出的 `trajectory.json`/`.npz`，不把 sidecar 对话文件当作训练轨迹。

## Validation status

代码级 adapter/runner tests、W1 受控 endpoint、W2 sidecar runtime smoke、remote186
的本地 Docker compatibility run 和正式 OpenYuanRong run 均已产生对应 artifact。
本地 Docker run
使用真实 Qwen3.5-9B TP8 gateway 与真实 Hermes loop，在
`/home/zxh/hermesrecipe/runtime/local-docker-qwen35-compat-v8/` 通过验收器 17/17：
`reward=1.0`、`finished=true`、单 session、单 trajectory、logprob 数组对齐。
该 run 使用本地 Docker task-image fallback 来验证 Uni-Agent/Hermes/framework 链路；
正式 OpenYuanRong 已在
`/home/zxh/hermesrecipe/artifacts/formal-hermes-20260915-191500-legacy8/` 通过验收器
17/17：真实 legacy provider、Qwen3.5-9B TP8 gateway、Hermes 72 次模型调用、
SWE-bench `resolved=True`、`reward=1.0`、`finished=true`、单 session/trajectory，
且 response/mask/logprob 数组长度一致。其目标工具镜像为
`swr.cn-east-3.myhuaweicloud.com/openyuanrong/hermes-agent-tool:20260915`，digest
记录在 run manifest；registry auth 和原始凭据不进入仓库或 artifact。
