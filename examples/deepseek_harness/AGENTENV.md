# DeepSeek Harness 接入 AgentEnv

本手册面向运行 SWE-Bench 或 SWE-Rebench rollout 的部署者。AgentEnv 用样本对应的任务镜像启动沙箱，把 DSH 工具镜像只读挂载到 `/opt/dsh`，然后通过 envd 执行 DSH。工具镜像可以独立于任务镜像更新。

## 1. 配置 AgentEnv 和 rollout worker

- 使用包含 AgentEnv provider 的 Uni-Agent 代码，在 rollout worker 的 Python 环境安装 `e2b` extra。从仓库根目录可执行 `pip install -e '.[e2b]'`。
- 部署需提供 E2B 兼容控制面、envd 数据面和 AgentEnv 原生 `attachedDrives`。按样本镜像启动需要 `POST /sandboxes-cold`，按模板启动使用 `POST /sandboxes`，该接口也需要支持 `attachedDrives`。上游 AgentENV 的旧版 `/sandboxes` 接口没有该字段，使用该版本时请走 cold start 挂载。创建后 SDK 会按 sandbox ID 连接沙箱。
- 给执行 rollout 的 worker 注入下列环境变量。Ray 集群通过部署的 worker 环境或作业的 `runtime_env.env_vars` 注入。当前 DSH 启动脚本主动转发的是 `OPENYUANRONG_*`，因此从提交端执行 `export` 无法替代 Ray worker 的 AgentEnv 环境配置。

| 变量 | 用途 |
| --- | --- |
| `AGENTENV_API_URL` | Worker 访问的 AgentEnv 控制面地址 |
| `AGENTENV_SANDBOX_URL` | Worker 访问 envd 数据面的地址 |
| `AGENTENV_API_KEY` | AgentEnv API 凭据，通过部署的密钥注入方式提供 |

AgentEnv 必须能拉取每条样本的任务镜像和 DSH 工具镜像。若任务镜像采用 `<repository>:<instance_id>`，重新生成 parquet，让预处理器把镜像引用直接写入每条样本的 `sandbox.image`。下面的命令适用于两个数据集的实例镜像都发布到同一仓库的部署。若验证集使用独立仓库，在第二条命令单独指定。

```bash
SWE_IMAGE_REPOSITORY=registry.90.91.181.189.nip.io:5000/swerebench-arm64-r
python -m uni_agent.tasks.swe_rebench.preprocess \
  --local-save-dir ~/data/swe_agent_agentenv \
  --image-repository "$SWE_IMAGE_REPOSITORY"
python -m uni_agent.tasks.swe_bench.preprocess \
  --local-save-dir ~/data/swe_agent_agentenv \
  --image-repository "$SWE_IMAGE_REPOSITORY"
```

例如，原始实例 ID `django__django-12345` 会生成 `registry.90.91.181.189.nip.io:5000/swerebench-arm64-r:django__django-12345`。DSH 工具镜像继续通过 `image_mounts[].image` 配置。自行构建工具镜像可参考 [DSH 构建说明](README.md)。

## 2. 选择任务配置和 Gateway 访问方式

使用 [AgentEnv 任务配置](task_config_deepseek_harness_agentenv.yaml)。这份配置对两个任务都设置了 `provider: agentenv`，把镜像根目录挂载到 `/opt/dsh`。挂载后，`executable_paths` 将 `dsh` 和 `node` 链接到 `/usr/bin`。最小工具模式使用挂载内的 `/opt/dsh/minimal_patch.yml`。

配置默认从每个样本的任务镜像 cold start。已有对应 AgentEnv 模板时，可以在该任务的 `sandbox_kwargs.template` 指定模板 ID 或别名。模板优先于样本镜像，每个挂载需要一个可用的 drive slot。整批样本使用不同任务镜像时，继续使用样本驱动的 cold start，或由调用方为每条样本选择对应模板。

示例配置 `sandbox_kwargs.proxy_port: 38197`。`run_task` 保留 Gateway session URL 的协议、主机和路径，将端口替换为 `38197`。DSH 从 guest 直接连接该地址，Gateway 必须在该端口提供服务。若 session URL 自带的端口可直接使用，可以移除 `proxy_port`。

## 3. 验证挂载并启动预跑

如果服务器只有 AgentEnv，直接按 [服务器冒烟验证手册](AGENTENV_SMOKE_TEST.md) 执行。

先在已配置凭据的 worker 环境检查一个可拉取的 SWE 样本镜像。将 `SWE_IMAGE` 设为 AgentEnv 能拉取的完整 OCI 引用。

```bash
python - <<'PY'
import asyncio
import os

from uni_agent.sandbox import SandboxConfig, build_sandbox


async def main():
    sandbox = build_sandbox(SandboxConfig(
        provider="agentenv",
        image=os.environ["SWE_IMAGE"],
        image_mounts=[{
            "image": "registry.90.91.181.189.nip.io:5000/kunpeng-ai/dsh-tool:latest",
            "mount_path": "/opt/dsh",
        }],
        executable_paths={"dsh": "/opt/dsh/bin/dsh", "node": "/opt/dsh/bin/node"},
    ))
    async with sandbox:
        for command in (["node", "--version"], ["dsh", "--version"],
                        ["test", "-r", "/opt/dsh/minimal_patch.yml"]):
            result = await sandbox.exec(command, timeout=20)
            if result.exit_code:
                raise RuntimeError(f"{command}: {result.stderr}")
            print(result.stdout.strip())


asyncio.run(main())
PY
```

确认工具能运行后，沿用 [DSH 示例的数据准备和模型配置](README.md)，用小样本做一次预跑。启动时选用 AgentEnv 配置。

```bash
TASK_CONFIG=examples/deepseek_harness/task_config_deepseek_harness_agentenv.yaml \
TRAIN_DATA="$HOME/data/swe_agent_agentenv/swe_rebench_filtered.parquet" \
VAL_DATA="$HOME/data/swe_agent_agentenv/swe_bench_verified.parquet" \
MODEL_PATH=/path/to/model \
bash examples/deepseek_harness/run_train.sh
```

在 `AGENT_LOG_DIR` 对应的 session 日志中检查 DSH 启动、首个模型响应和任务结束状态。确认流式响应在请求结束前到达，且对应的 Gateway session 能收到完整请求。

## 4. 接入现有流量分配

建议先用单一 AgentEnv provider 完成预跑。启用 `provider: routes` 时，每个 `routes[].config` 都是一份完整的子沙箱配置，`image`、`image_map`、`image_mounts` 和 `executable_paths` 放在各自的子配置里。SWE 样本提供的 `sandbox.image` 会由任务配置解析器自动传给需要镜像的子路由，每个子路由再应用自己的 `image_map`。子路由已指定的镜像、E2B/AgentEnv 模板以及本地路由配置优先。

使用包含 AgentEnv 的混合路由时，外层配置应省略 `proxy_port`，让各候选 guest 使用同一个可直连的 Gateway session URL。按 Gateway session ID 固定路由的行为由 runner 提供，启动阶段的瞬时故障可按路由配置回退。详见 [沙箱流量分配配置](../../docs/source/quickstart/launch-sandbox.md#traffic-splitting-between-sandbox-providers)。

## 常见故障

| 现象 | 检查项 |
| --- | --- |
| 缺少 AgentEnv 连接配置 | 在真正执行 rollout 的进程中检查三个环境变量是否已注入 |
| 创建请求返回 401 | 检查控制面地址和 API key 是否属于同一 AgentEnv 部署 |
| 创建镜像挂载失败 | 检查任务镜像和工具镜像的仓库访问权限、部署的原生挂载能力，以及 warm start 的 drive slot |
| cold start 超时 | 查看根镜像和工具镜像的拉取、转换日志，以及 worker 的 `SANDBOX_STARTUP_TIMEOUT` |
| `dsh` 或 `node` 启动失败 | 检查工具镜像架构，以及挂载目录下的 `bin/dsh`、`bin/node` 和运行库 |
| Gateway 直连失败 | 从 guest 检查 Gateway 主机、端口和网络访问策略 |
