# 在服务器验证 AgentEnv 上的 DeepSeek Harness

用一个真实 SWE-Rebench 实例验证任务镜像启动、DSH 只读挂载和 guest 内命令执行。第一阶段只需要 AgentEnv。第二阶段需要独立模型端点，验证 DSH 能调用模型并实际执行工具。两阶段都可以脱离 rollout 环境运行。

## 执行前准备

- 在能访问 AgentEnv 的服务器上执行。该服务器上的 `aenv` 已配置认证，部署支持 `--cold` 和 `--image-mount-ro`。
- 选一个已发布到 registry 的 SWE-Rebench `instance_id`。镜像 tag 用原始 ID，例如保留 `__`。
- 如果要跑第二阶段，准备从 guest 可达的 DeepSeek 兼容 API Base URL、服务公布的模型 ID，以及需要时的 API key。

## 第一阶段 验证冷启动和 DSH 挂载

在服务器 shell 中执行，记录最后输出的 sandbox ID。

```bash
INSTANCE_ID='<换成实际 instance_id>'
SWE_IMAGE="registry.90.91.181.189.nip.io:5000/swerebench-arm64-r:${INSTANCE_ID}"
DSH_IMAGE='registry.90.91.181.189.nip.io:5000/kunpeng-ai/dsh-tool:latest'

aenv start --cold --detach --timeout 1800 \
  --image-mount-ro "/opt/dsh=${DSH_IMAGE}" \
  "$SWE_IMAGE"
```

将输出的 ID 填到下面的命令。命令通过 envd 在新沙箱中执行，任意检查失败都会返回非零状态。

```bash
SANDBOX_ID='<上一步输出的 sandbox ID>'

aenv exec "$SANDBOX_ID" -- bash -ec '
  cd /testbed
  git rev-parse HEAD
  test -r /opt/dsh/minimal_patch.yml
  export PATH="/opt/dsh/bin:$PATH"
  node --version
  dsh --version
  /opt/miniconda3/envs/testbed/bin/python -c "import sys; print(sys.executable)"
  echo AGENTENV_DSH_STARTUP_OK
'
```

看到 `AGENTENV_DSH_STARTUP_OK` 就确认了两层镜像可拉取、DSH 可以启动、SWE 仓库和任务 Python 环境可用。这里通过 `PATH` 使用挂载目录的可执行文件。PR 的训练配置通过 `executable_paths` 暴露同一组文件。

## 第二阶段 验证模型调用和工具执行

如果暂无模型端点，可以直接跳到清理步骤。若有端点，从服务器进入刚创建的 guest。

```bash
aenv connect "$SANDBOX_ID"
```

以下命令在 **guest 的 Bash** 中执行。填入 guest 可访问的 API Base URL 与模型 ID。如果端点接受空凭据，在提示处直接回车。

```bash
export DEEPSEEK_BASE_URL='<guest 可达的模型 API Base URL>'
export DSH_MODEL_ID='<服务公布的模型 ID>'
read -rsp 'Model API key: ' DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-not-needed}"
```

在同一个 guest shell 运行以下检查。DSH 使用最小工具配置和指定模型，将 `/testbed` 当前 commit 写入临时文件。脚本比对文件内容和真实 commit。

```bash
(
  set -eu
  cd /testbed
  export PATH="/opt/dsh/bin:/opt/miniconda3/envs/testbed/bin:$PATH"
  export DSH_HOME="$(mktemp -d /tmp/dsh-smoke.XXXXXX)"
  export IS_SANDBOX=1 DISABLE_AUTOUPDATER=1
  export DSH_PERMISSION_MODE=danger-full-access

  MODEL_PATCH="$DSH_HOME/model.yml"
  python - "$MODEL_PATCH" <<'PY'
import json
import os
import sys
from pathlib import Path

model = os.environ["DSH_MODEL_ID"]
patch = [
    {
        "id": "llm-deepseek",
        "config": {
            "maxTokens": 1024,
            "models": [{"id": model, "name": model, "contextWindow": 8192, "maxTokens": 1024}],
        },
    },
    {
        "id": "agent-default-model",
        "config": {"provider": "deepseek-official", "model": model},
    },
]
Path(sys.argv[1]).write_text(json.dumps(patch, indent=2) + "\n")
PY

  proof="$DSH_HOME/git-head"
  dsh --profile headless \
    --patch /opt/dsh/minimal_patch.yml \
    --patch "$MODEL_PATCH" \
    "Use the bash tool to run git rev-parse HEAD in /testbed and redirect its output to $proof. Do not modify the repository."
  git rev-parse HEAD | cmp -s - "$proof"
  echo AGENTENV_DSH_TOOL_EXEC_OK
)
```

看到 `AGENTENV_DSH_TOOL_EXEC_OK` 就确认 DSH 完成了模型调用并实际执行了 Bash 工具。这是一条工具执行冒烟用例。验证真实 SWE 修复时，使用同一实例的 `problem_statement`，再运行该实例的测试来评估修复结果。验证 Uni-Agent 的路由、Gateway 直连和训练集成时，使用 [AgentEnv 接入手册](AGENTENV.md) 中的 rollout 流程。

## 故障定位

| 现象 | 先检查 |
| --- | --- |
| `aenv start` 失败 | 核对实例 tag、AgentEnv 服务端到 registry 的连通性和拉取权限，以及服务端是否支持原生镜像挂载 |
| `/testbed` 不存在或 Git 检查失败 | 核对任务镜像及 tag 是否对应所选 SWE 实例 |
| `node` 报 `Exec format error` | 核对 DSH 工具镜像内的 Node 二进制是否为 ARM64 |
| DSH 报模型端点不可达、401 或 404 | 从 guest 检查 Base URL，核对认证和模型 ID |
| DSH 退出后没有 `AGENTENV_DSH_TOOL_EXEC_OK` | 查 DSH 输出中的工具调用、权限错误和临时文件结果 |

## 清理沙箱

退出 guest 后，在服务器执行：

```bash
aenv delete "$SANDBOX_ID"
```
