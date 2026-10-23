# 第三方 Sidecar 构建与接入指南

本文面向要给 Uni-Agent 新增工具的贡献者，例如新增 OpenCode。交付物是工具镜像、验证用例、Agent 适配和任务配置。下文统一使用 `my-tool` 作为镜像 recipe 名、`my_tool` 作为 Agent 注册名；接入 OpenCode 时换成 `opencode`。公共脚本接受自定义 manifest，无需修改脚本来添加工具名称。

## 1. 准备环境和目录

在包含[公共脚本](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/README.md)的源码 checkout 根目录执行命令。准备 Linux x86_64、Python 3.11、可访问的 Docker Engine，以及下载基础镜像和工具依赖的网络。先检查：

```bash
set -euo pipefail
test "$(uname -m)" = x86_64
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
docker info >/dev/null
test -f examples/sidecar/build.py
test -f examples/sidecar/validation.py
```

新增工具的文件放在一个独立目录：

```text
examples/my_tool/
  Dockerfile             # 安装固定版本工具和运行时
  .dockerignore          # 只允许 COPY 所需的文件
  recipes.json           # 构建参数、挂载位置、入口与依赖探针
  run_agent.py           # 将框架输入转换成工具调用，归一化结果
  checks.py              # 需要定制协议时提供的功能检查
  fixtures/              # 检查所需的输入及期望结果
  task_config.yaml       # 挂载与 Agent 配置
  README.md              # 工具版本、额外依赖、运行方式和验证结果
```

构建机运行公共脚本；sidecar 提供工具文件；任务镜像提供 shell、项目目录和工具需要的系统库。检查脚本使用任务镜像中的 `python3`；模型工具可以另外携带自己的 Python、Node 或其他运行时。构建及固定响应检查无需 GPU 或真实模型凭据。

## 2. 定义镜像和 manifest

在新工具目录编写 `recipes.json`，完整格式可复制[最小自定义 manifest](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/sample/recipes.json)。新增工具逐项替换安装、版本、路径和入口；示例中的 BusyBox 只演示文件布局，不属于新增工具的验收步骤。

| 字段 | 约定 |
| --- | --- |
| `schema_version`、`recipes` | 版本为 `1`；`recipes` 是 recipe 名到配置的映射 |
| `context`、`dockerfile` | 相对 manifest 所在目录解析，构建上下文只包含这个工具 |
| `image`、`tag` | 不含 registry 前缀的镜像名和 tag；tag 可引用 `build_args` 中的版本，例如 `{TOOL_VERSION}` |
| `platform` | 当前公共流程支持 `linux/amd64` |
| `mount_target` | 沙箱内的绝对目录，例如 `/opt/opencode` |
| `entrypoint` | argv 数组，例如 `["/opt/opencode/bin/python", "/opt/opencode/bin/run_agent.py"]` |
| `probe_argv` | 检查工具可执行、版本和依赖的 argv；成功返回 `0`，缺依赖或版本不符返回非零 |
| `version_arg` | `build_args` 中主工具版本的键，例如 `TOOL_VERSION` |
| `build_args` | 字符串到字符串的映射，至少包含 `BASE_IMAGE` 和主工具版本 |

真实工具的 Dockerfile 使用固定基础镜像、工具版本和依赖版本，并在构建阶段执行版本或依赖检查。最终层使用 `FROM scratch`，将完整工具目录的**内容**复制到镜像根目录：

```dockerfile
# builder 中已安装工具及完整运行时到 /opt/my-tool。
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

这是 Dockerfile 最终层片段；安装步骤由工具自己的 builder 阶段提供。例如将 `bin/` 放在镜像根目录后，挂载到 `/opt/my-tool` 才会得到 `/opt/my-tool/bin/`。镜像本身无需启动命令，使用公共验证脚本挂载执行。

打包时一并携带解释器、原生二进制、动态库、数据文件和离线启动需要的缓存。处理好可执行权限、shebang、运行时前缀和共享库查找路径；普通 Python venv 应从最终挂载前缀构建，或使用可重定位运行时。构建机的系统库不会自动出现在任务沙箱中，动态链接程序需要兼容的 libc 和加载器；静态二进制没有该动态链接要求。`.dockerignore` 只允许所需源码和入口，凭据文件保存在目录外。

## 3. 构建自己的镜像

以下命令针对刚创建的目录。替换真实工具后，继续使用它自己的 manifest 和 recipe 名：

```bash
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --no-cache --pull
RESOLVED_RECIPE=$(python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe)
LOCAL_SIDECAR_IMAGE=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["image_ref"])')
export LOCAL_SIDECAR_IMAGE
```

`--describe` 输出解析后的配置，`--dry-run` 只打印命令。`--version` 覆盖 `version_arg` 对应版本；`--tag` 覆盖标签；`--base-image` 覆盖基础镜像；`--build-arg KEY=VALUE` 仅覆盖 manifest 已声明的键。公开包源可通过 `--pip-index`、`--npm-registry` 指定，但该 recipe 必须先声明对应参数。

后续命令在同一 shell 按顺序执行。修改版本参数时，构建和 `--describe` 使用同一组参数，以免解析成旧标签。安装目录、挂载目录和入口一起变更。发布可重建镜像时记录源码 SHA、基础镜像 digest、工具版本、完整依赖锁和最终 registry digest。

## 4. 验证工具入口

### 输入与输出确定的工具

通用检查会先执行 `probe_argv`，再执行 `entrypoint`，传入输入文件的原始字节，并逐字节比较 stdout，包括末尾换行：

```bash
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --image "$LOCAL_SIDECAR_IMAGE" \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt
```

最后一行应有 `"smoke":"passed"` 和 `"cleanup":"passed"`，实际 JSON 含空格。脚本导出 sidecar，在新容器内只读挂载并关闭外网，完成后清理临时容器。输入、输出错误或检查失败均返回非零。可用 `--runtime-image "$TASK_IMAGE"` 检查与实际任务镜像的兼容性；该镜像必须提供 `python3`。

### OpenCode 等模型工具

先让模型工具的 wrapper 实现第 7 节的 stdio 契约。Chat Completions 工具可以直接使用公共[模型检查脚本](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/check_model.py)：它启动本地固定响应端点，兼容流式和非流式请求，要求工具真正执行 shell 命令并返回完成状态。无需真实模型凭据：

```bash
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --image "$LOCAL_SIDECAR_IMAGE" \
  --check-script examples/sidecar/check_model.py
```

公共模型检查支持 `bash`、`exec_command`、`shell_command` 工具调用。使用其他协议或工具定义时，在新增工具目录实现 `checks.py`，将 `--check-script` 指向它。该脚本仅依赖标准库，接受 `--case CASE_JSON`。JSON 提供 `recipe`、`mount_target`、`entrypoint`、`probe_argv`；同时传入两个 fixture 参数时，还提供 `stdin_base64`、`expected_stdout_base64`。检查脚本在容器内启动，只能访问该容器的 localhost、任务文件和只读 sidecar。

检查脚本应启动本地固定模型端点，让真实工具执行一条可断言的命令，例如写入 `/testbed/probe.txt`，再断言工具正常完成、文件内容正确且收到预期模型请求。全部成功才输出最终一行 `{"result":"passed"}` 并退出 `0`；异常时返回非零，关闭本地服务并回收子进程。单独运行 `--version` 只能证明启动和依赖，不算模型功能检查。仅在返回码和最终 JSON 都成功时，公共脚本才报告通过。

OpenCode 的非交互入口是 `opencode run --format json`，模型通过 `--model provider/model` 指定；wrapper 要解析该固定版本的事件并识别成功、错误和未完成状态。provider 的 `baseURL` 与 `apiKey` 从本次请求读取，模型名及完整 session 路径原样传递。按所选协议选择 provider 包：Chat Completions 使用 `@ai-sdk/openai-compatible`，Responses 使用 `@ai-sdk/openai`。配置可通过 `OPENCODE_CONFIG_CONTENT` 在运行时传入。接口依据：[CLI](https://opencode.ai/docs/cli/)、[provider](https://opencode.ai/docs/providers/#custom-provider)、[配置](https://opencode.ai/docs/config/)。

## 5. 发布镜像并固定 digest

沙箱服务需要从 registry 拉取镜像。通过组织提供的方式登录仓库，并给新增工具发布独立 tag：

```bash
docker login registry.example.com
export SIDECAR_REGISTRY=registry.example.com/team
SIDECAR_TAG=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag"])')
export SIDECAR_TAG
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --registry "$SIDECAR_REGISTRY" --tag "$SIDECAR_TAG" --push
SIDECAR_NAME=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["image"])')
export SIDECAR_REPOSITORY="$SIDECAR_REGISTRY/$SIDECAR_NAME"
SIDECAR_IMAGE=$(
  docker image inspect "$SIDECAR_REPOSITORY:$SIDECAR_TAG" --format '{{json .RepoDigests}}' |
  python3 -c 'import json,sys; print(next(x for x in json.load(sys.stdin) if x.startswith(sys.argv[1]+"@")))' \
    "$SIDECAR_REPOSITORY"
)
export SIDECAR_IMAGE
```

`--registry` 只指定目标名称，`--push` 才上传。服务端也必须具备该仓库的 pull 权限。构建、版本探针及本地登录成功都不能证明服务端拉取成功。记录完整 `repository@sha256:...`，后续挂载与验收使用此不可变引用。

## 6. 在真实沙箱验证自定义 sidecar

准备可用的 OpenYuanRong 服务、frontend JWT 和任务镜像。任务镜像需要 Linux amd64、Bash、`python3`、pip 及可写的 `/testbed`，并能安装 `swe-rex`。在 Linux 验证机保留完整 checkout、工具目录、manifest 和 fixture，然后配置 SDK：

```bash
SIDECAR_VENV="$HOME/.venvs/sidecar-validation"
if [ ! -e "$SIDECAR_VENV" ] && [ ! -L "$SIDECAR_VENV" ]; then
  python3 -m venv "$SIDECAR_VENV"
fi
. "$SIDECAR_VENV/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
python -m pip install 'openyuanrong-sandbox==0.10.2rc9'
python -c 'import importlib.metadata, yr_sandbox; assert importlib.metadata.version("openyuanrong-sandbox") == "0.10.2rc9"; print(yr_sandbox.__file__)'
```

已有环境先检查再复用。服务凭据放在仓库外的文件中，由编辑器或安全渠道填写；不写进构建参数、镜像、命令参数或版本库：

```bash
install -d -m 700 "$HOME/.config/uni-agent"
touch "$HOME/.config/uni-agent/openyuanrong.env"
chmod 600 "$HOME/.config/uni-agent/openyuanrong.env"
${EDITOR:-vi} "$HOME/.config/uni-agent/openyuanrong.env"
```

文件提供 `AKERNEL_SERVER_ADDRESS='实际服务主机:端口'` 和 `AKERNEL_TOKEN='实际frontend JWT'`。服务地址不带 URL scheme，短 ACL 标识不能替代 JWT。443 端点由脚本设置 `gateway_use_tls=True`；默认验证服务证书，私有 CA 应加入验证机的信任配置。

```bash
set +x
set -a
. "$HOME/.config/uni-agent/openyuanrong.env"
set +a
export TASK_IMAGE=registry.example.com/team/task-runtime:validated
python test_sandbox.py \
  --image "$TASK_IMAGE" \
  --sidecar-recipe my-tool --sidecar-image "$SIDECAR_IMAGE" \
  --recipe-file examples/my_tool/recipes.json \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt \
  --swe-rex-spec 'swe-rex==1.4.0' \
  --report artifacts/my-tool-remote.json
```

模型工具将两个 fixture 参数替换为 `--check-script examples/sidecar/check_model.py`，或该工具自己的检查文件。本地和远端调用同一检查文件及 manifest，挂载目标读取 `mount_target`，不按工具名写死。

通过标准是命令退出 `0`、报告 `success=true`，且创建、命令执行、工具入口、SWE-ReX 安装、8000 端口内外健康与最终清理全部成功，`cleanup.confirmed=true`。报告记录实际 SDK 版本、文件路径、SHA-256 和 CPU/内存请求。默认请求为 `2000m/4096MB`，可通过 `--cpu`、`--memory` 明确调整；容量不足时保留报告，待有可调度节点后使用新的报告路径复验。

## 7. 适配 Agent 和任务配置

Sidecar 挂载验证之后，还需把工具接到 Uni-Agent 的 Agent 接口。提供的[Agent 模板](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/agent_template.py)定义了一个可复用的 stdio 契约：

```json
{
  "messages": [{"role": "user", "content": "任务说明"}],
  "model": {"base_url": "本次模型端点", "api_key": "本次运行凭据", "model_name": "本次模型名"},
  "workdir": "/testbed"
}
```

`run_agent.py` 从 stdin 读取该 JSON，调用本工具的真实 CLI，并向 stdout 输出**一个完整 JSON 对象**。CLI 的原始事件、进度和日志由 wrapper 解析或写入 stderr，stdout 的规范结果是：

```json
{"output": {"content": "最终结果"}, "finished": true}
```

`finished` 必须是布尔值，仅在实际任务完成时为 true；进程成功退出不能代替完成断言。CLI 返回错误、超时或没有完成事件时返回非零或 `finished=false`。OpenCode wrapper 在这里将自身事件转换为此结构；需要任务 patch、submission 等字段时将它们放进 `output`，与任务评分接口对齐。

复制模板并注册新的 Agent：

```bash
mkdir -p uni_agent/agents/my_tool
cp examples/sidecar/agent_template.py uni_agent/agents/my_tool/agent.py
touch uni_agent/agents/my_tool/__init__.py
```

将模板里的注册名和 `ToolConfig.name` 改成新工具的名称，并在 [`AGENT_MODULES`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/agents/registry.py) 现有字典中追加 `"my_tool": "uni_agent.agents.my_tool.agent"`。新增 OpenCode 时，这三个位置均使用 `opencode`。适配器只调用框架传入的 sandbox；沙箱生命周期由任务管理。

把下列配置合并到目标任务的 `task_config.yaml`，填写实际已验证镜像和入口。该文件是按任务名组成的列表，示例针对已有的 `swe_bench` 任务；任务镜像、项目和评估 metadata 仍由该任务的数据配置提供：

```yaml
- name: swe_bench
  sandbox:
    provider: openyuanrong
    image_mounts:
      - mount_path: /opt/my-tool
        image: registry.example.com/team/my-tool-sidecar@sha256:REPLACE_WITH_VERIFIED_DIGEST
    sandbox_kwargs:
      proxy_port: 38197
  agent:
    name: my_tool
    command: [/opt/my-tool/bin/python, /opt/my-tool/bin/run_agent.py]
    run_timeout: 600
```

这是替换为真实模型工具后的配置，最小 BusyBox 示例不实现该模型协议。宿主框架按此配置实例化 Agent，并将模型配置从 `agent.model` 传入 wrapper。通过 [`run_task`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/framework/task_runner.py) 的 `task_config_path` 指向该文件；该入口还会按当前 Gateway session 注入隧道 upstream，保留模型 URL 的完整 `/sessions/<session_id>/v1` 路径。运行使用此 provider 的任务前，令 `OPENYUANRONG_SERVER_ADDRESS`、`OPENYUANRONG_TOKEN` 指向已验证的服务地址及 frontend JWT。

加载任务前先验证新进程中的懒加载注册和配置解析，确认工具名称、入口与挂载目录一致：

```bash
python - <<'PY'
from uni_agent.agents import get_agent_cls
from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import TaskConfigResolver
task = TaskConfigResolver.from_file("examples/my_tool/task_config.yaml").resolve({"name": "swe_bench"})
agent_cls = get_agent_cls(task["agent"]["name"])
config = agent_cls.config_model(**task["agent"])
sandbox_config = SandboxConfig.model_validate(task["sandbox"])
target = sandbox_config.image_mounts[0].mount_path
assert config.command[0].startswith(target + "/")
print("AGENT_REGISTRATION_AND_CONFIG_OK")
PY
export OPENYUANRONG_SERVER_ADDRESS="$AKERNEL_SERVER_ADDRESS"
export OPENYUANRONG_TOKEN="$AKERNEL_TOKEN"
export OPENYUANRONG_TLS=1
export OPENYUANRONG_GATEWAY_TLS=1
export OPENYUANRONG_TLS_VERIFY=1
export OPENYUANRONG_TUNNEL_SSL_VERIFY=1
```

以上连接参数对应 TLS frontend 和 TLS gateway，证书通过验证机信任的 CA 校验；其他部署按服务端配置调整。该检查只加载配置，不执行模型任务。需要的框架依赖按[安装文档](installation.md)准备；只安装 SDK 的环境不足以运行正式任务。先运行目标数据的一条任务，检查：CLI 真正执行、任务改动存在、`AgentResult.finished` 与工具完成状态一致、目标任务的 reward/评估完成，且沙箱被回收；再增加并发或启动训练。

## 8. 提交新增工具前

- 新工具目录包含固定版本构建、manifest、wrapper、功能检查、fixture、任务配置和工具 README。
- 只构建并验证新增工具；本地检查、真实挂载和目标任务的报告使用同一源码及已发布 digest。
- 模型协议、运行时依赖和任务评分字段都有明确映射；错误或未完成不能返回成功。
- 镜像、源码和日志中没有服务凭据；报告保留通过阶段、失败阶段和清理结果。

常见失败：`Exec format error` 检查架构；文件存在却报找不到文件时检查 shebang、动态链接器和共享库；`MANIFEST_UNKNOWN` 检查发布和 digest；模型 404 检查协议与 session 路径；`ERR_RESOURCE_NOT_ENOUGH` 检查同一节点能否同时满足 CPU 与内存请求。
