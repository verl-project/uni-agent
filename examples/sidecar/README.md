# Sidecar 公共脚本

用于新增第三方工具的构建、功能检查和真实沙箱挂载。完整步骤见[第三方 Sidecar 构建与接入指南](../../docs/source/quickstart/sidecar-contribution.md)。在源码 checkout 根目录的 Linux x86_64 环境运行，构建机使用 Python 3.11 与 Docker。

| 文件 | 用途 |
| --- | --- |
| [build.py](build.py) | 根据自定义 manifest 构建镜像，覆盖版本、基础镜像及公开包源，显式发布 |
| [smoke.py](smoke.py) | 在新容器中只读挂载工具、关闭外网，执行功能检查并清理资源 |
| [check_entrypoint.py](check_entrypoint.py) | 执行 manifest 中的依赖探针和入口，验证 stdin/stdout 字节契约 |
| [check_model.py](check_model.py) | 通过本地 Chat Completions 固定响应检查真实工具调用和 stdio wrapper 完成状态 |
| [validation.py](validation.py) | 本地与远端共享的 manifest、fixture 和自定义检查装配 |
| [test_sandbox.py](../../test_sandbox.py) | 使用同一工具检查完成真实创建、挂载、入口、SWE-ReX、端口健康和清理验收 |
| [sample/](sample/recipes.json) | 可复制的最小自定义 recipe，演示版本、路径、manifest 和挂载契约 |
| [agent_template.py](agent_template.py) | 将自定义 stdio 工具接到框架 Agent 的可复制模板 |

## 构建与验证自己的工具

准备 `examples/my_tool/recipes.json` 和该工具所需文件后：

```bash
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --no-cache --pull
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json --image "$LOCAL_SIDECAR_IMAGE" \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt
```

`LOCAL_SIDECAR_IMAGE` 使用构建完成时输出的镜像名。通用检查会执行 `probe_argv` 和 `entrypoint`，同时要求退出码、期望输出和清理成功。模型工具将两个 fixture 参数替换为 `--check-script examples/sidecar/check_model.py`；需要其他协议时传入自己的检查脚本。脚本接受 `--case CASE_JSON`，全部断言通过后输出 `{"result":"passed"}`，详细契约见指南。

`--version`、`--tag`、`--base-image` 与已声明的 `--build-arg KEY=VALUE` 覆盖构建配置；`--dry-run` 只打印命令。发布时使用 `--registry HOST/NAMESPACE --push`，随后记录 registry digest。

真实验收同样接受 `--recipe-file`、fixture 或 `--check-script`。`--sidecar-recipe` 是该 manifest 中的自定义 recipe 名，挂载位置由 `mount_target` 决定。SDK、凭据准备和完整命令见指南。
