# AgentEnv 常驻与卸载恢复 Demo

本目录用于在同一个 AgentEnv 部署内比较常驻沙箱与运行态 pause/resume。后端源码固定为 `a27b2db909b035f58696ba8798ea0e285c7bfa17`，第一版只验证单节点、同节点恢复。

本指南给出已执行 Demo 的复现方式与结果边界。实验记录、原始数据与调研文档不随本目录提交，保存在 `zzzzzzzxh/uni-agent` 的 `feat/agentenv-offload-restore` 分支提交 `c6b652f` 中，下文相关链接均指向该提交。完整结论见 [结果报告](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/docs/sandbox_offload_results.md)，环境和过程证据见 [实验记录](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/docs/sandbox_offload_experiments.md)，指标定义见 [实验规格](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/docs/sandbox_offload_demo_plan.md)。组件测试、节点 `/health` 成功和 API 返回成功仍须与实际运行态恢复、物理进程退出和资源审计区分。

当前 remote186 的 Conda Python 3.11 环境已有 **47 项后端组件测试通过**，覆盖状态探针、宿主采样、编排保护与分析口径，另有本地 6 项绘图测试；它们与下面的真实 AgentEnv 实验分别记录。最新 [64 MiB 预恢复结果](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/docs/sandbox_offload_64m_results.md) 保留了成功条件与完整成本，原来的按需恢复边界仍作为反证保留。

## 已验证的主场景与边界

正式主组 `benefit-bound1-m8-w30-20261021T154549Z` 使用 8 MiB 低熵探针、`dirty-ratio=0`、30 秒合成等待、每任务 2 周期、3 对任务及 `bound1` 池配置。8 MiB 是额外触碰的探针内存，沙箱规格仍为 1 vCPU / 1024 MiB。每任务至少采样 90 秒，实际分析共同窗按清理完成加 I/O 尾段确定，为约 70.08–70.17 秒。

| 指标 | 该组实测 |
|---|---:|
| 正确完成且清理确认 | 6/6 |
| 归属进程 PSS 时间积分 | 降低 56.542% |
| 任务总耗时 p95 | 增加 2.495% |
| 配对周期额外首工具耗时的任务 p95 | 0.870608 秒 |
| 内层节点 vda 平均写入 | 常驻 13.207 → offload 132.725 MiB/任务 |

证据：[正式报告](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/artifacts/agentenv-offload-results/benefit-bound1-m8-w30-20261021T154549Z/report.md)、[原始分析汇总](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/artifacts/agentenv-offload-results/benefit-bound1-m8-w30-20261021T154549Z/summary.json)。PSS 包含已归属的服务、存储服务、运行沙箱与预热池后代进程；它证明该合成负载的进程驻留成本降低，并伴随明显写盘代价。3 对任务的分位数为小样本观测，不能作为生产 SLA 或外层宿主物理 RAM 节省保证。

同设置的 64 MiB 边界组只有 1 对任务：状态和清理均正确，但任务耗时增加约 5.132%，配对额外首工具指标为 1.798241 秒，未达到预设延迟门槛；一次 PSS 缺测使该组内存收益保持不可验证。[边界报告](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/artifacts/agentenv-offload-results/boundary-bound1-m64-w30-20261021T155607Z/report.md)

## 显式预恢复调度

默认 `--resume-lead-seconds 0`，工具请求到达后恢复。参数大于零时，在固定合成等待的预计结束前开始恢复和完整 RAM oracle；工具副作用仍须等到原定请求时刻之后才执行。创建、恢复、校验、提前占回的进程内存及 I/O 均保留在整任务与资源曲线中，首工具计时起点也不移动。

例如对 64 MiB、30 秒已知等待，提前 1.5 秒恢复：

```bash
bash /home/zxh/uni-agent-sandbox-offload/code/scripts/agentenv_offload/run_case.sh \
  lead64-recheck both 30 64 2 3 90 --resume-lead-seconds 1.5
```

这项策略减少请求后的等待，没有缩短底层 VM 恢复本身。它依赖可提前获知的工具到达期限；本 Demo 的固定等待提供了这个前提，未证明真实模型预测误差、突发工具调用或预测错误时仍能满足 1 秒门槛。提前量必须不大于等待时间，默认不开启。没有可靠期限时继续使用按需恢复；不能把理想预测的数据当成通用效果。

正式数据中的 `recovery_to_oracle_seconds` 保留实际恢复加全 RAM 校验耗时，`restore_oracle_seconds` 是原定工具时刻后仍需等待的部分。`resume_lead_seconds`、实际恢复开始、oracle 就绪时刻和固定请求期限均写入原始事件与结果。

## 拓扑与测量范围

```text
remote186：Python 3.11 外部编排器、结果文件
    │ SSH 隧道访问 API；SSH 调用节点上的 aenv
    ▼
remote145：专属 QEMU VM 的外层宿主
    ▼
agentenv-offload-node：Ubuntu 专用 VM
    ├── AgentEnv、ublk/OverlayBD、宿主采样器
    └── Firecracker 沙箱：Python 状态探针，HTTP 8000
```

remote186 使用 `/home/zxh` 和 Conda `uni-agent` Python 3.11。AgentEnv 与原生 `aenv` CLI 在专用 Ubuntu 节点运行，避免客户端旧系统的 libc/内核成为部署依赖。节点内部要求 Linux 6.8+、可访问 `/dev/kvm`、ublk 驱动及匹配的固定版本依赖；不能仅因外层宿主有 KVM 就假定嵌套节点已经可用。

内存有三个独立边界：remote186 客户端、AgentEnv 内层节点、remote145 外层 QEMU。采样器输出内层进程 PSS、匿名页、页缓存和内层设备计数。它们不能直接证明 remote145 已回收相同数量的物理 RAM；外层缺少观测时，分析报告保留“节点物理收益未验证”。

## 程序与归属

| 文件 | 用途与边界 |
|---|---|
| `prepare_vm.sh` | 在外层宿主准备专用 VM；只用指定目录的镜像、磁盘、PID 与专用密钥，不清理共享 VM |
| `provision_node.sh` | 仅在 hostname 为 `agentenv-offload-node` 的专用 x86_64 节点上部署；固定源码、构建依赖和 systemd 服务 |
| `run_case.sh` | 在 remote186 使用已核实的部署与隧道运行单组实验，保存配置前后摘要、部署记录和源码并调用分析器 |
| `demo.py` | HTTP 生命周期与外部等待编排，SSH 执行同版本 `aenv exec`，启动采样和探针，保存每个已开始任务及清理状态 |
| `guest_probe.py` | 真正写入匿名内存，保存随机 nonce、打开文件偏移和文件/计数状态；采集接口不改变逻辑 oracle |
| `host_sampler.py` | 根据 PID 加 starttime 身份采集归属进程和节点指标，不以进程名粗略认领共享进程 |
| `analyze.py` | 读取原始输出，计算任务成功率、延迟、共同窗口内存积分和每设备 I/O，缺测保持不可验证 |

`prepare_vm.sh` 检查既有 `vm.pid` 是否仍对应指定目录的 `system.qcow2`，身份不符会停止；它保留已有磁盘，不自动重置环境。默认 SSH 转发端口为 18422，API 转发为外层 loopback 18800。SSH 端口实际监听地址见脚本，访问控制由该专用部署的网络配置承担。停止 VM 时应先核实 PID、启动时间与磁盘归属，不能根据进程名批量终止。

`provision_node.sh` 会在专用节点安装软件、配置设备权限并创建 `agentenv-offload.service`；禁止把它直接用于 remote145 或 remote186 的共享宿主环境。已有源码目录非 Git 或存在未提交改动时脚本停止，不覆盖其内容。固定的节点目录为：

- 源码：`/opt/agentenv-offload/source`。
- 配置：`/opt/agentenv-offload/config.toml`。
- 持久状态：`/opt/agentenv-offload/state`。
- Firecracker serial：`/opt/agentenv-offload/state/serial`。
- 运行目录：`/run/agentenv-offload`。
- CLI：`/usr/local/bin/aenv`。

早期 canary 使用关闭池的配置；本轮正式主组使用 **`AENV_POOL_PRESET=bound1`**：`[pool]` 的 low/high watermark 均为 1，Firecracker pool 的 enabled、maintenance_enabled、startup_prewarm 均开启，fill_concurrency 为 1；block pool 关闭，network pool maintenance 关闭。A/B 使用同一配置，池和共享缓存的成本也进入采样。允许存在已核实的一个预热 Firecracker；清理验收检查本任务所属 PID 和资源，不要求全机 Firecracker 数量归零。

恢复不是 `snapshot create` 后另起一个沙箱：编排器使用 `/sandboxes/{id}/pause` 和 `/v2/sandboxes/{id}/connect`，验证原 Firecracker 退出，再检查恢复后的 RAM/进程/文件 oracle。普通 pause 产物属于本节点 persister，不验证跨节点迁移。

## 部署与客户端准备

下面的命令是模板。先把所需文件按仓库相对目录复制到远端实验目录，并核实 VM 基础镜像的官方摘要；参数中的示例路径必须指向本轮专用资源。

在 remote145 上准备专用 VM：

```bash
bash /home/zxh/agentenv-offload-deployment/prepare_vm.sh \
  --root /home/zxh/agentenv-offload-deployment \
  --image /home/zxh/agentenv-offload-deployment/ubuntu-noble.img \
  --ssh-port 18422 --api-port 18800 --memory-mib 32768 --cpus 8
```

在新建或已停止服务的专用 VM 内，以本轮实际 Rust/Cargo **1.99.1** 和 `bound1` 配置部署：

```bash
AENV_RUST_TOOLCHAIN=1.99.1 AENV_POOL_PRESET=bound1 bash /root/provision_node.sh
```

部署脚本在已有服务活动时拒绝覆盖配置；复用当前节点时先核对现有部署记录，不重复执行安装。脚本默认池预设为 `off`，因此复现正式组必须显式选择 `bound1`，仅在客户端设置环境变量不会改变已经运行的服务。

部署后保留 `binary-sha256.txt`、`node-verification.json`、配置摘要、源码 SHA、基础 VM 镜像摘要与 VM 启动参数。`demo.py` manifest 中的 `required_agentenv_sha` 是要求值，实际运行版本仍须由部署记录及 `/nodes` 核验，不能把该常量当作实测版本。

在 remote186 核实客户端解释器：

```bash
cd /home/zxh
/home/zxh/miniconda3/envs/uni-agent/bin/python --version
```

客户端使用 stdlib，无需安装其他沙箱 SDK。首次 smoke 不安装 `swe-rex`；后续真实工具负载可在同一 AgentEnv 后端单独验证。

在 remote186 的权限受限 `node-access` 目录准备以下文件；这些文件不进入仓库或实验输出：

- SSH config 与专用私钥：配置能进入专用 VM 的 `agentenv-node` alias；它连接 remote145 转发到专用 VM 的 SSH 端口 18422。核实服务端 host key，不在命令里关闭校验。
- API 凭据 TOML：仅包含 `url` 和 `api_key`，当前用户所有、权限 `0600`。`url` 指向客户端 loopback 隧道地址；凭据通过受限传输或文件操作写入，不把值放入命令参数。
- VM 内 `/root/.config/aenv/credentials`：由部署脚本写入，权限 `0600`。原生 CLI 自行读取，不通过 SSH argv 传 token。

客户端 API 必须通过已有的安全隧道或可信 HTTPS 入口。一个与默认端口匹配的隧道模板为：

```bash
ssh -F /home/zxh/uni-agent-sandbox-offload/node-access/ssh_config \
  -S none -o ControlMaster=no -o ControlPath=none -o ControlPersist=no \
  -o ExitOnForwardFailure=yes -N -T \
  -L 127.0.0.1:18800:127.0.0.1:8000 agentenv-node
```

API 隧道使用独立 SSH 连接，避免复用连接的 ControlPersist 空闲到期后转发端口消失；普通节点命令仍可用复用连接降低开销。隧道需要在实验期间保持运行，开始每轮前验证 loopback 监听与 `/health`；若已经有对应隧道则复用，不重复占用端口。不要使用 `set -x`、打印 credentials、导出完整环境或把 API 响应原样写日志。平台 API、envd 与应用代理使用不同凭据：8000 状态服务通过 `e2b-traffic-access-token` 访问私有 ingress，不能用平台 API key 或 envd token 替代。

下载代理与 API 隧道是独立通路。本轮停用 17899 下载代理后，缓存独立验证出现新的 create HTTP 500，原因仍在调查；当前不能声称无需外网、无需代理或镜像已缓存就可离线复现。开始新一轮前应核实实际镜像/依赖获取路径，保留下载代理状态及失败证据，不能将失败结果改写为成功。

## 动态进程身份文件

`--provider-pid-file` 指向 **VM 节点上的** JSON 文件，不是 remote186 的本地文件。它登记专用 AgentEnv server 与 ublk daemon 的当前 PID 和 `/proc/PID/stat` 第 22 字段 `starttime_ticks`。只给 PID 不足以防止复用；服务重启后必须重新生成。

下面模板在专用节点内按已固定的可执行文件路径识别服务，并通过原子替换更新 registry；不打印命令行、环境或凭据。必须至少发现 server 和 ublk daemon，否则停止实验准备。

```bash
ssh -F /home/zxh/uni-agent-sandbox-offload/node-access/ssh_config \
  agentenv-node 'python3 -' <<'PY'
import json
import os
from pathlib import Path

expected = {
    '/opt/agentenv-offload/source/target/release/server': 'provider',
    '/opt/agentenv-offload/source/target/release/uvm-ublk-daemon': 'storage',
}
processes = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        label = expected.get(os.readlink(entry / 'exe'))
        if label is None:
            continue
        fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            continue
        processes.append({'pid': int(entry.name), 'starttime_ticks': int(fields[19]),
                          'label': label, 'include_descendants': True})
    except (OSError, ValueError, IndexError):
        continue
if {item['label'] for item in processes} != {'provider', 'storage'}:
    raise SystemExit('server/storage identities are incomplete')
target = Path('/opt/agentenv-offload/provider-pids.json')
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps({'processes': processes}) + '\n')
os.chmod(temporary, 0o600)
temporary.replace(target)
PY
```

这是专用 VM 的识别规则，不能复制到存在其他相同路径服务的共享节点。provider/storage 根必须跟踪后代；编排器将这两类根的 `include_descendants` 统一设为 `true`，使 create/connect 返回之前已经运行的 VM 与转换进程也进入总 PSS，不能把尚未登记 sandbox 标签解释为未占内存。

每个沙箱的 Firecracker 由编排器另行登记：进程 executable 必须为 Firecracker，其 fd/1 精确指向本次 `sandboxID` 的 serial 日志。显式 sandbox 根覆盖同一 PID/starttime 的归属标签，采样器按身份唯一计数，不重复累加。恢复后更新 PID/starttime；API state 只是粗粒度状态，不能替代物理进程退出证据。

## 使用 run_case.sh 简便复现

完成上述部署、隧道、私有配置和动态 PID 身份准备后，在 remote186 调用已有包装脚本。其真实参数顺序是：

```text
run_case.sh LABEL MODE WAIT_SECONDS MEMORY_MIB CYCLES PAIRS OBSERVATION_SECONDS [demo options]
```

复现正式主组：

```bash
cd /home/zxh
bash /home/zxh/uni-agent-sandbox-offload/code/scripts/agentenv_offload/run_case.sh \
  benefit-bound1-m8-w30 both 30 8 2 3 90
```

复现 64 MiB 探索性边界组：

```bash
bash /home/zxh/uni-agent-sandbox-offload/code/scripts/agentenv_offload/run_case.sh \
  boundary-bound1-m64-w30 both 30 64 2 1 90
```

脚本使用同一固定镜像 digest、1024 MiB 沙箱规格、低熵、dirty-ratio 0、seed 1 和 100 ms 采样；额外 demo 选项放在七个位置参数之后。`LABEL` 只接受小写字母、数字及连字符。每次生成新的 UTC 时间戳输出目录，不覆盖上述正式数据。

包装脚本通过 `case.lock` 阻止重叠运行，创建前检查 API 隧道，读取节点 `/opt/agentenv-offload/pool-variant-bound1.json`，比较运行前后的配置 SHA-256，保存日志及 `source/run_case.sh`，并调用本轮归档的分析器。它不会自动调整池配置或重建下载代理；该部署记录必须存在并与当前节点实际配置一致。默认根目录是 `/home/zxh/uni-agent-sandbox-offload`、Python 是 `/home/zxh/miniconda3/envs/uni-agent/bin/python`，可通过非敏感的 `AGENTENV_EXPERIMENT_ROOT`、`AGENTENV_CLIENT_PYTHON` 路径变量覆盖。

完整参数需要调整时使用以下直接调用模板。

## 完整运行模板

以下 shell 变量只有路径和非敏感参数。`GUEST_IMAGE` 已填写本轮解析并校验的 Linux amd64 digest；解析记录保留原 tag `python:3.11-slim-bookworm`。需要更新镜像时，先重新核验并记录摘要，不能使用浮动 tag 冒充固定版本。

```bash
cd /home/zxh
PYTHON=/home/zxh/miniconda3/envs/uni-agent/bin/python
EXPERIMENT_ROOT=/home/zxh/uni-agent-sandbox-offload
SCRIPT_DIR="$EXPERIMENT_ROOT/code/scripts/agentenv_offload"
ACCESS_DIR="$EXPERIMENT_ROOT/node-access"
GUEST_IMAGE='docker.io/library/python@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199'
RUN_DIR="$EXPERIMENT_ROOT/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)"

"$PYTHON" "$SCRIPT_DIR/demo.py" \
  --api-url http://127.0.0.1:18800 \
  --credentials-file "$ACCESS_DIR/api-credentials.toml" \
  --node-ssh agentenv-node \
  --ssh-config "$ACCESS_DIR/ssh_config" \
  --node-root /opt/agentenv-offload/runs \
  --aenv-bin /usr/local/bin/aenv \
  --serial-dir /opt/agentenv-offload/state/serial \
  --state-dir /opt/agentenv-offload/state \
  --runtime-dir /run/agentenv-offload \
  --provider-pid-file /opt/agentenv-offload/provider-pids.json \
  --guest-image "$GUEST_IMAGE" \
  --output "$RUN_DIR" \
  --mode both --wait-seconds 2 --memory-mib 64 --sandbox-memory-mib 1024 \
  --cycles 1 --pairs 1 --entropy low --dirty-ratio 0 --seed 1 --cpus 1 \
  --operation-timeout 300 --sample-interval 0.1 \
  --io-tail-seconds 2 --observation-seconds 45
```

输出目录必须不存在，避免覆盖既有实验。`--api-url` 可省略，此时使用客户端 credentials TOML 中的 `url`。`--state-dir` 与 `--runtime-dir` 必须匹配实际部署；示例显式使用 `/run/agentenv-offload`，不要误用程序的通用默认 `/run/aenv`。

启动时将本次 `demo.py`、`guest_probe.py`、`host_sampler.py` 及存在的 `analyze.py` 归档到输出目录的 `source/`，manifest 的 `source_sha256` 对应归档字节。guest 探针和节点采样器也从该归档部署，使运行中或运行后的源码修改不改变本轮所用版本。归档只包含上述固定文件，不包含 credentials、上级目录或其他源码。

外部模拟推理计时在 pause 前开始，暂停写出可以与等待重叠；若暂停超过等待，超出部分计入工具等待。每个周期确认没有在途或结果未知命令才允许 pause。autoResume 关闭；TTL 有界并刷新，`autoPause:false` 的含义是 TTL 到期删除，不是禁用 TTL。

第一版 `aenv exec` 负责启动原生 guest 进程；周期工具负载是私有 HTTP 8000 探针的文件写入、计数和 FD 推进。恢复指标覆盖控制面恢复、SSH 归属检查、全 RAM 摘要触页及首个 HTTP 工具完成。它 **不是原生 shell 命令延迟 benchmark，也不是真实模型或完整 agent 的吞吐 benchmark**。恢复期间的 SSH 观测开销单列在周期结果中，不能全部归因于 VM 恢复。

## 从 smoke 到性能与边界

| 阶段 | 建议控制 | 进入下一阶段的条件 |
|---|---|---|
| 短 smoke | 上述 64 MiB、2 秒、1 周期、1 对 | RAM nonce/摘要、进程身份、FD、文件和计数正确；原 VM 退出；恢复可用；清理确认 |
| 独立 warmup | 保持镜像与资源，单独输出目录，少量完整周期 | 镜像转换、基础依赖和缓存状态已记录；warmup 不混入正式性能分母 |
| 配对性能 | 已测主组为 8 MiB、30 秒、2 周期、3 对；两组顺序由固定 seed 随机 | 预登记的观察覆盖较慢任务至清理/I/O 收尾；扩大重复数时另开运行，不剔除失败 |
| 等待边界 | 固定内存、脏页和并发，只改变等待 1/5/15/30/60 秒 | 找到恢复开销覆盖收益的区间 |
| 内存/脏页边界 | 固定已测等待，只改变内存规模，或另轮只改变 dirty-ratio/entropy | 检查快照写入和首触页成本，不同时调整多个因素 |
| 缓存边界 | 记录受控暖/冷条件 | 不在共享宿主执行全局 drop-caches；无法控制就标未知 |

已测主组证明了特定条件下的收益，不代表参数已经最优。单实例脚本不提供并发恢复风暴验证；不能把它的结果外推到多沙箱容量。

`--observation-seconds` 是从 measurement start 起的最短采样窗口。正式 A/B 应在 warmup 后预先选取足够的共同窗口；较慢任务超出窗口时，保留数据并扩大下一轮窗口，不外推较短一侧。采样会覆盖创建前、任务执行、清理和 I/O 尾段；跨客户端/节点时钟用单次往返校准及其误差界记录。

编排器在首个任务失败、状态损坏、结果未知或清理未确认后停止后续 trial。已经开始的失败任务仍写入 `results.jsonl` 并留在分母；未开始的任务不能算成功或失败。先定位最小失败，再新开输出目录复验，不同步扩大重试。

## 独立验证暂停态删除

`--pause-before-cleanup` 用于验证暂停产物的删除路径，默认关闭，且只允许与 `--mode offload` 一起使用。为它建立独立输出目录：采用上面的完整命令，将模式改为 `offload`、周期/配对数均设为 1，并增加该选项；例如参数尾部为：

```text
--mode offload --wait-seconds 2 --memory-mib 64 --sandbox-memory-mib 1024
--cycles 1 --pairs 1 --pause-before-cleanup
```

正常工具周期结束后，程序再明确 pause 并确认旧 Firecracker 退出，然后在 `finally` 中调用原生 DELETE。结果标记 `validation_only=true`，`final_cleanup_state` 应为 `paused`，`delete_requested` 事件记录实际 `from_state`。只有 API/PID 检查和 owned persister/managed-snapshot artifact、设备、网络审计均通过才标记完整清理成功；暂停产物仍存在时保持 pending。此运行额外增加了一次 pause，不能混入常驻/offload 性能对照。

## 分析与验收

分析器的当前真实入口为目录位置参数：

```bash
"$PYTHON" "$RUN_DIR/source/analyze.py" "$RUN_DIR" \
  --memory-saving-min 0.20 \
  --task-p95-increase-max 0.05 \
  --first-tool-extra-max 1.0 \
  --max-gap 1.0
```

这些是预先选择的建议门槛，不是 SLA。分析会写 `summary.json` 和中文 `report.md`；`run_summary.json` 保留编排器的原始计数。失败任务和无完整终态的已开始任务保留在分母；任务内周期有相关性，不能当作独立任务扩大样本量。

优先使用 run 内归档的分析器；旧版本输出若没有 `source/`，应先根据原 manifest 的摘要找到对应源码，不直接以当前文件冒充当时版本。

### 四卡片资源时间曲线

已有 `summary.json`、各 trial 原始指标和事件后，可用同一绘图入口生成浅灰底、蓝橙对照的四张资源卡片；不会重新运行业务或改写原始数据：

```bash
python -s scripts/agentenv_offload/plot_results.py "$RUN_DIR" \
  --view resources --pair 0 --output "$RUN_DIR/resource-cards.png"
```

绘图需要 matplotlib、numpy 与中文字体（例如 Microsoft YaHei 或 Noto Sans CJK SC）；可在具备这些依赖的本地环境读取下载后的结果目录。`--pair` 选择展示的真实配对，标题同时说明全组配对数、状态内存、合成等待和共同窗口；选中一对的曲线与节省率不能冒充全组加权结果。

四图分别为归属进程 PSS、累计 PSS GiB·s、已验证业务 VM 驻留阶段，以及节点匿名页/缓存相对各自测量起点的变化。原始采样点不平滑，缺样与过长间隔保留断线；累计曲线遇缺口后不再填写完整累计值。业务 VM 图依据生命周期证据和对应 PID/starttime，转换阶段留白；预热池虽不计为业务 VM，其资源仍包含在总 PSS 中。节点缓存曲线包含背景负载，不能据此推导外层物理 RAM 净节省。原有耗时/I/O 总览仍可使用 `--view summary`。

主要产物：

- 根目录 `manifest.json`、`results.jsonl`、`run_summary.json`、固定源码 `source/`、分析后的 `summary.json` 和 `report.md`。
- 每个 trial 的 `events.jsonl`、`metrics.jsonl`、`result.json`、`cleanup.json`、PID registry 与采样器日志。
- 部署目录中的真实源码/二进制/依赖摘要、配置、VM 镜像摘要与专属资源归属记录。凭据不复制到结果目录。

分析中内存积分需要 A/B 完整覆盖共同窗口，时钟误差及缺测必须可接受；不把缺测填零。进程 I/O 按 `(pid,starttime_ticks)` 单独差分，设备 I/O 按每设备列出，不把逻辑与物理层重复相加。进程退出前未采到的 I/O 不冒充完整计数。

清理至少区分 API/PID 消失与完整资源审计：后者还检查本轮 persister/managed-snapshot artifact、ublk 设备及网络 namespace 的前后差异。`resource_audit.status=pending` 表示仍需核对，不能算清理通过。不得为了消除差异而删除共享基础镜像、缓存、其他 VM 或未归属设备。

namespace 审计同时覆盖全局 `/run/netns` 和部署的 `<runtime-dir>/netns`，本拓扑后者为 `/run/agentenv-offload/netns`。`netns_by_directory` 保留来源，`netns` 与 `new_netns` 使用完整路径标识，避免不同目录中的同名 namespace 混淆。新增的私有目录 namespace 也会阻止完整清理确认；只有独立证据核实为共享缓存、并纳入后续固定 warmup baseline 后，才能作为后续轮次的既有背景。旧 run 的原始记录保持原样，外部补审计证据另行保存。

已完成的合成等待实验、主组收益、64 MiB 延迟边界和下载通路限制统一见 [结果报告](https://github.com/zzzzzzzxh/uni-agent/blob/c6b652ff8a54318967f88598535aaa16160c1b10/docs/sandbox_offload_results.md)。当前可确认的是给定配置下的运行态正确恢复、归属进程 PSS 成本下降及相应恢复/I/O 代价；真实模型任务吞吐、跨节点恢复、并发容量和外层物理 RAM 收益仍需各自验证。
