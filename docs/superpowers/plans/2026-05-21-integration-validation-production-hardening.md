# Plan E: 集成验证 + Production Hardening 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This plan is a validation/hardening gate for Plans A-D; do not mark it complete from unit tests alone.

**Goal:** 把 Plans A-D 已完成的 `llm_router` 代码骨架从"单机单测/局部 e2e 可信"推进到"真实 GPU + Mooncake + verl trainer 路径已验证，且生产首日不会因已知 follow-up 崩掉"。Plan E 不重新设计 RFC；它只做两件事：第一，把之前所有 skipped / mocked / inferred 的关键路径在真实环境里跑通；第二，修掉 Plan B/C 中会直接影响生产可用性或命中率的缺口。

**Current boundary:** Plans A-D 已经交付了生产代码文件、配置解析、策略逻辑、Mooncake connector 骨架、prewarm/report 路由链路，以及 77 个自动化测试。但这些测试没有覆盖真实 GPU KV tensor layout、真实 Mooncake TransferEngine/RDMA、真实 vLLM scheduler connector 加载、真实 verl trainer 配置加载和 worker 触发的 `report_prefixes` / `prewarm_prefixes`。

**Architecture:** Plan E 分两条线推进：

- **Integration validation track:** 新增 host-only 验证脚本/测试配置，把 `test_manager_parity.py`、`test_mooncake_store.py`、vLLM connector、verl trainer minimal step、context-aware report、prewarm flow 都在真实环境中跑起来，并保存可复现命令与结果。
- **Production hardening track:** 修 Plan B/C follow-up：Mooncake buffer eviction、async RDMA load、跨 replica peer pool、connector -> LB direct reporting、stride-aligned key granularity。每个改动必须带单测；涉及 GPU/Mooncake/vLLM 的改动还必须有 host-only 集成验证。

**Tech Stack:** Python 3.10+, pytest, Ray, vLLM v1 KV connector framework, Mooncake TransferEngine, PyTorch CUDA, OmegaConf, verl trainer。需要至少一台可安装 Mooncake 的 GPU 主机；如要验证 peer pool/RDMA，至少两台互通主机或同机多 engine 模拟。

---

## Non-Goals

- 不重写 Plans A-D 的 RFC 或策略目标。
- 不用 `MagicMock` 或 `InMemoryKVStore` 作为 Plan E 的完成证据；它们只能作为快速回归测试。
- 不把 skipped 测试视为通过。GPU/Mooncake host-only 测试在非目标环境可以 skip，但 Plan E 完成记录必须包含目标环境上的真实 pass 结果。
- 不把"读 verl 代码推断链路存在"视为验收；必须跑真实 trainer 启动序列和至少一个最小 RL/rollout step。

---

## Environment Matrix

| Environment | Purpose | Required before complete |
|---|---|---|
| CPU dev host | 快速单测、ruff、schema/config 回归 | Yes |
| Single GPU host | Plan A parity、vLLM connector 加载、真实 KV tensor layout | Yes |
| Single GPU + Mooncake package | `MooncakeKVStore` round-trip、TransferEngine 初始化、buffer 注册 | Yes |
| verl trainer minimal run | FQN plugin、OmegaConf -> `LLMRouter.__init__`、worker-side report/prewarm | Yes |
| Multi-replica Mooncake/RDMA host(s) | cross-replica peer pool、remote KV load、routing 命中率 benchmark | Yes before production rollout |

---

## Task 1: 建立 Plan E 验证入口与运行记录

**Files:**
- Create: `llm_router/tests/integration/README.md`
- Create: `llm_router/tests/integration/test_env_contract.py`
- Create: `llm_router/tests/integration/run_plan_e_validation.sh`
- Create: `docs/superpowers/validation/plan-e-results.md`

- [ ] **Step 1: 写 integration README**

记录三类命令：

```bash
# CPU regression
pytest llm_router/ -v
ruff check llm_router

# GPU-only validation
CUDA_VISIBLE_DEVICES=0 pytest llm_router/tests/test_manager_parity.py -v -s

# Mooncake host validation
CUDA_VISIBLE_DEVICES=0 pytest llm_router/connector/tests/test_mooncake_store.py -v -s
```

README 必须明确：在非 GPU/Mooncake host 上 skip 是允许的；但 Plan E 完成证据必须来自目标 host 的 pass。

- [ ] **Step 2: 增加环境契约测试**

`test_env_contract.py` 只检查并打印可诊断信息，不替代功能测试：

- `torch.cuda.is_available()`
- CUDA device count/name
- `mooncake` import/version 或 import error
- vLLM import/version
- Ray import/version
- verl import path

Expected: 在 CPU host 上该测试可 pass，但会把缺失能力写进 pytest output；在 GPU/Mooncake host 上必须显示 CUDA + Mooncake + vLLM 可用。

- [ ] **Step 3: 写一键验证脚本**

`llm_router/tests/integration/run_plan_e_validation.sh` 顺序执行：

1. `pytest llm_router/ -v --tb=short`
2. `ruff check llm_router`
3. GPU parity test
4. Mooncake store test
5. vLLM connector scheduler smoke test（Task 3 之后接入）
6. verl trainer minimal step（Task 4 之后接入）

脚本必须把每段 stdout/stderr tee 到 `artifacts/plan-e/<timestamp>/`，并在任一 required gate fail 时非零退出。

- [ ] **Step 4: 新建结果记录文档**

`plan-e-results.md` 维护每次真实 host 验证结果：

| Date | Host | GPU | Mooncake | vLLM | verl commit | Command | Result | Notes |
|---|---|---|---|---|---|---|---|---|

- [ ] **Step 5: 提交**

```bash
git add llm_router/tests/integration/ docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] test: add Plan E integration validation harness"
```

---

## Task 2: 真 GPU 跑 Plan A parity，验证 FQN 插件点

**Files:**
- Modify: `llm_router/tests/test_manager_parity.py`
- Modify: `llm_router/tests/integration/README.md`
- Modify: `docs/superpowers/validation/plan-e-results.md`

- [ ] **Step 1: 在 GPU host 上运行 parity**

```bash
CUDA_VISIBLE_DEVICES=0 pytest llm_router/tests/test_manager_parity.py -v -s --tb=short
```

Expected: 不再 skip；stock `AgentLoopManager` 与 `llm_router.LLMRouter` 的最小 rollout 输出 token parity 通过。

- [ ] **Step 2: 若失败，按失败类型修**

常见失败分支：

- FQN import/load 失败：修 `pyproject.toml` package discovery 或 `llm_router.__init__` export。
- `LLMRouter.__init__` 签名不匹配：对齐 verl 当前 `AgentLoopManager` 调用参数。
- Ray actor/worker 参数缺失：修 `LLMRouter.create()` 与 rollout replica 初始化。
- 输出不一致：定位 `LegacyStickyPolicy` 与 stock load balancer 差异。

- [ ] **Step 3: 把 parity test 从 stub 变成真实断言**

删除仅记录 TODO 的占位逻辑；保留 GPU skip guard，但在 GPU 可用时必须执行完整 stock-vs-router 对比。

- [ ] **Step 4: 更新结果记录**

在 `plan-e-results.md` 写入命令、host、GPU 型号、verl commit、结果和修复摘要。

- [ ] **Step 5: 提交**

```bash
git add llm_router/tests/test_manager_parity.py llm_router/tests/integration/README.md docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] test: validate LLMRouter parity on real GPU"
```

---

## Task 3: 真 Mooncake + vLLM scheduler 加载 connector

**Files:**
- Modify: `llm_router/connector/tests/test_mooncake_store.py`
- Create: `llm_router/connector/tests/test_vllm_scheduler_connector_integration.py`
- Modify: `llm_router/connector/README.md`
- Modify: `docs/superpowers/validation/plan-e-results.md`

- [ ] **Step 1: 在 Mooncake host 上跑 store 测试**

```bash
CUDA_VISIBLE_DEVICES=0 pytest llm_router/connector/tests/test_mooncake_store.py -v -s --tb=short
```

Expected:

- `mooncake` 包真实 import；
- `TransferEngine.initialize` 成功；
- registered buffer 成功；
- put/get/delete 或等价 round-trip 成功；
- buffer full 行为可观测。

- [ ] **Step 2: 新增 vLLM scheduler connector smoke test**

测试必须通过 vLLM 的真实 `KVConnectorFactory` / scheduler 初始化路径加载 `"MooncakeKVConnector"`，不能只 import class。

Minimum assertions:

- `register_with_vllm()` 幂等；
- `kv_transfer_config.kv_connector: "MooncakeKVConnector"` 能被 scheduler 接受；
- connector 的 `start_load_kv` / `save_kv_layer` hook 在真实 vLLM request 生命周期中至少被调用一次；
- 对 FlashAttention backend 的 KV tensor shape 做断言并记录。

如果环境支持 MLA backend，增加参数化运行并记录 layout 差异。

- [ ] **Step 3: 验证真实 tensor layout**

在 connector hook 内记录每层 KV tensor：

- dtype
- device
- shape
- stride
- layer name / layer index
- backend name

Expected: `MooncakeKVConnector` 的序列化/反序列化逻辑不依赖错误 shape 假设。

- [ ] **Step 4: 更新文档**

`llm_router/connector/README.md` 增加真实 vLLM connector 验证命令和已验证 backend matrix。

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/tests/test_mooncake_store.py llm_router/connector/tests/test_vllm_scheduler_connector_integration.py llm_router/connector/README.md docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] test: validate Mooncake connector in real vLLM scheduler"
```

---

## Task 4: 真 verl trainer minimal step，验证配置入口与跨进程协议

**Files:**
- Create: `examples/agent_train/llm_router_minimal_plan_e.yaml`
- Create: `examples/agent_train/run_llm_router_plan_e_minimal.sh`
- Create: `llm_router/tests/integration/test_verl_trainer_minimal.py`
- Modify: `llm_router/README.md`
- Modify: `docs/superpowers/validation/plan-e-results.md`

- [ ] **Step 1: 写最小 trainer config**

Config 必须显式打开：

```yaml
actor_rollout_ref:
  llm_router:
    policy: rule_based
    hit_threshold: 1
    load_threshold: 1024
    max_prefix_entries_per_server: 8192
    routing_cache_size: 10000
  rollout:
    agent:
      agent_loop_manager_class: "llm_router.LLMRouter"
    context_aware_scheduling:
      enable: true
      prefix_probe_stride: 256
      prewarm_enable: true
      prewarm_decode_tokens: 1
```

Keep model/dataset/batch sizes minimal enough to run one rollout/training step on a single GPU.

- [ ] **Step 2: Instrument `LLMRouter.__init__` config receipt**

Add debug-level structured logging or test-only observable fields showing:

- selected policy;
- thresholds;
- `context_aware_scheduling.enable`;
- prewarm config;
- server ids.

Do not print secrets or huge OmegaConf dumps.

- [ ] **Step 3: Run true trainer startup**

```bash
bash examples/agent_train/run_llm_router_plan_e_minimal.sh
```

Expected:

- trainer imports `llm_router.LLMRouter` through FQN;
- OmegaConf values reach `LLMRouter.__init__`;
- Ray actors start without local_mode;
- rollout invokes `LoadBalancer.acquire_server.remote(...)` with `prefix_signatures`;
- generation completion invokes `LoadBalancer.report_prefixes.remote(...)`;
- `_maybe_prewarm_prefixes` invokes real prefill and then report;
- run completes at least one minimal RL/rollout step.

- [ ] **Step 4: Add integration assertion**

`test_verl_trainer_minimal.py` may shell out to the script behind a marker such as `@pytest.mark.integration_gpu`. It must fail if logs do not contain evidence for:

- FQN load;
- config propagation;
- at least one acquire with non-empty `prefix_signatures`;
- at least one report from worker;
- at least one prewarm report when `prewarm_enable=true`.

- [ ] **Step 5: 提交**

```bash
git add examples/agent_train/llm_router_minimal_plan_e.yaml examples/agent_train/run_llm_router_plan_e_minimal.sh llm_router/tests/integration/test_verl_trainer_minimal.py llm_router/README.md docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] test: validate LLMRouter in real verl trainer"
```

---

## Task 5: MooncakeKVStore buffer eviction

**Files:**
- Modify: `llm_router/connector/store/mooncake.py`
- Modify: `llm_router/connector/tests/test_mooncake_store.py`
- Modify: `llm_router/connector/README.md`

- [ ] **Step 1: Specify eviction semantics**

Replace bump-only allocator behavior with bounded LRU semantics:

- each key maps to one or more registered buffer slices;
- `put()` evicts least-recently-used entries until enough contiguous or reusable space exists;
- `get()` refreshes recency;
- failed oversize `put()` returns a clear error if a single value exceeds total capacity;
- delete/free returns slices to a reusable free list;
- metrics expose used bytes, free bytes, entry count, eviction count.

- [ ] **Step 2: Write failing tests**

Tests must cover:

- filling the buffer beyond capacity evicts old entries instead of raising generic `RuntimeError`;
- recently read entry survives eviction;
- freed slices are reused;
- oversize single payload fails with a typed error;
- Mooncake memory registration remains valid after eviction/reuse.

- [ ] **Step 3: Implement reusable allocator**

Use a simple free-list allocator first. Avoid over-optimizing; correctness matters more than allocator sophistication.

- [ ] **Step 4: Run tests on CPU and Mooncake host**

CPU tests can use a fake/mock TransferEngine for allocator semantics. Mooncake host test must perform real put/get after eviction.

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/store/mooncake.py llm_router/connector/tests/test_mooncake_store.py llm_router/connector/README.md
git commit -m "[llm_router] fix: add eviction to MooncakeKVStore buffer"
```

---

## Task 6: Async RDMA load path

**Files:**
- Modify: `llm_router/connector/connector.py`
- Modify: `llm_router/connector/store/base.py`
- Modify: `llm_router/connector/store/mooncake.py`
- Modify: `llm_router/connector/store/in_memory.py`
- Modify: `llm_router/connector/tests/test_connector_unit.py`
- Modify: `llm_router/connector/tests/test_mooncake_store.py`

- [ ] **Step 1: Define async store contract**

Extend `KVStore` with an async transfer contract suitable for vLLM:

- `begin_get(key) -> transfer_id`
- `poll_get(transfer_id) -> pending | done(payload) | failed(error)`
- `cancel(transfer_id)`

Keep synchronous `get()` for tests/backwards compatibility if useful, but connector production path should use async methods.

- [ ] **Step 2: Update connector lifecycle**

`MooncakeKVConnector.start_load_kv` should enqueue async loads and return promptly. `get_finished` should return request ids only after all required layer transfers are complete. `wait_for_layer_load` / layer copy should consume completed payloads without blocking on RDMA.

- [ ] **Step 3: Write tests**

Use a controllable fake store to assert:

- `start_load_kv` does not synchronously block on payload availability;
- request is absent from `get_finished` while transfer pending;
- request appears when all layers complete;
- failed transfer marks request failed and falls back/recomputes according to vLLM connector expectations.

- [ ] **Step 4: Validate on Mooncake host**

Run a real vLLM request with connector enabled and verify async transfer state transitions are observed.

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/connector.py llm_router/connector/store/ llm_router/connector/tests/
git commit -m "[llm_router] feat: make Mooncake KV loads asynchronous"
```

---

## Task 7: Cross-replica Mooncake peer pool

**Files:**
- Modify: `llm_router/connector/store/mooncake.py`
- Create: `llm_router/connector/tests/test_mooncake_peer_pool.py`
- Modify: `llm_router/connector/README.md`

- [ ] **Step 1: Define peer configuration**

Add config for:

- local engine id/address;
- peer engine addresses;
- RDMA device/interface;
- namespace or job id;
- timeout/retry policy.

- [ ] **Step 2: Implement remote lookup/load**

`MooncakeKVStore.get()` / async get should:

1. check local metadata;
2. if absent, query peer metadata;
3. transfer from the peer owning the key;
4. optionally admit into local buffer subject to eviction.

- [ ] **Step 3: Test same-host multi-engine first**

Create two store instances with separate buffers and assert:

- store A writes key;
- store B misses locally;
- store B loads from A through peer path;
- version mismatch still misses.

- [ ] **Step 4: Validate RDMA/multi-host**

Run the same test across the intended deployment topology and record results in `plan-e-results.md`.

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/store/mooncake.py llm_router/connector/tests/test_mooncake_peer_pool.py llm_router/connector/README.md docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] feat: add cross-replica Mooncake peer pool"
```

---

## Task 8: Connector -> LB direct reporting

**Files:**
- Modify: `llm_router/connector/connector.py`
- Modify: `llm_router/connector/meta.py`
- Modify: `llm_router/load_balancer.py`
- Modify: `llm_router/tests/test_load_balancer.py`
- Modify: `llm_router/connector/tests/test_connector_unit.py`
- Modify: `llm_router/README.md`

- [ ] **Step 1: Define reporting contract**

Connector should report actual KV store state, not just worker-observed prompt handling:

- on successful save/admit into Mooncake;
- on successful local/remote async load;
- on eviction/delete as negative report or invalidation;
- include `server_id`, `weight_version`, `prefix_hash`, `prefix_len`, and source (`local_gpu`, `mooncake_local`, `mooncake_peer`).

- [ ] **Step 2: Extend LB/policy as needed**

If `RuleBasedPolicy` only supports positive `report_prefixes`, add an invalidation path such as `drop_prefixes` or `report_prefixes(..., present=False)` without breaking existing verl worker reports.

- [ ] **Step 3: Write tests**

Assert:

- connector save causes LB prefix location update;
- connector eviction invalidates stale route signal;
- worker report and connector report are idempotent;
- direct report wins over proxy when actual store state differs.

- [ ] **Step 4: Validate in real trainer**

Run minimal trainer and verify routing signals appear from both worker report and connector direct report.

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/ llm_router/load_balancer.py llm_router/tests/test_load_balancer.py llm_router/README.md
git commit -m "[llm_router] feat: report real connector KV state to load balancer"
```

---

## Task 9: Stride-aligned key granularity

**Files:**
- Modify: `llm_router/connector/prefix_hash.py`
- Modify: `llm_router/connector/connector.py`
- Modify: `llm_router/connector/tests/test_prefix_hash.py`
- Modify: `llm_router/connector/tests/test_connector_unit.py`
- Modify: `llm_router/tests/test_rule_based_routing_e2e.py`
- Modify: `llm_router/README.md`

- [ ] **Step 1: Pin one prefix signature implementation**

Stop duplicating `_signatures()` helpers in tests. Export one helper that mirrors verl `_iter_prefix_signatures`:

- same stride behavior;
- same inclusion of full prompt length;
- same `(weight_version, prefix_hash, prefix_len)` tuple shape;
- same token hashing endianness and signedness.

- [ ] **Step 2: Save KV at stride-aligned granularity**

Connector save path must store payloads under keys matching the prefix lengths reported by worker/prewarm. If storing every stride is too expensive, define and test a deterministic nearest-lower/nearest-upper policy; routing must use the same policy.

- [ ] **Step 3: Update load path**

Given request signatures, connector should select the longest available compatible prefix key. This must match `RuleBasedPolicy`'s "largest GPU prefix hit" choice.

- [ ] **Step 4: Write mismatch regression test**

Reproduce the known failure:

1. store full prompt only;
2. worker reports 256-token stride prefix;
3. router routes as hit;
4. connector cannot find KV.

Then prove the fix: stride-reported key exists and load succeeds.

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/ llm_router/tests/test_rule_based_routing_e2e.py llm_router/README.md
git commit -m "[llm_router] fix: align connector KV keys with reported prefix stride"
```

---

## Task 10: End-to-end hit-rate benchmark and production gate

**Files:**
- Create: `scripts/llm_router/benchmark_context_aware_routing.py`
- Create: `docs/superpowers/validation/plan-e-benchmark.md`
- Modify: `docs/superpowers/validation/plan-e-results.md`

- [ ] **Step 1: Build minimal benchmark**

Benchmark should compare:

- stock verl routing;
- `LLMRouter` + `legacy_sticky`;
- `LLMRouter` + `rule_based`;
- `rule_based` + Mooncake connector;
- `rule_based` + Mooncake connector + prewarm.

Use the same model, prompts, rollout shape, and weight update cadence across runs.

- [ ] **Step 2: Record metrics**

At minimum:

- local GPU prefix hit rate;
- Mooncake pool hit rate;
- end-to-end rollout latency;
- prefill time;
- decode throughput;
- RDMA transfer latency;
- connector load/save failures;
- eviction count;
- route distribution;
- training step wall time.

- [ ] **Step 3: Define pass/fail gate**

Plan E is complete only if:

- all CPU tests pass;
- GPU parity test passes on real GPU;
- Mooncake store tests pass with real package;
- vLLM scheduler loads `MooncakeKVConnector`;
- true verl trainer minimal step runs with `context_aware_scheduling.enable=True`;
- worker-origin `report_prefixes` observed;
- real prewarm observed;
- buffer eviction works under pressure;
- async load path works;
- peer pool verified in target topology;
- stride-aligned key regression fixed;
- benchmark shows non-regressing correctness and records hit-rate/latency results.

- [ ] **Step 4: 提交 completion record**

```bash
git add scripts/llm_router/benchmark_context_aware_routing.py docs/superpowers/validation/plan-e-benchmark.md docs/superpowers/validation/plan-e-results.md
git commit -m "[llm_router] docs: record Plan E production validation results"
```

---

## Acceptance Criteria

1. `pytest llm_router/ -v` passes on CPU dev host with only environment-appropriate skips.
2. `CUDA_VISIBLE_DEVICES=0 pytest llm_router/tests/test_manager_parity.py -v -s` passes on a real GPU host and no longer acts as a placeholder.
3. `pytest llm_router/connector/tests/test_mooncake_store.py -v -s` passes on a host with the real `mooncake` package.
4. A real vLLM scheduler run loads `"MooncakeKVConnector"` through `kv_transfer_config` and exercises connector hooks with real KV tensors.
5. A real verl trainer minimal run loads `actor_rollout_ref.rollout.agent.agent_loop_manager_class: "llm_router.LLMRouter"` and propagates `actor_rollout_ref.llm_router.*` config to `LLMRouter.__init__`.
6. The same trainer run observes non-empty `prefix_signatures`, worker-origin `report_prefixes`, and prewarm-origin `report_prefixes`.
7. `MooncakeKVStore` has bounded eviction instead of bump-allocator exhaustion.
8. Mooncake loads are asynchronous from the connector perspective and integrated with vLLM completion polling.
9. Cross-replica Mooncake peer load works in the target topology.
10. Routing key granularity is aligned with worker/prewarm stride signatures, so a routed prefix hit can actually find matching KV.
11. `docs/superpowers/validation/plan-e-results.md` and `plan-e-benchmark.md` contain dated command outputs, host details, and benchmark results.

---

## Completion Note

Plans A-D can be described as "code skeleton + wiring diagram + automated local confidence." Plan E is the line where that becomes "validated integration + hardened production path." Do not close Plan E until the evidence comes from real GPU/Mooncake/vLLM/verl runs, not from mocks or local Ray mode.
