# RFC: 上下文感知调度及KVC池化 for Agentic RL Rollout

> **Stack**: LLM Router (AiBrix / Dynamo / ...) + vLLM (主线) / SGLang (对照) + Mooncake KV Store
> **Status**: Draft · 2026-05-14

## 1. 背景：Agentic RL Rollout 的 Prefix 增长曲线

Agentic RL 的 rollout 与传统 single-turn 推理的根本差别，在于 prompt 不是一次性给定的，而是**多轮拼接、单调增长**的。一条典型 SWE-Bench 任务的 trajectory 大致长这样：

```
turn 1:  send = system_prompt + tools_schema + user_task
         recv = assistant_msg_1 (含 tool_call_1)
         exec = tool_runtime → observation_1

turn 2:  send = turn-1 send + assistant_msg_1 + observation_1
         recv = assistant_msg_2
         ...

turn N:  send = turn-(N-1) send + assistant_msg_(N-1) + observation_(N-1)
```

每一轮发给推理引擎的 prompt，**都包含上一轮发送的所有 token，再加上一轮的 response 与新一轮的 observation**。也就是说，从第二轮起，发送内容里有一段不断变长的 shared prefix，这段 prefix 在上一轮已经被推理引擎完整算过一次了。

这个性质给出了一个非常诱人的优化空间：只要推理引擎能复用上一轮的 KV，第 N 轮的 prefill 就只需要算「上一轮 response + 这一轮 observation」这一小段增量；否则就要把 N-1 轮累积下来的几万到几十万 token 全部重新 prefill 一遍。在多 worker、长 trajectory 的 RL rollout 里，后者意味着把绝大部分 GPU 时间花在重复计算上。

## 2. 问题：Prefix Cache 命中率的几类失效场景

§1 的 prefix 增长性质把 **prefix cache 命中率**放到了 rollout 性能的中心位置——命中一段长 prefix，意味着对应那段 token 的 prefill 不用算；丢一段 prefix，则要从零重算。Agentic RL 的真实工作流里，命中率会从三个独立的方向被打掉。verl 现有的 sticky session 调度（`GlobalRequestLoadBalancer` + `AsyncLLMServerManager`：以 `request_id → server_id` 写入全局 LRU cache，同 episode 后续 turn 复用绑定）只覆盖了其中一类的一部分，另外两类完全没有解决方案。

### 2.1 同 session 多轮，但 KV 漂走了

sticky 调度试图解决的就是这一类：让同一 episode 的多轮请求落到同一 replica，命中本地 GPU KV。但绑定本身只是一张 LRU 表，不感知 KV 实际状态。三个独立的失效路径都不会被它检测到：

- **LRU 淘汰**：长尾 session 在 sticky 表里被新 request 挤出去，下一轮被当成新 request 重新分配；
- **本地驱逐**：映射还在，但 vLLM PagedAttention 内部的 block 早被驱逐——长 prompt 累积得越多，越容易把自己挤出去；
- **Replica 故障 / 扩缩容**：replica OOM 重启、被替换、被扩缩容移除时，所有绑定到它的 session 都要重新分配；扩容新加进来的 replica 短期内是个 KV 孤岛，老 session 不会迁移过来。

### 2.2 跨 session 的共享 prefix 没有共享

一个 RL step 里同一 task 往往要跑多份 parallel sample（GRPO 的 group），它们共享 `system_prompt + task_description + tool_schema`——通常几千 token。但每个 sample 是独立 session_id，会被打散到不同 replica，**并发**到达；各 replica 本地 miss、池中也 miss（首次出现），N 个 replica 同时启动同样的 prefill，构成经典的 cache stampede。

这一类失效 sticky 路由根本看不到——它只处理 session 内复用，不处理 session 间共享。

### 2.3 Weight Update 之后，整池失效

Sync RL 每个 step 都要 reset 推理引擎的 KV——不同 weight 算出的 KV 不能混用，否则梯度估计就坏了。下一个 step 的 turn 1 不论怎么调度都是从零 prefill：`N_session × (system_prompt + task + tools_schema)` tokens 全量重算。这是 RL 场景独有的痛点，sticky session 完全没办法。

把三类合在一起，结论很清楚：现有 sticky 调度只在 §2.1 上有部分覆盖，§2.2 与 §2.3 完全没有应对手段。我们要补齐的也正是这一组缺口。

## 3. 设计目标

**唯一目标：最大化 rollout 阶段的 prefix cache 命中率。**

围绕这个目标，三个互补的手段分别对应 §2 的三类失效——

1. **亲和调度**（§5.1 + §5.3 路由规则的 fast path）：把同一 session 的多轮请求尽量路由到同一 replica，优先命中本地 GPU KV。常态走 session_id 一致性哈希后做一次轻量检查（健康 + 命中 + 不过载），通过即 O(1) 转发；不通过则在所有 replica 上跑 §5.3 的两段规则。
2. **CPU 内存 KVC 二级池**（§5.1 connector + §5.2 version key）：用 Mooncake 作为跨 replica 共享的 L2 KV 池，常驻 CPU 内存、RDMA 互通。当亲和被打破（sticky 表淘汰、本地驱逐、replica 重启、扩缩容），从池 load 而非从零重算。这一层对 agentic RL 多轮拼接、prompt 持续增长的场景尤其关键——一条长 session 累积的几万到几十万 token KV，一旦本地被驱逐就只能靠池子兜底。
3. **Prefill 预热**（§5.4）：每次 weight update 完成、rollout 放行前，让每个 replica 独立预先 prefill 共享前缀（`system_prompt + tools_schema + 当前 step 的 task descriptions`）。weight 更新之后第一轮请求就能本地命中，消除 §2.3 的 step turn-1 全 miss 和 §2.2 的 GRPO group cold-start stampede。

一条横切的语义约束贯穿三个手段：**所有 KV 复用必须严格匹配 weight version**（§5.2 的 version-tagged key），保证任何被复用的 KV 一定是用正确 weight 算出来的，不会污染梯度。

对 agent 上层透明是非功能性要求：继续保持 OpenAI-compatible 接口，agent 只需要在请求里多带 `session_id` 和 `weight_version` 两个 metadata 字段。

## 4. 架构：Context-aware Scheduling 的三层

```
                ┌─────────────────────────────────────────┐
                │            Agent Loop    │
                │   每轮请求带: session_id, weight_version  │
                └────────────────────┬────────────────────┘
                                     │ OpenAI-compatible API
                                     ▼
                ┌─────────────────────────────────────────┐
                │   LLM Router (AiBrix / Dynamo / ...)     │
                │  · session_id consistent hashing         │
                │  · rule-based fallback (§5.3)            │
                │  · replica health & load 感知              │
                └──────────┬───────────────────┬──────────┘
                           │                   │
              主路由        ▼     fallback      ▼
                ┌──────────────────┐   ┌──────────────────┐
                │  vLLM replica 1   │   │  vLLM replica 2   │
                │  PagedAttention   │   │  PagedAttention   │
                │  + Mooncake conn. │   │  + Mooncake conn. │
                └────────┬──────────┘   └────────┬──────────┘
                         │  load/store KV blocks │
                         ▼                       ▼
                ┌─────────────────────────────────────────┐
                │   Mooncake KV Pool  (CPU + RDMA-shared)  │
                │   key = (hash(token_prefix), weight_ver) │
                │   eviction: LRU                          │
                └─────────────────────────────────────────┘
                                     ▲
                                     │ trainer 推新 weight 时附带新 weight_version
                ┌─────────────────────────────────────────┐
                │              Trainer (verl)              │
                └─────────────────────────────────────────┘
```

三层各司其职：

- **Routing Layer (LLM Router)**：唯一的入口路由，负责把 agent 请求送到合适的 vLLM replica。常态走 session 一致性哈希 fast path（O(1)）；不通过则按 §5.3 的两段规则——优先选 GPU 命中且负载达标的 replica，否则选负载最轻者。本 RFC 不绑定具体实现——AiBrix、Dynamo 等开源 LLM router 任选其一，按下文 §5.3 描述的规则与上报协议接入即可。
- **Inference Layer (vLLM)**：PagedAttention 不变；增加一个 Mooncake KV connector，负责本地 GPU KV 与 Mooncake 池之间的双向迁移：本地 miss 时从池里 load，block 被驱逐前 dump 到池里。
- **KV Pool (Mooncake)**：跨 replica 共享的二级 KV 池，常驻 CPU 内存，依靠 RDMA 在 replica 之间高速搬运。Key 由 token prefix hash 与 weight_version 共同决定，LRU 驱逐。

## 5. 关键机制

### 5.1 请求路径

一个典型的 turn-N 请求按以下顺序流动：

1. **Agent 发起请求**。Agent 调用 OpenAI-compatible 接口，request body 增加两个 metadata 字段：`session_id`（=episode id）和 `weight_version`（agent 持有的当前 actor weight 版本号）。
2. **LLM Router 路由**。Router 按 §5.3 的两段规则选 replica：常态走 fast path——用 `session_id` 一致性哈希取 primary，对它单独跑规则 (1) 的检查（健康 + GPU 命中 + 不过载），通过即 O(1) 转发；不通过则在所有 replica 上跑完整规则。
3. **vLLM 接收请求**。Replica 在本地 PagedAttention 中查 prefix：
   - **本地命中**：直接增量 prefill 剩余 token。
   - **本地未命中**（首次到达 / KV 已被驱逐 / 规则 (2) 兜底路由到此 replica）：通过 Mooncake connector 用 `(hash(prefix), weight_version)` 查 Mooncake 池：
     - **池中命中**：异步 RDMA load 缺失的 KV blocks 到本地 GPU；
     - **池中未命中**：完整 prefill，并在结束后异步把新算出的 KV blocks 回写 Mooncake。
4. **Decode + 返回**。Decode 与正常 vLLM 流程一致，结果按 OpenAI-compatible response 返回。
5. **Block 被驱逐前 dump**。PagedAttention 在驱逐 block 之前，通过 connector hook 把 block 写入 Mooncake（如果尚未写入），保证 GPU KV 的「淘汰」等价于「降级到二级池」，而不是「丢失」。

### 5.2 Weight 更新协同

Mooncake key 显式包含 `weight_version`，使 weight 协同退化为「自然过期」：

- **Sync RL**：每个 step 结束、trainer 把新 weight 推给 inference replica 时，附带新的 `weight_version`。Replica 局部 KV 按既有方式 reset；Mooncake 池里的老 version KV 不需要任何主动 invalidation——下一个 step 的请求带的是新 version，查询时 key 不匹配，自然 miss；老 version KV 在 LRU 下被慢慢淘汰。这把 §2.3 描述的「整池 KV 必须显式 reset」需求转化成了「老 entry 自然失效」，避免了 trainer-router-pool 之间的同步广播。
- **Fully-Async RL**：训练与 rollout 并发，多 weight 版本天然共存。每个 rollout worker 启动时携带它当前用的 `weight_version`，对应版本的 KV 在 Mooncake 池里独立编 key，互不干扰。

这套机制有一个直接的副作用：Mooncake 池容量被多个 weight_version 分摊。但因为 LRU 会优先淘汰老 version 的冷 entry，热 prefix（system_prompt + tool_schema 这类几乎全场景共用的内容）始终留在池里，命中率受影响有限。

### 5.3 路由规则

为了让规则容易实现、容易解释，本 RFC 采用一条两段式的 naive 规则，而不是复杂的代价函数：

1. **优先**：选 prefix 在 GPU 显存能命中（`L_gpu(r) ≥ hit_threshold`）**且**负载低于阈值（`wait(r) < load_threshold`）的 replica；多个候选取本地命中长度 `L_gpu(r)` 最大者。
2. **兜底**：若 (1) 无任何候选，选负载最轻（`argmin wait(r)`）的 replica，让它本地 prefill 或从 Mooncake load。

三件事被分别处理：

- **GPU 命中**：规则 (1) 的第一个条件。
- **负载均衡**：规则 (1) 的阈值过滤把高负载 replica 挡在 GPU 命中候选之外；规则 (2) 进一步以最轻负载兜底。
- **重算 vs Mooncake load**：**不在路由层**做选择——一旦请求落到 replica，§5.1 的请求路径自动处理（池中命中走 RDMA load，否则 prefill），无需 router 提前判断。

**输入**：`L_gpu(r)` 来自 replica 周期性上报的本地 prefix 索引；`wait(r)` 来自 replica 周期性上报的 in-flight 计数。`hit_threshold` 与 `load_threshold` 由 profiling 给出，可配置。

**Fast path**：在 session 多轮的常态请求里，`session_id` 一致性哈希选定的 primary replica 几乎总是规则 (1) 的赢家——本地 KV 完整、绑定连续、负载稳定。Router 可以先用一致性哈希拿 primary，对它单独跑一次规则 (1) 检查；通过则跳过对其他 replica 的扫描，O(1) 直接转发。

**退化行为**：

- Primary 故障：视为 `wait(primary) := +∞`（或健康检查不通过），自动跌出候选集；
- 所有 replica 本地都 miss：规则 (1) 无候选，规则 (2) 选最轻负载者；
- Mooncake 池也 miss：被选中的 replica 完整 prefill，与无 cache 持平。

### 5.4 Prefix 预热

§2.2 与 §2.3 两个痛点——GRPO group 的 cold-start stampede 与 Sync RL 每 step turn-1 全 miss——本质是同一件事：weight 刚更新完，replica 上没有任何匹配新 `weight_version` 的 KV，rollout 一开放就有 N 个并发请求扑过来。与其让 stampede 在请求路径上发生，不如把这段必然要算的 prefix prefill 前置到 weight update 完成、rollout 真正开放之前。

**预热范围**：默认 `system_prompt + tools_schema + 当前 step 的所有 task descriptions`。前两者跨整个训练几乎不变，每个 weight version 只需预热一次；task descriptions 是 GRPO group 共享的部分，预热之后同 group 内的 N 个 sample 全部本地直接命中。

**执行**：每个 replica **独立、并行** prefill 整个预热集合。这里刻意不走 leader 算 + 其他从 Mooncake load 的路线——预热 prefix 一般只有几千 token，相比 Mooncake RDMA 往返，每个 replica 本地各算一次更快、更简单；wall-clock 时间等于单次 prefill，而不是 R 倍。算出的 KV 进入本地 PagedAttention 的常规 prefix cache，被后续请求频繁访问后自然停留在 LRU 热端，无需额外 pinning 机制。

**时序**：阻塞。所有 replica 上报 warm done 后 rollout 才放行。Sync RL 中 weight update 本就是 critical path，把预热附在它之后只是把"原本第一个 step 第一轮的 prefill 时间"前置了，总耗时基本不变；收益是后续所有第一轮请求直接本地命中，决定性地消除了 stampede。

这一节与 §5.1–§5.3 完全解耦：路由策略、Mooncake 池、weight_version 协议都不需要任何修改。预热生成的 KV 在本地命中路径上被使用，连 Mooncake load 都跳过。

## 6. 与现有方案的对照

| 能力 | vLLM 原生 prefix cache | verl `AsyncLLMServerManager` (sticky) | SGLang RadixAttention | Context-aware Scheduling (本 RFC) |
|---|---|---|---|---|
| 同 episode 跨 turn 复用 | 单 replica 内 | ✓ (sticky binding) | 单 replica 内 | ✓ (session_id 路由) |
| 跨 replica KV 共享 | ✗ | ✗ | ✗ | ✓ (Mooncake 池) |
| Replica 故障/重启后恢复 | ✗ | ✗ (映射作废) | ✗ | ✓ (从池 load) |
| 弹性扩容后新 replica 利用 | ✗ | ✗ (新 session 才用) | ✗ | ✓ (规则选优) |
| Weight 更新后的语义安全 | 需手动清 | 需手动清 | 需手动清 | ✓ (version-tagged key) |
| 长 session 被本地驱逐后 | 全重算 | 全重算 | 全重算 | 池中命中即可 |
| Cold-start (step turn-1 / GRPO group) | ✗ | ✗ | ✗ | ✓ (pre-warming) |
| 路由开销（常态） | N/A | O(1) (LRU lookup) | N/A | O(1) (consistent hash) |
| 路由开销（非 fast path） | N/A | 无 | N/A | O(R) 规则扫描 (R=replica 数) |

简而言之：vLLM/SGLang 解决「单进程内的 prefix 复用」，verl sticky 解决「同 episode 路由稳定」，Context-aware Scheduling 在它们之上补两件事——一个跨进程的 KV 共享层，和一个能感知 KV 状态的路由策略。

## 7. SGLang 接入差异

切换到 SGLang 时，本 RFC 的三层架构保持不变，主要差别集中在 inference layer：

- **本地 prefix 索引天然就有**：SGLang 的 RadixAttention 内部以 radix tree 形式维护 prefix 索引，给 router 上报本地 prefix 集合时不需要额外维护数据结构，一次树遍历即可。
- **Mooncake connector 接入点不同**：vLLM 的 connector hook 在 PagedAttention 的 block manager 层；SGLang 的 hook 在 radix tree 节点的分配/驱逐回调上。语义等价——把「block 驱逐」转化为「写 Mooncake」。
- **Weight 更新 API 不同**：SGLang 通过 `/update_weights` 接口；接入时 `weight_version` 参数加在 metadata 即可，与 router/pool 的协议不变。

Routing layer (LLM Router) 与 KV Pool (Mooncake) 完全引擎无关。一套 router 可以同时管 vLLM 和 SGLang replica；甚至允许混合部署（罕见但可行）。

## 8. Open Questions / Future Work

- **跨 weight version 的 KV 复用**。目前我们要求严格匹配 `weight_version`。理论上，相邻 weight 之间 KV 近似可用，配合 importance-sampling 修正项可以进一步提升命中率。代价是数学复杂度与额外验证。本 RFC 不展开。
- **PD Disaggregation 的接入**。Mooncake 原生支持 prefill/decode 分离部署，可以把 prefill 负载隔离到独立 pool，进一步降低长 prompt 对 decode 延迟的干扰。本 RFC 暂不涉及，作为后续阶段。
- **多模态 KV 的 prefix hash**。图像 patch tokens 的语义 hash 与文本 token id hash 规则不同（同一图片在不同预处理路径下可能产生不同 token），需要单独定义可复现的 hash 协议才能进入 Mooncake 池。

---

*本 RFC 为对外技术分享版本；内部实施细节、依赖版本与里程碑请见对应工程文档。*
