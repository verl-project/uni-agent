# llm_router.connector

vLLM v1 KV connector backed by Mooncake. Implements the connector layer
described in [RFC §5.1 / §5.2](../../docs/superpowers/specs/2026-05-14-context-aware-scheduling-design.md):

- bidirectional load/dump between local PagedAttention and a CPU-resident L2 pool;
- cache key = `(hash(token_prefix), weight_version)` — KV produced under one
  weight version never matches a query from another.

## Usage

In your training script, register once before the trainer constructs vLLM:

```python
from llm_router.connector import register_with_vllm
register_with_vllm()
```

Then point vLLM at the connector via `kv_transfer_config`:

```yaml
actor_rollout_ref:
  rollout:
    kv_transfer_config:
      kv_connector: "MooncakeKVConnector"
      kv_role: "kv_both"
      kv_connector_extra_config:
        weight_version: "${global_step}"   # or any string id
        mooncake_backend: "auto"            # auto | pooled | local
        mooncake_master: "10.0.0.1:50051"   # Mooncake master; enables pooled L2
        mooncake_metadata_server: "P2PHANDSHAKE"
        mooncake_protocol: "tcp"            # tcp or rdma
        mooncake_device_name: ""            # RDMA device when protocol=rdma
        mooncake_global_segment_bytes: 3355443200
        mooncake_local_buffer_bytes: 1073741824
        mooncake_prefer_alloc_in_same_node: true
        mooncake_replica_num: 1
        prefix_probe_stride: 256
        server_id: "${rollout_server_id}"   # optional, enables connector -> LB reporting
        load_balancer_handle: null          # optional Ray actor handle injected by trainer glue
```

`weight_version` MUST be threaded from the trainer in lockstep with weight
updates (Plan A's RFC §5.2 protocol). Trainers that don't rotate
`weight_version` will, by RFC, behave as if every entry in the pool came
from the same weight — which is *only* safe in evaluation/inference, never
in RL training.

## Backends

The connector talks to a `KVStore` interface. Two backends ship:

| Backend | When | Cap |
|---|---|---|
| `InMemoryKVStore` | unit tests | LRU eviction by total bytes |
| `MooncakePooledKVStore` | production | Mooncake `MooncakeDistributedStore` pooled L2 |
| `MooncakeKVStore` local mode | compatibility / single-host tests | LRU/free-list allocator inside one registered TransferEngine buffer |

By default `mooncake_backend: auto` selects the pooled backend whenever
`mooncake_master` or `MOONCAKE_MASTER` is configured. The pooled backend uses
Mooncake's `MooncakeDistributedStore` client, so capacity and placement come
from Mooncake master/metadata services rather than this connector's local
process. Set `mooncake_backend: local` only for single-host debugging.

For pooled mode, `MooncakePooledKVStore` defaults
`ReplicateConfig.prefer_alloc_in_same_node=True`. That asks Mooncake to place
new KV in the writer node's CPU memory first; with hard pinning off, Mooncake
can fall back to another node's memory segment when local CPU capacity is
insufficient. After each save, the connector reads Mooncake replica descriptors
and reports CPU placement hints back to the router.

`MooncakePooledKVStore` requires `mooncake-transfer-engine`, which provides
the `mooncake.store.MooncakeDistributedStore` Python module. Start the
Mooncake master / metadata services before constructing vLLM replicas, then
give each replica the same `mooncake_master` address.

`KVStore` also exposes a small async read contract:

- `begin_get(key) -> transfer_id`
- `poll_get(transfer_id) -> pending | done(payload) | failed(error)`
- `cancel(transfer_id)`

`MooncakeKVConnector.start_load_kv()` uses that contract and reports load
completion through vLLM's `get_finished()` path before copying payloads into
the registered KV cache on `wait_for_layer_load()`. The default in-memory
implementation completes reads synchronously; a Mooncake host can replace that
with true RDMA polling behind the same interface.

## Tests

```
pytest llm_router/connector/tests/ -v
```

Current local result with `mooncake-transfer-engine==0.3.10.post2`:

```
54 passed.
```

## Plan B/E status

- Done: production path can use Mooncake's pooled `MooncakeDistributedStore`
  via `MooncakePooledKVStore`.
- Done: compatibility `MooncakeKVStore` local mode still provides LRU-style
  eviction inside a single registered TransferEngine buffer.
- Done: connector-side async load state machine (`begin_get`/`poll_get`,
  `get_finished`, `wait_for_layer_load`) with unit coverage.
- Done: shared stride-compatible prefix signature helper, so connector keys
  and worker/prewarm routing reports use the same `(version, hash, len)` shape.
- Done: connector-side positive reporting hook for KV saves/loads when a
  `server_id` and `load_balancer_handle` are supplied through extra config.
- Remaining: true nonblocking Mooncake read polling behind `poll_get`; the
  pooled backend currently exposes the synchronous `MooncakeDistributedStore`
  get/upsert contract through the existing `KVStore` interface.
- Remaining: negative reporting/invalidation on eviction; today the connector
  only sends positive availability hints.
