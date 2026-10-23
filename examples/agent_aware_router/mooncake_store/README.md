# Mooncake Store

Enable a shared CPU KV cache for `run_infer.sh` using vLLM's
`MooncakeStoreConnector`. The router continues to use GPU KV events; Store
lookup, saving and loading are handled by the connector.

The vLLM KV-event parser admits only GPU Stored/Removed events to the router's
cache view; a missing/None medium remains GPU for older vLLM versions. CPU and
SSD events, including native CPU offload without Mooncake, are excluded and
logged at INFO. Other non-None media, including empty strings, are excluded
with a WARNING. Matching is case-insensitive. These events do not update GPU
hash/block-size state or populate CPU/SSD residency counts. Filtered events and
blocks are counted per node and medium, with a summary on the first event and
then at most every 30 seconds when more events arrive. These diagnostic totals
describe received events, not current CPU/SSD occupancy.
Backend Store operations and separately collected vLLM external-hit metrics
remain available. Malformed recognized events produce one WARNING summary per
batch, while valid events in that batch continue to be processed.

## Environment

Set up the model, dataset and task environment as described in the
[parent README](../README.md). Mooncake is optional and only needed when
`--mooncake-config-path` is supplied. This entrypoint uses the vLLM backend
(`--engine vllm`, the default).

The GPU experiments used these dependencies with upstream fixes:

| Dependency | Tested version |
| --- | --- |
| vLLM | `0.29.0` with [#55844](https://github.com/vllm-project/vllm/pull/55844) KV-event discovery support and [#54643](https://github.com/vllm-project/vllm/pull/54643) / [#54853](https://github.com/vllm-project/vllm/pull/54853) Store runtime backports |
| Mooncake | CUDA-enabled `0.3.13.post1` with the TCP pump fix from [#4063](https://github.com/kvcache-ai/Mooncake/pull/4063), commit `74d26b56` |


## Start the Store

Both modes use a master. The commands below assume one host with a shared network
namespace and free ports 50051/50053. Start the master in a separate terminal:

```bash
mooncake_master --port=50051 --enable_offload=false
```

Choose one mode and copy its JSON to `/absolute/path/mooncake.json`:

| Mode | Configuration | CPU pool ownership |
| --- | --- | --- |
| `standalone-store` | [standalone-cpu.example.json](standalone-cpu.example.json) | A separate owner process; capacity and lifetime are independent of model workers. |
| `embedded` | [embedded-cpu.example.json](embedded-cpu.example.json) | Each vLLM worker/rank contributes memory; its segment goes offline when that worker exits. |

**Standalone:** the JSON sets worker `global_segment_size=0`. Start an owner in
another terminal before inference; this example contributes 128 GiB:

```bash
mooncake_client \
  --host=127.0.0.1 --port=50053 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=P2PHANDSHAKE --protocol=tcp \
  --global_segment_size=137438953472 \
  --enable_offload=false
```

**Embedded:** start inference after the master; no separate owner is needed.
The JSON contributes 8 GiB **per rank**: three TP2 replicas contribute 48 GiB in
total. Both templates also allocate a separate 1 GiB private buffer per rank.
Adjust these sizes to the host's available memory; use reachable addresses in
the JSON and service commands if processes do not share a network namespace.

## Run inference

From the repository root, keep your working model/data/task arguments and add
the chosen config path. For example:

```bash
export PYTHONHASHSEED=42
bash examples/agent_aware_router/run_infer.sh \
  --model-path /absolute/path/to/model \
  --data-path /absolute/path/to/data.parquet \
  --task-config examples/agent_aware_router/task_config_mini_swe_agent.yaml \
  --device gpu --n-gpus-per-node 2 --tp 2 \
  --concurrency 16 --max-model-len 16384 \
  --kv-events \
  --mooncake-config-path /absolute/path/mooncake.json
```

Decode KV saving is disabled by default; use `--mooncake-save-decode-cache` to
enable it together with the config path. Omit `--mooncake-config-path` for the
normal inference path.

- Use the same model, TP, KV layout and hash settings across replicas sharing
  the Store, including the same `PYTHONHASHSEED` before Python starts.
- The config path is passed through the Ray job environment. The JSON must be
  readable at that same path on every worker; passing the path does not upload it.
- Startup logs list the explicitly forwarded environment variable names. Result
  metadata preserves the submitted connector settings captured before startup.
- Set any `MC_TCP_*` tuning in the actual worker and owner environments. This
  entry point does not automatically forward those variables to an existing
  Ray cluster. Check worker initialization logs for the effective configuration.
- Confirm successful PUT/GET operations when checking reuse. A lookup hit alone
  does not establish a successful load; the tested configuration uses
  `kv_load_failure_policy="fail"`, which terminates requests when KV loading fails.
  For fallback, set `kv_load_failure_policy="recompute"` in vLLM's
  `kv_transfer_config` to reschedule affected requests for local KV recomputation;
  this adds prefill work and may increase latency.

CPU caching can help when reusable prefixes have been evicted from GPU memory
or are reused across replicas. Size the pool for that reusable working set,
reserve memory for private buffers and other processes, and compare with a
Mooncake-disabled run under the same workload.

For additional configuration, see the
[vLLM connector documentation](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
and [Mooncake deployment guide](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html).
