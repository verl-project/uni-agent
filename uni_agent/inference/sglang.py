# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang 0.5.20 startup with unmodified verl generation and control methods.

The driver only imports verl's backend-neutral rollout interfaces. Each server
actor creates verl's SGLang adapter in its own runtime and supplies a native
SGLang Engine, bypassing the older dataclass-only server startup implementation.
"""

import asyncio
import logging
import os

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_visible_devices_keyword
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.utils import run_uvicorn

logger = logging.getLogger(__name__)


def _validate_inference_config(config, model_config):
    if config.disaggregation.enabled or config.pipeline_model_parallel_size != 1:
        raise ValueError("SGLang inference supports TP/DP replicas, without PD or pipeline parallelism")
    lora_rank = max(
        int(model_config.get("lora_rank", 0) or 0),
        int((model_config.get("lora") or {}).get("rank", 0) or 0),
    )
    if lora_rank > 0 or config.quantization is not None:
        raise ValueError("SGLang inference currently requires unquantized base-model weights without LoRA")
    if config.mtp is not None and config.mtp.enable and config.mtp.enable_rollout:
        raise ValueError("SGLang inference does not support MTP rollout")


def _engine_args(server):
    """Translate the standalone rollout config into native SGLang arguments."""
    config, model = server.config, server.model_config
    args = {
        "model_path": model.local_path,
        "tokenizer_path": model.local_tokenizer_path or model.tokenizer_path or model.local_path,
        "dtype": config.dtype,
        "random_seed": config.seed,
        "mem_fraction_static": config.gpu_memory_utilization,
        "disable_cuda_graph": config.enforce_eager,
        "enable_memory_saver": True,
        "enable_weights_cpu_backup": False,
        "base_gpu_id": server.base_gpu_id,
        "gpu_id_step": 1,
        "tp_size": config.tensor_model_parallel_size * config.data_parallel_size,
        "dp_size": config.data_parallel_size,
        "ep_size": config.expert_parallel_size,
        "node_rank": server.node_rank,
        "nnodes": server.nnodes,
        "load_format": config.load_format,
        "trust_remote_code": model.trust_remote_code,
        "max_running_requests": config.max_num_seqs,
        "log_level": "error",
        "skip_tokenizer_init": config.skip_tokenizer_init,
        "skip_server_warmup": True,
        "enable_return_routed_experts": config.enable_rollout_routing_replay,
    }
    if server.nnodes > 1:
        host = server._master_address
        host = f"[{host}]" if is_valid_ipv6_address(host) else host
        args["dist_init_addr"] = f"{host}:{server._master_port}"
    args.update(dict(config.engine_kwargs.get("sglang") or {}))
    return args


class SGLangInferenceServer:
    """Ray boundary around verl's unchanged token and lifecycle contracts."""

    def __init__(self, *, cuda_visible_devices, **kwargs):
        # Importing SGLang belongs to the actor, after its device selection.
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

        self._server = SGLangHttpServer(cuda_visible_devices=cuda_visible_devices, **kwargs)
        self._engine = None

    def get_master_address(self):
        return self._server.get_master_address()

    def get_server_address(self):
        return self._server.get_server_address()

    async def launch_server(self, master_address=None, master_port=None):
        import torch
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.entrypoints.http_server import ServerArgs, _GlobalState, app, set_global_state
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        from verl.utils.tracking import RLInsightLogger

        server = self._server
        if server.rollout_mode is not RolloutMode.STANDALONE:
            raise ValueError("The Uni-Agent SGLang adapter supports standalone inference only")
        if server.nnodes > 1 and server.node_rank != 0:
            if not master_address or master_port is None:
                raise ValueError("Non-master SGLang nodes require a distributed initialization address")
            server._master_address, server._master_port = master_address, master_port
        args = _engine_args(server)
        args.setdefault("attention_backend", "aiter" if torch.version.hip is not None else "flashinfer")
        if server.config.prometheus.enable or RLInsightLogger.enabled():
            args["enable_metrics"] = True
            if server.config.prometheus.served_model_name:
                args["served_model_name"] = server.config.prometheus.served_model_name.rsplit("/", 1)[-1]
        server_args = ServerArgs(**args)
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        try:
            self._engine = Engine(server_args=server_args)
            server.tokenizer_manager = self._engine.tokenizer_manager
            server.template_manager = self._engine.template_manager
            if server.node_rank != 0:
                return
            # 0.5.20 returns scheduler metadata through SchedulerInitResult;
            # the third _launch_subprocesses result is PortArgs, not metadata.
            scheduler_info = self._engine._scheduler_init_result.scheduler_infos[0]
            set_global_state(_GlobalState(server.tokenizer_manager, server.template_manager, scheduler_info))
            app.is_single_tokenizer_mode = True
            app.server_args = server_args
            app.warmup_thread_kwargs = {"server_args": server_args}
            if server_args.enable_metrics:
                from sglang.srt.utils.common import add_prometheus_middleware

                add_prometheus_middleware(app)
            server._server_port, server._server_task = await run_uvicorn(app, server_args, server._server_address)
            server.tokenizer_manager.server_status = ServerStatus.Up
        except BaseException:
            if self._engine is not None:
                self._engine.shutdown()
            raise

    async def generate(self, *, prompt_ids, sampling_params, request_id, **kwargs):
        return await self._server.generate(
            prompt_ids=prompt_ids, sampling_params=sampling_params, request_id=request_id, **kwargs
        )

    async def wake_up(self):
        return await self._server.wake_up()

    async def sleep(self):
        return await self._server.sleep()

    async def clear_kv_cache(self):
        return await self._server.clear_kv_cache()

    async def release_kv_cache(self):
        return await self._server.release_kv_cache()

    async def resume_kv_cache(self):
        return await self._server.resume_kv_cache()

    async def abort_all_requests(self, reject_request=False):
        if reject_request:
            logger.warning("SGLang pauses new requests until resume_generation; reject_request=True is not supported")
        return await self._server.abort_all_requests()

    async def resume_generation(self):
        return await self._server.resume_generation()

    async def set_global_steps(self, global_steps):
        return await self._server.set_global_steps(global_steps)

    async def start_profile(self, **kwargs):
        return await self._server.start_profile(**kwargs)

    async def stop_profile(self):
        return await self._server.stop_profile()


class SGLangInferenceReplica(RolloutReplica):
    """Keep verl placement and routing; launch the recipe's native server actor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_inference_config(self.config, self.model_config)
        self.server_class = ray.remote(SGLangInferenceServer)

    async def launch_servers(self):
        if self.rollout_mode is not RolloutMode.STANDALONE:
            raise ValueError("The Uni-Agent SGLang adapter supports standalone inference only")
        if len(self.workers) != self.world_size:
            raise ValueError("SGLang replica workers do not match the requested GPU count")
        visible_devices = get_visible_devices_keyword()
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices])
                )
                for worker in self.workers
            ]
        )
        base_gpu_id = 0
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices}"):
            base_gpu_id = (self.replica_rank * self.world_size) % self.gpus_per_node
        for node_rank in range(self.nnodes):
            first = node_rank * self.gpus_per_replica_node
            last = first + self.gpus_per_replica_node
            devices = sorted(
                {int(device) for _, ids in worker_infos[first:last] for device in ids.split(",") if device}
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=worker_infos[first][0], soft=False
                ),
                runtime_env={
                    "env_vars": {
                        **{var: "1" for var in get_platform().ray_noset_envvars()},
                        **get_platform().rollout_env_vars(),
                    }
                },
                name=f"sglang_server_{self.replica_rank}_{node_rank}{self.name_suffix}",
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=self.workers[first:last],
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=",".join(map(str, devices)),
                base_gpu_id=base_gpu_id,
            )
            self.servers.append(server)
        master_address, master_port = None, None
        if self.nnodes > 1:
            master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )
        host, port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"[{host}]:{port}" if is_valid_ipv6_address(host) else f"{host}:{port}"
