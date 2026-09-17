# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""vLLMHttpServer subclass with per-replica kv-events port allocation.

With plain ``kv-events-config`` passthrough every replica receives the same
endpoint ports, and vLLM's ``ZmqEventPublisher`` only offsets ports by
``dp_rank`` *within* a replica — so multiple replicas on one node all try to
bind the same ports and crash at startup (``EADDRINUSE``). This subclass
allocates a free contiguous ``2*dp_size`` port block per replica via
:meth:`_assign_kv_events_ports` (triggered from ``_preprocess_engine_kwargs``
when ``enable_kv_cache_events`` is set), writes the resolved bases back into
``kv-events-config`` (the same dict later forwarded to vLLM as
``--kv-events-config``), and holds the block until just before vLLM binds it
(vLLM is the real binder; holding longer would fail its ZMQ bind).

The router discovers the resolved endpoints through the
``get_kv_events_endpoints`` actor getter (see ``balancer``/``collectors``).
"""

import argparse
import asyncio
import inspect
import logging
import os
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from uni_agent.agent_aware_router.server.net_utils import get_free_port_range
from verl.workers.rollout.utils import get_vision_placeholder_token_ids, run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import SuppressSignalInThread
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

logger = logging.getLogger(__name__)


class KvEventsHttpServer(vLLMHttpServer):
    """vLLMHttpServer plus kv-events port allocation and server getters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_events_endpoints = None
        # Sockets holding the kv-events port block; see _release_kv_events_ports.
        self._kv_events_socks: list[Any] = []

    def get_kv_events_endpoints(self):
        """Get kv-events ZMQ endpoint addresses.

        Returns list [endpoint, replay_endpoint, publisher, topic] or None.
        publisher/topic are None when not configured in kv-events-config.
        """
        return self._kv_events_endpoints

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        kv_events_config = engine_kwargs.get("kv-events-config")
        if kv_events_config and kv_events_config.get("enable_kv_cache_events", False):
            self._assign_kv_events_ports(kv_events_config)

    def _release_kv_events_ports(self) -> None:
        """Close the held kv-events sockets (idempotent).

        ``_assign_kv_events_ports`` held a ``2*dp_size`` block to keep neighbors
        free during launch; vLLM is the real binder, so the sockets must be
        closed before its ZMQ bind. Called from ``run_server`` (right before
        ``AsyncLLM.from_vllm_config``) and ``run_headless`` (whose node never
        binds them). The residual release→bind window can't be closed from
        outside vLLM; the upstream fix is for vLLM to allocate and report the
        ports itself.
        """
        for s in self._kv_events_socks:
            s.close()
        self._kv_events_socks = []

    def _assign_kv_events_ports(self, kv_events_config: dict) -> None:
        """Allocate one contiguous free port block for kv-events and record endpoints.

        vLLM's ``ZmqEventPublisher`` offsets each configured port by ``dp_rank``
        (``base_port + data_parallel_rank``), so ``endpoint`` and
        ``replay_endpoint`` each occupy ``dp_size`` consecutive ports. Reserve a
        single ``2*dp_size`` wide block via :func:`get_free_port_range` (bind +
        hold all neighbors so the block cannot be grabbed mid-launch), write the
        two bases into the config, and keep the holding sockets on
        ``self._kv_events_socks`` for :meth:`run_server` to release right before
        vLLM binds them — vLLM is the real binder, so holding longer would make
        its own bind fail with ``EADDRINUSE``.
        """
        dp_size = self.config.data_parallel_size
        parsed = []  # [(key, addr)] for tcp endpoints only
        for key, default in [("endpoint", "tcp://*:0"), ("replay_endpoint", "tcp://*:0")]:
            ep = kv_events_config.get(key, default)
            if "tcp" not in ep or ":" not in ep:
                continue
            colon = ep.rfind(":")
            parsed.append((key, ep[:colon]))
        if not parsed:
            return

        host = parsed[0][1].replace("tcp://", "").strip("[]") or "0.0.0.0"
        if "*" in host:
            host = "0.0.0.0"
        block = len(parsed) * dp_size
        base_port, socks = get_free_port_range(host, block, with_alive_sock=True)

        endpoints, assigned = [], []
        for idx, (key, addr) in enumerate(parsed):
            port = base_port + idx * dp_size
            kv_events_config[key] = f"{addr}:{port}"
            endpoints.append(f"{self._server_address}:{port}")
            assigned.append(f"{key}={addr}:{port}")
        self._kv_events_socks = socks
        logger.info(
            "kv-events ports assigned: replica_rank=%s dp_size=%s base=%s block=%s -> %s",
            self.replica_rank,
            dp_size,
            base_port,
            block,
            ", ".join(assigned),
        )
        # Report whatever the config actually carries; no defaults invented here
        # (None when unset, so the router can tell "not configured" apart).
        publisher = kv_events_config.get("publisher")
        topic = kv_events_config.get("topic")
        endpoints.extend([publisher, topic])

        self._kv_events_endpoints = endpoints

    # ``run_server`` / ``run_headless`` below are verbatim copies from verl's
    # vLLMHttpServer (base 3c62a15d) with a single insertion each: the
    # ``_release_kv_events_ports()`` call. verl offers no hook at that point, so
    # the methods must be copied wholesale — re-sync them when verl changes.

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        # Release the kv-events port reservation right before vLLM binds it.
        self._release_kv_events_ports()

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        # A sampled <|image_pad|>/<|video_pad|> has no image behind it, and every consumer of the
        # sequence assumes it does. Mask them out with the OOV tail, so the policy cannot pick one.
        await engine_client.collective_rpc(
            method="monkey_patch_model",
            kwargs={
                "vocab_size": len(self.model_config.tokenizer),
                "banned_token_ids": get_vision_placeholder_token_ids(self.model_config.processor),
            },
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        build_app_kwargs: dict[str, Any] = {}
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            build_app_kwargs["supported_tasks"] = supported_tasks
        # vLLM >= 0.20.0 requires `model_config` to register pooling API routes
        # (e.g. ``/classify``, ``/embed``); see
        # ``register_pooling_api_routers`` in vllm/entrypoints/pooling/factories.py
        # which short-circuits when ``model_config`` is ``None``.
        if "model_config" in build_app_sig.parameters:
            build_app_kwargs["model_config"] = engine_client.model_config
        app = build_app(args, **build_app_kwargs)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args.api_server_count = 0
        # Non-master nodes never start the vLLM engine that would bind these
        # ports; release the reservation so it isn't held for the actor's
        # lifetime (see _release_kv_events_ports).
        self._release_kv_events_ports()

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed with {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)
