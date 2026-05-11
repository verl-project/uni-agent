import asyncio
import logging
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import ray
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")
DEFAULT_RAY_NAMESPACE = os.getenv("RAY_NAMESPACE", "localwiki")
DEFAULT_ACTOR_NAME = os.getenv("SHARED_ENCODER_ACTOR_NAME", "localwiki-shared-encoder")
DEFAULT_BATCH_TIMEOUT_S = float(os.getenv("SHARED_ENCODER_BATCH_TIMEOUT", "0.01"))
DEFAULT_MAX_BATCH_SIZE = int(os.getenv("SHARED_ENCODER_MAX_BATCH_SIZE", "2048"))

logger = logging.getLogger(__name__)


def ensure_ray_initialized() -> None:
    if ray.is_initialized():
        return

    ray.init(
        address=DEFAULT_RAY_ADDRESS,
        namespace=DEFAULT_RAY_NAMESPACE,
        ignore_reinit_error=True,
    )


def load_model(model_path: str, use_fp16: bool = True):
    logger.info("Loading shared encoder model from %s...", model_path)
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    logger.info("Shared encoder model loaded and moved to GPU.")
    return model, tokenizer


def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask=None,
    pooling_method: str = "mean",
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]
        elif "bge-m3" in self.model_name.lower():
            # BGE-M3 does NOT use an instruction prefix on either side
            # (per BAAI/bge-m3 model card). Our corpus
            # (Upstash/wikipedia-2024-06-bge-m3) was encoded as
            # f"{title}\n{paragraph}" with raw BGE-M3, so the query side
            # must also avoid any instruction prefix to stay aligned.
            pass
        elif "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


@ray.remote(num_gpus=1, num_cpus=1, max_concurrency=1000)
class SharedEncoderActor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    ):
        self.encoder = Encoder(
            model_name=model_name,
            model_path=model_path,
            pooling_method=pooling_method,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_s
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: deque[tuple[list[str], bool, asyncio.Future]] = deque()
        self.batching_task = asyncio.get_event_loop().create_task(self._batching_loop())

    async def encode(self, query_list: list[str], is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        future = asyncio.get_running_loop().create_future()
        await self.request_queue.put((query_list, is_query, future))
        return await future

    async def _batching_loop(self) -> None:
        while True:
            query_list, is_query, future = await self._get_next_request()
            batch = [(query_list, future)]
            total_queries = len(query_list)
            batch_mode = is_query
            start_time = time.monotonic()

            while total_queries < self.max_batch_size and (time.monotonic() - start_time) < self.batch_timeout_s:
                try:
                    next_queries, next_is_query, next_future = await asyncio.wait_for(
                        self._get_next_request(),
                        timeout=self.batch_timeout_s - (time.monotonic() - start_time),
                    )
                except asyncio.TimeoutError:
                    break

                if next_is_query != batch_mode:
                    self.pending_requests.appendleft((next_queries, next_is_query, next_future))
                    break

                remaining_capacity = self.max_batch_size - total_queries
                if len(next_queries) <= remaining_capacity:
                    batch.append((next_queries, next_future))
                    total_queries += len(next_queries)
                    continue

                self.pending_requests.appendleft((next_queries, next_is_query, next_future))
                break

            flat_queries: list[str] = []
            request_sizes: list[int] = []
            request_futures: list[asyncio.Future] = []
            for queries, req_future in batch:
                flat_queries.extend(queries)
                request_sizes.append(len(queries))
                request_futures.append(req_future)

            try:
                embeddings = self.encoder.encode(query_list=flat_queries, is_query=batch_mode)
                start_idx = 0
                for req_future, req_size in zip(request_futures, request_sizes, strict=False):
                    req_future.set_result(embeddings[start_idx : start_idx + req_size])
                    start_idx += req_size
            except Exception as exc:
                logger.exception("Shared encoder batch failed: %s", exc)
                for req_future in request_futures:
                    req_future.set_exception(exc)

    async def _get_next_request(self):
        if self.pending_requests:
            return self.pending_requests.popleft()
        return await self.request_queue.get()


def get_shared_encoder_actor(actor_name: Optional[str] = None):
    ensure_ray_initialized()
    return ray.get_actor(actor_name or DEFAULT_ACTOR_NAME, namespace=DEFAULT_RAY_NAMESPACE)


def get_shared_encoder_actors(base_name: Optional[str] = None, num_replicas: int = 1) -> list:
    """Get shared encoder actor handle(s). Returns a list of actor handles."""
    ensure_ray_initialized()
    base = base_name or DEFAULT_ACTOR_NAME
    if num_replicas == 1:
        return [ray.get_actor(base, namespace=DEFAULT_RAY_NAMESPACE)]
    return [ray.get_actor(f"{base}-{i}", namespace=DEFAULT_RAY_NAMESPACE) for i in range(num_replicas)]
