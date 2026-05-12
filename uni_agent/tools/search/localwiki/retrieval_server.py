import asyncio
import functools
import logging
import os
import pickle
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, Optional

import faiss
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shared_encoder import get_shared_encoder_actors

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

GLOBAL_CORPUS_LINES = None
GLOBAL_CORPUS_DATA: list[dict[str, Any]] = None
GLOBAL_URL_TO_IDS: dict[str, list[int]] = {}
SHARED_FAISS_INDEX = None
SHARED_CONFIG = None
retriever = None

# app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Retrieval server starting up...")
    load_worker_resources()
    num_threads = max(1, (os.cpu_count() or 4) - 4)
    faiss.omp_set_num_threads(num_threads)
    logger.info("FAISS OpenMP threads set to %d.", faiss.omp_get_max_threads())
    asyncio.create_task(batching_worker())
    yield
    logger.info("Retrieval server shutting down.")


app = FastAPI(lifespan=lifespan)
request_queue = asyncio.Queue()
# Keep HTTP-layer batching lightweight so the shared encoder actor
# remains the main place where cross-worker requests are merged.
MAX_REQUEST_BATCH_SIZE = int(os.getenv("HTTP_MAX_REQUEST_BATCH_SIZE", "64"))
BATCH_TIMEOUT = float(os.getenv("HTTP_BATCH_TIMEOUT", "0.002"))


async def batching_worker():
    global SHARED_CONFIG
    config = SHARED_CONFIG

    while True:
        try:
            first_request_item = await asyncio.wait_for(request_queue.get(), BATCH_TIMEOUT)
        except asyncio.TimeoutError:
            continue

        batch = [first_request_item]
        reason = "timeout"
        start_time = time.time()
        while len(batch) < MAX_REQUEST_BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT:
            try:
                request_item = request_queue.get_nowait()
                batch.append(request_item)

                if len(batch) == MAX_REQUEST_BATCH_SIZE:
                    reason = "max_size"
                    break

            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

        if not batch:
            continue

        all_queries = []
        futures_info = []  # (future, num_queries, topk, return_scores)

        for req, future in batch:
            all_queries.extend(req.queries)
            topk = req.topk if req.topk is not None else config.retrieval_topk
            futures_info.append((future, len(req.queries), topk, req.return_scores))

        logger.debug(f"Batch Worker: Aggregated {len(batch)} requests into {len(all_queries)} total queries.")
        logger.debug(f"Batch Worker: Processing batch due to {reason}.")

        max_topk = max(info[2] for info in futures_info)

        try:
            loop = asyncio.get_running_loop()
            results, scores = await loop.run_in_executor(
                None,
                functools.partial(
                    retriever.batch_search,
                    query_list=all_queries,
                    num=max_topk,
                    return_score=True,
                ),
            )

            current_idx = 0
            for future, num_queries, topk, return_scores_flag in futures_info:
                req_results = results[current_idx : current_idx + num_queries]
                req_scores = scores[current_idx : current_idx + num_queries]
                current_idx += num_queries

                trimmed_results = [r[:topk] for r in req_results]
                trimmed_scores = [s[:topk] for s in req_scores]

                if return_scores_flag:
                    resp = []
                    for i, single_result in enumerate(trimmed_results):
                        combined = [
                            {"document": doc, "score": score}
                            for doc, score in zip(single_result, trimmed_scores[i], strict=False)
                        ]
                        resp.append(combined)
                else:
                    resp = trimmed_results

                if not future.done():
                    future.set_result({"result": resp})

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for future, _, _, _ in futures_info:
                if not future.done():
                    future.set_exception(e)


def load_all_corpus_lines(corpus_path: str):
    """load all corpus lines into memory and build URL to IDs index."""
    global GLOBAL_CORPUS_DATA
    global GLOBAL_URL_TO_IDS

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

    parsed_data = []
    url_to_ids_map = {}
    doc_count = 0

    # with open(corpus_path, 'rb') as f:
    #     for doc_index, line in enumerate(f):
    #         doc_count += 1
    #         try:
    #             doc = json.loads(line)
    #             parsed_data.append(doc)

    #             url = doc.get("url")
    #             if url:
    #                 if url not in url_to_ids_map:
    #                     url_to_ids_map[url] = []
    #                 url_to_ids_map[url].append(doc_index)

    #         except json.JSONDecodeError:
    #             warnings.warn("JSONDecodeError encountered, skipping line.", RuntimeWarning)
    #             parsed_data.append(None)
    corpus_file_path = os.path.join(os.path.dirname(corpus_path), "corpus.pkl")
    url_to_ids_file_path = os.path.join(os.path.dirname(corpus_path), "url_to_ids.pkl")
    logger.info(
        f"Loading ALL JSONL lines from {corpus_file_path} "
        f"({os.path.getsize(corpus_file_path) / (1024**3):.2f} GB) into RAM..."
    )
    with open(corpus_file_path, "rb") as f:
        parsed_data = pickle.load(f)
        doc_count = len(parsed_data)
    with open(url_to_ids_file_path, "rb") as f:
        url_to_ids_map = pickle.load(f)

    GLOBAL_CORPUS_DATA = parsed_data
    GLOBAL_URL_TO_IDS = url_to_ids_map
    logger.info("Corpus loaded.")
    logger.info(f"URL Index built. Total unique URLs: {len(GLOBAL_URL_TO_IDS):,}.")
    return doc_count


def load_docs(doc_idxs: list[int]) -> list[dict[str, Any]]:
    global GLOBAL_CORPUS_DATA

    if GLOBAL_CORPUS_DATA is None:
        raise RuntimeError("Corpus data not loaded. The server must be initialized correctly.")

    results = []
    max_len = len(GLOBAL_CORPUS_DATA)

    for idx in doc_idxs:
        if 0 <= idx < max_len:
            doc_data = GLOBAL_CORPUS_DATA[idx]
            if doc_data is not None:
                results.append(doc_data)
            else:
                warnings.warn(
                    f"Document index {idx} was skipped during loading due to JSON error.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                results.append({"error": f"JSON decode error at index {idx}"})
        else:
            warnings.warn(f"Invalid document index: {idx}", RuntimeWarning, stacklevel=2)
            results.append({"error": f"Invalid doc index {idx}"})
    return results


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)

        global SHARED_FAISS_INDEX
        if SHARED_FAISS_INDEX is None:
            raise RuntimeError("FAISS index not loaded. The server must be initialized correctly.")
        self.index = SHARED_FAISS_INDEX

        self.encoder_actors = get_shared_encoder_actors(
            config.shared_encoder_actor_name,
            num_replicas=config.num_encoder_gpus,
        )
        self._encoder_idx = 0
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        logger.info(
            "Retriever initialized with %d encoder actor(s).",
            len(self.encoder_actors),
        )

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []

        SEARCH_FACTOR = 2
        k_prime = max(num * SEARCH_FACTOR, 6)

        for start_idx in range(0, len(query_list), self.batch_size):
            t_batch_start = time.time()
            query_batch = query_list[start_idx : start_idx + self.batch_size]

            # encode (round-robin across encoder actors)
            t_encode_start = time.time()
            encoder_actor = self.encoder_actors[self._encoder_idx]
            self._encoder_idx = (self._encoder_idx + 1) % len(self.encoder_actors)
            batch_emb = ray.get(encoder_actor.encode.remote(query_batch))
            t_encode_end = time.time()
            encode_time = t_encode_end - t_encode_start

            # FAISS search
            t_faiss_start = time.time()
            batch_scores, batch_idxs = self.index.search(batch_emb, k=k_prime)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            t_faiss_end = time.time()
            faiss_time = t_faiss_end - t_faiss_start

            # load
            flat_idxs = sum(batch_idxs, [])
            t_load_start = time.time()
            batch_results = load_docs(flat_idxs)
            t_load_end = time.time()
            load_time = t_load_end - t_load_start

            # post proccess
            t_post_start = time.time()
            batch_results = [batch_results[i * k_prime : (i + 1) * k_prime] for i in range(len(batch_idxs))]

            final_batch_results = []
            final_batch_scores = []

            for docs_prime, scores_prime in zip(batch_results, batch_scores, strict=False):
                unique_results_for_query = []
                unique_scores_for_query = []
                seen_urls = set()

                for doc, score in zip(docs_prime, scores_prime, strict=False):
                    doc_url = doc.get("url")

                    if doc_url and doc_url not in seen_urls:
                        unique_results_for_query.append(doc)
                        unique_scores_for_query.append(score)
                        seen_urls.add(doc_url)

                    if len(unique_results_for_query) >= num:
                        break

                final_batch_results.append(unique_results_for_query)
                final_batch_scores.append(unique_scores_for_query)

            results.extend(final_batch_results)
            scores.extend(final_batch_scores)

            t_post_end = time.time()
            post_process_time = t_post_end - t_post_start

            batch_total_time = t_post_end - t_batch_start
            logger.debug(
                f"Batch completed (size: {len(query_batch)}): "
                f"Total: {batch_total_time:.4f}s | "
                f"Encode: {encode_time:.4f}s | "
                f"Faiss: {faiss_time:.4f}s | "
                f"Load Docs: {load_time:.4f}s | "
                f"Post-Process: {post_process_time:.4f}s"
            )

            del (
                batch_emb,
                batch_scores,
                batch_idxs,
                query_batch,
                flat_idxs,
                batch_results,
                final_batch_results,
                final_batch_scores,
            )

        if return_score:
            return results, scores
        else:
            return results, None


def get_retriever(config):
    return DenseRetriever(config)


class Config:
    def __init__(self):
        self.index_path = os.getenv("INDEX_PATH", "wiki24_faiss.index")
        self.corpus_path = os.getenv("CORPUS_PATH", "wiki24_data.jsonl")
        self.retrieval_model_path = os.getenv("RETRIEVER_MODEL", "BAAI/bge-m3")
        self.retrieval_method = os.getenv("RETRIEVER_NAME", "bge-m3")
        self.retrieval_topk = int(os.getenv("TOPK", 3))
        self.retrieval_batch_size = int(os.getenv("BATCH_SIZE", "512"))
        self.shared_encoder_actor_name = os.getenv("SHARED_ENCODER_ACTOR_NAME", "localwiki-shared-encoder")
        self.num_encoder_gpus = int(os.getenv("NUM_ENCODER_GPUS", "1"))

        self.retrieval_pooling_method = "mean"
        self.retrieval_query_max_length = 512
        self.retrieval_use_fp16 = True


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class URLRequest(BaseModel):
    urls: list[str]


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    future = asyncio.Future()

    try:
        await request_queue.put((request, future))
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/crawl")
def crawl_endpoint(request: URLRequest):
    """
    Batch endpoint: accepts a list of URLs and returns passages for each one
    using O(1) hash map lookups.
    """
    global GLOBAL_CORPUS_DATA
    global GLOBAL_URL_TO_IDS

    if GLOBAL_URL_TO_IDS is None:
        raise HTTPException(status_code=500, detail="Corpus URL Index not built. Check server initialization.")

    results = []
    for target_url in request.urls:
        doc_indices = GLOBAL_URL_TO_IDS.get(target_url, [])
        if not doc_indices:
            results.append({"url": target_url, "result": f"No passages found for URL: {target_url}", "texts": []})
            continue

        matching_texts = []
        for idx in doc_indices:
            doc = GLOBAL_CORPUS_DATA[idx]
            if doc:
                matching_texts.append(doc.get("text", "Text field missing"))

        results.append(
            {
                "url": target_url,
                "result": f"Found {len(matching_texts)} passages for URL: {target_url}",
                "texts": matching_texts,
            }
        )

    return {"results": results}


def load_worker_resources():
    """
    Initialize the DenseRetriever: connects to the shared Ray encoder
    actor(s) and links with the pre-loaded FAISS index / corpus data.
    Safe to call multiple times (idempotent).
    """
    global retriever
    if retriever is not None:
        return

    config = SHARED_CONFIG
    if config is None:
        raise RuntimeError("Configuration not loaded.")

    logger.info("Connecting to shared encoder actor(s) and loading worker resources...")
    retriever = get_retriever(config)
    logger.info("DenseRetriever fully initialized.")


def load_shared_resources():
    """
    MASTER PROCESS ONLY:
    1. Loads Config.
    2. Loads FAISS Index (CPU).
    3. Loads Corpus data (GLOBAL_CORPUS_LINES, GLOBAL_URL_TO_IDS).
    These resources are shared among workers via Copy-on-Write (COW).
    """
    global SHARED_CONFIG
    global SHARED_FAISS_INDEX
    global GLOBAL_CORPUS_LINES

    logger.info("Master: Starting to load shared CPU resources...")

    # 1. Load config
    if SHARED_CONFIG is None:
        SHARED_CONFIG = Config()
        logger.info("Master: Configuration loaded.")

    # 2. Initialize global corpus lines
    if GLOBAL_CORPUS_LINES is None:
        GLOBAL_CORPUS_LINES = load_all_corpus_lines(SHARED_CONFIG.corpus_path)
        logger.info(f"Master: Corpus lines loaded. Total documents: {GLOBAL_CORPUS_LINES:,}.")

    # 3. Load FAISS index into CPU memory
    index_path = SHARED_CONFIG.index_path
    if SHARED_FAISS_INDEX is None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        SHARED_FAISS_INDEX = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        SHARED_FAISS_INDEX.nprobe = int(os.getenv("FAISS_NPROBE", 256))
        logger.info(
            f"Master: FAISS index loaded (CPU). NTotal: {SHARED_FAISS_INDEX.ntotal:,}, "
            f"nprobe={SHARED_FAISS_INDEX.nprobe}."
        )

    logger.info("Master: Shared CPU resources initialization complete.")


load_shared_resources()
