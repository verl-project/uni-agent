# LocalWiki Search Server

## Overview

LocalWiki Search Server is a high-performance semantic search service that provides two key API endpoints to simulate real-world search engine and crawler functionalities using the LocalWiki dataset.

## Installation

```bash
pip install -r requirements_lwiki.txt
```

## Dataset Preparation

### Source

We use [`Upstash/wikipedia-2024-06-bge-m3`](https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3),
which provides pre-computed BGE-M3 embeddings (1024-dim, L2-normalised) over
the June 2024 Wikipedia dump split into ~47M paragraph-level passages. Each
passage was encoded as `f"{title}\n{paragraph}"` with raw BGE-M3 (no
instruction prefix). Our query encoder (`shared_encoder.py`) is configured
to match: it uses raw BGE-M3 with no instruction prefix on the query side.

All paths default to `$DATA_ROOT` (default `~/uni_agent_data`). Override
the root by exporting `DATA_ROOT=/your/path` before running any of the scripts
below; all of `download.sh`, `ivf.py`, `ivf_cpu.py`, `preprocess.py`,
`wiki_ray.py` and `retrieval_server.py` honour it. Layout produced by these
scripts:

```
$DATA_ROOT/
├── wiki24-raw/data/en/*.parquet     # download.sh output
└── wiki24/
    ├── wiki24_faiss.index           # ivf.py / ivf_cpu.py output
    ├── wiki24_data.jsonl            # ivf.py / ivf_cpu.py output (text corpus)
    └── wiki24_preprocessed/
        ├── corpus.pkl               # preprocess.py output
        └── url_to_ids.pkl           # preprocess.py output
```

### Option 1: Download Pre-built Artifacts from HuggingFace (recommended)

A pre-built FAISS index and the matching corpus are mirrored at
[`begunner/wikipedia-2024-06-bge-m3-faiss-ivf`](https://huggingface.co/datasets/begunner/wikipedia-2024-06-bge-m3-faiss-ivf).
The repo contains:

- `wiki24_data.jsonl` (~25 GB) — text corpus, one passage per line.
- `wiki24_faiss.index.part00` .. `wiki24_faiss.index.part08` (9 × ~20 GB) —
  the FAISS IVF index, split because HF LFS caps single files at 50 GB.
  `cat` them back together to recover the original ~180 GB index.
- `preprocessed/corpus.pkl` (~24 GB) and `preprocessed/url_to_ids.pkl`
  (~615 MB) — preprocessed corpus used by the retrieval server for fast
  startup and URL-based lookup.

Build parameters baked into the index (must match the server / encoder
configuration in this repo): `METRIC_INNER_PRODUCT` (cosine), `NLIST=16384`,
`TRAINING_SAMPLES=4_000_000`, vector dim 1024 fp32, raw BGE-M3 query
encoding with **no instruction prefix**. Recommended runtime:
`FAISS_NPROBE=256`.

```bash
export DATA_ROOT=${HOME}/uni_agent_data

# 1. Download everything into $DATA_ROOT/wiki24/.
hf download begunner/wikipedia-2024-06-bge-m3-faiss-ivf \
    --repo-type dataset \
    --local-dir "$DATA_ROOT/wiki24"

# 2. Reassemble the split FAISS index (~180 GB single file).
cd "$DATA_ROOT/wiki24"
cat wiki24_faiss.index.part?? > wiki24_faiss.index
# Optional: free ~180 GB of disk by deleting the parts once the join is verified.
# rm wiki24_faiss.index.part??
cd -

# 3. Rename the preprocessed directory to match the layout this repo expects.
mv "$DATA_ROOT/wiki24/preprocessed" "$DATA_ROOT/wiki24/wiki24_preprocessed"
```

After step 3 the layout matches the diagram above and the server can be
started directly (see "Server Setup" below) — no need to run `ivf.py` or
`preprocess.py`.

### Option 2: Build from Scratch

```bash
export DATA_ROOT=${HOME}/uni_agent_data

# 1. Download raw Wikipedia 2024 parquet files with bge-m3 embeddings.
#    Output goes to $DATA_ROOT/wiki24-raw/data/en/.
./scripts/download.sh

# 2. Build the FAISS IVF index AND generate the JSONL corpus file.
#    Reads parquet from $DATA_ROOT/wiki24-raw/data/en/ and writes
#    $DATA_ROOT/wiki24/wiki24_faiss.index + $DATA_ROOT/wiki24/wiki24_data.jsonl.
python scripts/ivf.py        # GPU build - much faster on a multi-A100 node
# OR
python scripts/ivf_cpu.py    # CPU-only build - takes hours on 96 cores

# 3. Preprocess the JSONL corpus into pickle format for faster loading.
#    Writes $DATA_ROOT/wiki24/wiki24_preprocessed/{corpus,url_to_ids}.pkl.
python scripts/preprocess.py
```
**Note**: The program loads the entire dataset into memory during build,
requiring substantial RAM (>200GB for 47M x 1024 fp32). Modify the batching
in `scripts/ivf.py` if memory constraints exist. `scripts/preprocess.py`
converts the JSONL corpus into `corpus.pkl` and `url_to_ids.pkl`, which are
required by the retrieval server for efficient startup and URL-based lookup.

## Model for Retrieval

```bash
hf download BAAI/bge-m3 --local-dir ./bge-m3
```

## Server Setup

### Starting the Server

```bash
DATA_ROOT=... ./run_localwiki.sh
```

The script will:
1. Set up environment variables
2. Start Gunicorn with multiple workers
3. Load shared resources (FAISS index, corpus data) in the master process
4. Initialize GPU models in each worker process
5. Start the server on the specified port

### Configuration via Environment Variables

| Environment Variable | Description | Default Value |
|----------------------|-------------|---------------|
| `DATA_ROOT` | Root directory for all wiki artefacts. Used by `download.sh`, `ivf.py`, `ivf_cpu.py`, `preprocess.py`, `wiki_ray.py`, and `retrieval_server.py` as the default parent for raw downloads / `INDEX_PATH` / `CORPUS_PATH`. | `~/uni_agent_data` |
| `INDEX_PATH` | Path to the FAISS index file. | `${DATA_ROOT}/wiki24/wiki24_faiss.index` |
| `CORPUS_PATH` | Path to the preprocessed corpus directory (containing `corpus.pkl` and `url_to_ids.pkl`). | `${DATA_ROOT}/wiki24/wiki24_preprocessed/` |
| `RETRIEVER_MODEL` | Hugging Face model path for embedding generation | `BAAI/bge-m3` |
| `RETRIEVER_NAME` | Name/type of the retriever model. Must contain `bge-m3` to skip the BGE instruction-prefix path. | `bge-m3` |
| `TOPK` | Default number of results per query | `3` |
| `BATCH_SIZE` | Number of queries to process in each batch | `2048` |
| `MAX_REQUEST_BATCH_SIZE` | Maximum number of requests to batch at API level (One request may contain multiple queries) | `512` |
| `BATCH_TIMEOUT` | Maximum time (in seconds) to wait for requests batching | `0.01` |
| `FAISS_NPROBE` | Number of IVF cells inspected per query. Trade off recall vs latency: low values speed up search but make ranking unstable for paraphrased queries. | `256` |
<!-- | `FAISS_GPU` | Whether to use GPU for FAISS index operations | `False` | -->


## Usage Examples

### Example 1: Search by Query
```bash
curl -X POST "http://localhost:8001/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["What is the capital of France?", "What is Python?"], "topk": 3, "return_scores": true}'
```

### Example 2: Get Content by URL
```bash
curl -X POST "http://localhost:8001/crawl" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://en.wikipedia.org/wiki/Outline%20of%20France"}'
```

<!-- ## Main Features

### API Endpoints

#### 1. `/retrieve` (Search Engine API)
- **Purpose**: Simulates a search engine API that returns relevant documents based on query content
- **Input**: JSON object with `queries` (list of search queries), optional `topk` (number of results per query), and optional `return_scores` (boolean to include relevance scores)
- **Output**: For each query, returns topk search results with:
  - URL of the matching document
  - Summary text (chunks of the document from the wiki database, where a single URL's content may be split into multiple chunks)
  - Optional relevance scores

#### 2. `/crawl` (Crawler API)
- **Purpose**: Simulates a crawler API that retrieves full content for a specific URL
- **Input**: JSON object with `url` parameter
- **Output**: All text passages/chunks associated with the specified URL, effectively providing the full content -->

## Technical Optimizations

### 1. Request Batching
- **API Level Batching**: Combines multiple HTTP requests into batches using `MAX_REQUEST_BATCH_SIZE` (default: 512) and `BATCH_TIMEOUT` (default: 0.01s) to reduce overhead
- **Model Inference Batching**: Processes query embeddings in large batches (configurable via `BATCH_SIZE`, default: 2048) to optimize GPU utilization

### 2. Resource Sharing
- FAISS index and corpus data are loaded once by the master process and shared across all worker processes

### 3. Deduplication Handling
- **URL-based Deduplication**: Ensures that even when multiple chunks from the same URL appear in the topk results, only one chunk is returned for each URL
- **Result Sufficiency Guarantee**: To prevent having fewer results than the requested topk after deduplication, the system actually queries for more results upfront using a `SEARCH_FACTOR` multiplier (default: 2). This means it retrieves `topk * SEARCH_FACTOR` results initially, then performs deduplication.
