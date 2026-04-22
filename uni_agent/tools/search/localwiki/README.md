# LocalWiki Search Server

## Overview

LocalWiki Search Server is a high-performance semantic search service that provides two key API endpoints to simulate real-world search engine and crawler functionalities using the LocalWiki dataset.

## Installation

```bash
pip install -r requirements_lwiki.txt
```

## Dataset Preparation

### Method 1: Download from ModelScope (Recommended)
```bash
# 1. Download pre-built FAISS index and JSONL corpus from ModelScope
modelscope download --dataset int040728/wiki24 --local_dir ./wiki24
cd wiki24

# 2. Concatenate split FAISS index parts into a single file
cat wiki24_faiss.index.part* > wiki24_faiss.index

# 3. Preprocess the JSONL corpus into pickle format for faster loading
DATA_ROOT=... python scripts/preprocess.py
```

### Method 2: Start from Scratch
```bash
# 1. Download raw Wikipedia 2024 parquet files with bge-m3 embeddings
./scripts/download.sh

# 2. Build the FAISS IVF index and generate the JSONL corpus file
python scripts/ivf.py  # or python scripts/ivf_cpu.py for CPU-only environments

# 3. Preprocess the JSONL corpus into pickle format for faster loading
DATA_ROOT=... python scripts/preprocess.py
```

**Note**: The program loads the entire dataset into memory, requiring substantial RAM. Modify `scripts/ivf.py` if memory constraints exist. The `scripts/preprocess.py` step converts the JSONL corpus into `corpus.pkl` and `url_to_ids.pkl`, which are required by the retrieval server for efficient startup and URL-based lookup.

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
| `INDEX_PATH` | Path to the FAISS index file | `wiki24_faiss.index` |
| `CORPUS_PATH` | Path to the corpus JSONL file | `wiki24_data.jsonl` |
| `RETRIEVER_MODEL` | Hugging Face model path for embedding generation | `BAAI/bge-m3` |
| `RETRIEVER_NAME` | Name/type of the retriever model | `bge-m3` |
| `TOPK` | Default number of results per query | `3` |
| `BATCH_SIZE` | Number of queries to process in each batch | `2048` |
| `MAX_REQUEST_BATCH_SIZE` | Maximum number of requests to batch at API level (One request may contain multiple queries) | `512` |
| `BATCH_TIMEOUT` | Maximum time (in seconds) to wait for requests batching | `0.01` |
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
curl -X POST "http://localhost:8001/search_by_url" \
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

#### 2. `/search_by_url` (Crawler API)
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
