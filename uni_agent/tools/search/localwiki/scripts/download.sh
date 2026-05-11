#!/usr/bin/env bash
# Download Upstash BGE-M3 Wikipedia 2024 embeddings into the rebuild dir.
# Override DATA_ROOT (or WIKI_RAW_DIR for finer control) to write elsewhere.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-${HOME}/uni_agent_data}"
RAW_DIR="${WIKI_RAW_DIR:-${DATA_ROOT}/wiki24-raw}"

mkdir -p "${RAW_DIR}"

hf download \
    --repo-type dataset \
    Upstash/wikipedia-2024-06-bge-m3 \
    --include 'data/en/*' \
    --local-dir "${RAW_DIR}"
