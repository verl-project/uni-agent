#!/usr/bin/env bash

python examples/inference/parallel_infer_api.py \
    --data-path "${DATA_PATH}" \
    --task-config examples/deepswe_v11/task_config.yaml \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY:-dummy}" \
    --model "${MODEL}" \
    --concurrency "${CONCURRENCY:-4}" \
    --log-dir "${LOG_DIR:-outputs/deepswe-v1.1/logs}" \
    --result-path "${RESULT_PATH:-outputs/deepswe-v1.1/results.json}"
